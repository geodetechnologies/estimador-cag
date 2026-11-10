"""Extract text locally from supported document attachments."""

from io import BytesIO
from pathlib import PurePath

from docx import Document
from docx.table import Table
from fastapi import UploadFile
from pypdf import PdfReader

MAX_ATTACHMENT_BYTES = 10 * 1024 * 1024


class AttachmentError(ValueError):
    """A document is unsupported, too large, or cannot be read."""


def _docx_text(container) -> str:
    parts = []
    for block in container.iter_inner_content():
        if isinstance(block, Table):
            for row in block.rows:
                for cell in row.cells:
                    parts.append(_docx_text(cell))
        else:
            parts.append(block.text)
    return '\n'.join(parts)


async def enrich_transcript(transcript: str, attachments: list[UploadFile]) -> str:
    parts = [transcript]
    for upload in attachments:
        filename = PurePath((upload.filename or '').replace('\\', '/')).name
        suffix = PurePath(filename).suffix.lower()
        if suffix not in {'.pdf', '.docx'} or '\n' in filename or '\r' in filename:
            raise AttachmentError('Unsupported attachment; use PDF or DOCX.')
        data = await upload.read(MAX_ATTACHMENT_BYTES + 1)
        if len(data) > MAX_ATTACHMENT_BYTES:
            raise AttachmentError('Attachment exceeds the 10 MiB limit.')
        try:
            if suffix == '.pdf':
                text = '\n'.join(page.extract_text() or '' for page in PdfReader(BytesIO(data)).pages)
            else:
                text = _docx_text(Document(BytesIO(data)))
        except Exception:
            raise AttachmentError('Could not extract attachment text.') from None
        parts.append(f'--- attachment: {filename} ---\n{text}')
    return '\n\n'.join(parts)
