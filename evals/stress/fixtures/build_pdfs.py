"""Generate deterministic, text-extractable PDFs for synthetic stress tests."""

import argparse
from io import BytesIO
from pathlib import Path
import sys
import zlib

from pypdf import PdfReader


PAGE_COUNTS = {5: 2, 20: 8, 50: 20, 100: 20}
LINE = 'SYNTHETIC STRESS FIXTURE - scheduling requirements and project details.'


def _content_stream(page_number):
    # Fixed content and layout; no timestamps, random identifiers, or input data.
    lines = [f'{LINE} Page {page_number:02d} line {line:02d}.' for line in range(1, 34)]
    commands = ['BT', '/F1 9 Tf', '12 TL', '40 750 Td']
    commands.extend(f'({line}) Tj T*' for line in lines)
    commands.append('ET')
    return zlib.compress('\n'.join(commands).encode('ascii'), level=9)


def _pdf_bytes(size):
    """Build a minimal PDF with compressed text streams and a valid xref table."""
    page_count = PAGE_COUNTS[size]
    kids = ' '.join(f'{4 + 2 * i} 0 R' for i in range(page_count))
    objects = [
        b'<< /Type /Catalog /Pages 2 0 R >>',
        f'<< /Type /Pages /Kids [{kids}] /Count {page_count} >>'.encode('ascii'),
        b'<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>',
    ]
    for i in range(page_count):
        stream = _content_stream(i + 1)
        objects.append(
            f'<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] '
            f'/Resources << /Font << /F1 3 0 R >> >> /Contents {5 + 2 * i} 0 R >>'.encode('ascii')
        )
        objects.append(
            f'<< /Length {len(stream)} /Filter /FlateDecode >>\nstream\n'.encode('ascii')
            + stream + b'\nendstream'
        )

    pdf = bytearray(b'%PDF-1.4\n%\xe2\xe3\xcf\xd3\n')
    offsets = []
    for number, obj in enumerate(objects, 1):
        offsets.append(len(pdf))
        pdf.extend(f'{number} 0 obj\n'.encode('ascii') + obj + b'\nendobj\n')
    xref = bytearray(f'xref\n0 {len(objects) + 1}\n0000000000 65535 f \n'.encode('ascii'))
    for offset in offsets:
        xref.extend(f'{offset:010d} 00000 n \n'.encode('ascii'))

    def trailer(offset):
        return (f'trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n'
                f'startxref\n{offset}\n%%EOF\n').encode('ascii')

    # Put the padding comment before xref, not between startxref and EOF.
    # Account for the number of digits in the resulting xref offset.
    padding = size * 1024 - len(pdf) - len(xref) - len(trailer(size * 1024))
    if padding < 2:
        raise ValueError('PDF content exceeds calibrated size')
    pdf.extend(b'%' + b' ' * (padding - 2) + b'\n')
    xref_offset = len(pdf)
    pdf.extend(xref + trailer(xref_offset))
    assert len(pdf) == size * 1024
    return bytes(pdf)


def main(argv=None):
    parser = argparse.ArgumentParser(description='Generate deterministic synthetic PDF fixtures.')
    parser.add_argument('--size', required=True, type=int, choices=(0, *PAGE_COUNTS),
                        help='Size in KiB: 0 means no attachment; otherwise 5,20,50,100.')
    parser.add_argument('--output-dir', required=True, type=Path,
                        help='Existing directory for the PDF (not used for size 0).')
    args = parser.parse_args(argv)
    if args.size == 0:
        print('0 KB: no attachment (0 bytes, 0 extracted characters)')
        return 0

    data = _pdf_bytes(args.size)
    characters = sum(len(page.extract_text()) for page in PdfReader(BytesIO(data)).pages)
    target = args.output_dir / f'stress-{args.size}kb.pdf'
    try:
        with target.open('xb') as output:
            output.write(data)
    except FileExistsError:
        print(f'Error: output already exists: {target}', file=sys.stderr)
        return 1
    except OSError as exc:
        print(f'Error: cannot write {target}: {exc.strerror}', file=sys.stderr)
        return 1
    print(f'{args.size} KB: {target} ({len(data)} bytes, {characters} extracted characters)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
