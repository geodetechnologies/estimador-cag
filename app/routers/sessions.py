"""Session creation and atomic, structured conversational estimates."""

import json
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import ValidationError

from app.config import Settings, get_settings
from app.prompts.loader import render_metadata_prompt, render_session_estimation_system
from app.services import llm_service
from app.services.attachments import AttachmentError, enrich_transcript
from app.sessions import ProjectMetadata, create_session, get_session

router = APIRouter(prefix='/sessions', tags=['sessions'])


@router.post('')
async def create(settings: Annotated[Settings, Depends(get_settings)]) -> dict[str, str]:
    return {'session_id': create_session(settings.max_turns)}


@router.post('/{session_id}/estimate')
async def estimate_session(
    session_id: str,
    settings: Annotated[Settings, Depends(get_settings)],
    transcript: Annotated[str, Form(min_length=1)],
    attachments: Annotated[list[UploadFile] | None, File()] = None,
) -> dict:
    session = get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail='Session not found.')
    if not transcript.strip():
        raise HTTPException(status_code=422, detail='Transcript must not be blank.')
    try:
        user = await enrich_transcript(transcript, attachments or [])
    except AttachmentError as error:
        raise HTTPException(status_code=422, detail=str(error)) from None

    async with session.lock:
        system = render_session_estimation_system(session.metadata)
        history = session.history.to_messages_list(system)[1:]
        previous_turn_limit = session.history.max_turns - 1
        history = history[-2 * previous_turn_limit:] if previous_turn_limit else []
        try:
            result = await llm_service.structured_estimate(user, settings, system=system, history=history)
            result_json = result.model_dump_json()
            extractor_input = json.dumps({'previous_metadata': session.metadata.model_dump(),
                                          'user': user, 'assistant': result.model_dump()}, ensure_ascii=False)
            extracted = await llm_service.estimate(extractor_input, settings,
                system=render_metadata_prompt(), json_output=True)
            try:
                metadata = ProjectMetadata.model_validate_json(extracted)
                if metadata.model_fields_set != set(ProjectMetadata.model_fields):
                    raise llm_service.EstimationError(502, 'Estimation provider returned an invalid result.')
            except ValidationError:
                raise llm_service.EstimationError(502, 'Estimation provider returned an invalid result.') from None
        except llm_service.EstimationError as error:
            raise HTTPException(status_code=error.status_code, detail=error.detail) from None
        session.history.add_turn(user, result_json)
        session.metadata = metadata
        return {'result': result.model_dump(), 'prompt_version': 'v1', 'cached': False,
                'project_metadata': metadata.model_dump()}
