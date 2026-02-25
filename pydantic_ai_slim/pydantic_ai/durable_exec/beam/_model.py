from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any

from pydantic_ai import (
    ModelMessage,
    ModelResponse,
    ModelResponseStreamEvent,
)
from pydantic_ai.models import Model, ModelRequestParameters, StreamedResponse
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import RunContext
from pydantic_ai.usage import RequestUsage

from ._checkpoint import (
    StepCounter,
    deserialize_model_response,
    get_checkpoint,
    serialize_model_response,
    set_checkpoint,
)
from ._erlang import in_beam_worker


class BEAMStreamedResponse(StreamedResponse):
    """A pre-computed streamed response wrapping an already-complete `ModelResponse`."""

    def __init__(self, model_request_parameters: ModelRequestParameters, response: ModelResponse):
        super().__init__(model_request_parameters)
        self.response = response

    async def _get_event_iterator(self) -> AsyncIterator[ModelResponseStreamEvent]:
        return
        # noinspection PyUnreachableCode
        yield

    def get(self) -> ModelResponse:
        return self.response

    def usage(self) -> RequestUsage:
        return self.response.usage  # pragma: no cover

    @property
    def model_name(self) -> str:
        return self.response.model_name or ''  # pragma: no cover

    @property
    def provider_name(self) -> str:
        return self.response.provider_name or ''  # pragma: no cover

    @property
    def provider_url(self) -> str | None:
        return self.response.provider_url  # pragma: no cover

    @property
    def timestamp(self) -> datetime:
        return self.response.timestamp  # pragma: no cover


class BEAMModel(WrapperModel):
    """A wrapper for Model that integrates with BEAM, checkpointing model requests to ETS for durable replay."""

    def __init__(
        self,
        model: Model,
        *,
        workflow_id: str,
        step_counter: StepCounter,
    ):
        super().__init__(model)
        self._workflow_id = workflow_id
        self._step_counter = step_counter

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        if not in_beam_worker():
            return await super().request(messages, model_settings, model_request_parameters)

        step_id = self._step_counter.next('model.request')

        cached = get_checkpoint(self._workflow_id, step_id)
        if cached is not None:
            return deserialize_model_response(cached)

        response = await super().request(messages, model_settings, model_request_parameters)
        set_checkpoint(self._workflow_id, step_id, serialize_model_response(response))
        return response

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: RunContext[Any] | None = None,
    ) -> AsyncIterator[StreamedResponse]:
        if not in_beam_worker():
            async with super().request_stream(
                messages, model_settings, model_request_parameters, run_context
            ) as streamed_response:
                yield streamed_response
                return

        step_id = self._step_counter.next('model.request_stream')

        cached = get_checkpoint(self._workflow_id, step_id)
        if cached is not None:
            yield BEAMStreamedResponse(model_request_parameters, deserialize_model_response(cached))
            return

        # Yield the real stream for real-time tokens; checkpoint after the caller finishes consuming it.
        async with super().request_stream(
            messages, model_settings, model_request_parameters, run_context
        ) as streamed_response:
            yield streamed_response
            # Caller has finished iterating — checkpoint the complete response.
            response = streamed_response.get()
            set_checkpoint(self._workflow_id, step_id, serialize_model_response(response))
