from __future__ import annotations

import json
from collections import defaultdict
from typing import Any

import pydantic

from pydantic_ai.messages import ModelResponse

from ._erlang import erlang_call, get_erlang

_model_response_ta = pydantic.TypeAdapter(ModelResponse)


class StepCounter:
    """Deterministic step ID generator for checkpoint replay.

    The agent loop is deterministic given the same model/tool responses,
    so replaying produces the same step IDs in the same order.
    """

    def __init__(self) -> None:
        self._counts: dict[str, int] = defaultdict(int)

    def next(self, category: str) -> str:
        """Return the next step ID for the given category."""
        count = self._counts[category]
        self._counts[category] = count + 1
        return f'{category}:{count}'

    def reset(self) -> None:
        """Clear all counters."""
        self._counts.clear()


# --- Checkpoint operations (backed by ETS via erlang_python's built-in state API) ---


def get_checkpoint(workflow_id: str, step_id: str) -> bytes | None:
    """Read a checkpoint from ETS via erlang_python's built-in state API."""
    erl = get_erlang()
    return erl.state_get(f'beam_ckpt:{workflow_id}:{step_id}')


def set_checkpoint(workflow_id: str, step_id: str, data: bytes) -> None:
    """Write a checkpoint to ETS."""
    erl = get_erlang()
    erl.state_set(f'beam_ckpt:{workflow_id}:{step_id}', data)


def start_workflow(workflow_id: str, agent_name: str) -> None:
    """Register a workflow as running in ETS."""
    erl = get_erlang()
    erl.state_set(f'beam_wf:{workflow_id}', {'name': agent_name, 'status': 'running'})


def complete_workflow(workflow_id: str) -> None:
    """Mark a workflow as completed and clean up its checkpoints."""
    erl = get_erlang()
    erl.state_set(f'beam_wf:{workflow_id}', {'status': 'completed'})
    # Clean up checkpoint keys
    keys = erl.state_keys(f'beam_ckpt:{workflow_id}:')
    for key in keys:
        erl.state_delete(key)


def publish_event(workflow_id: str, event_data: Any) -> None:
    """Publish an event to subscribers via a registered Erlang function."""
    erlang_call('beam_event_publish', workflow_id, event_data)


# --- Serialization helpers ---


def serialize_model_response(response: ModelResponse) -> bytes:
    """Serialize a `ModelResponse` to bytes using Pydantic's TypeAdapter."""
    return _model_response_ta.dump_json(response)


def deserialize_model_response(data: bytes) -> ModelResponse:
    """Deserialize bytes back into a `ModelResponse`."""
    return _model_response_ta.validate_json(data)


def serialize_tool_result(result: Any) -> bytes:
    """Serialize a tool result to JSON bytes."""
    return json.dumps(result).encode('utf-8')


def deserialize_tool_result(data: bytes) -> Any:
    """Deserialize JSON bytes back into a tool result."""
    return json.loads(data)
