from __future__ import annotations

import json
import re
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from pydantic_ai import Agent
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import RunContext
from pydantic_ai.toolsets.function import FunctionToolset
from pydantic_ai.usage import RequestUsage, RunUsage

try:
    from pydantic_ai.durable_exec.beam import BEAMAgent, BEAMModel
    from pydantic_ai.durable_exec.beam._checkpoint import (
        StepCounter,
        deserialize_model_response,
        deserialize_tool_result,
        serialize_model_response,
        serialize_tool_result,
    )
    from pydantic_ai.durable_exec.beam._erlang import (
        BEAMNotAvailableError,
        erlang_call,
        in_beam_worker,
    )
    from pydantic_ai.durable_exec.beam._function_toolset import BEAMFunctionToolset
    from pydantic_ai.durable_exec.beam._toolset import beamify_toolset
except ImportError:  # pragma: lax no cover
    pytest.skip('BEAM integration not fully available', allow_module_level=True)

pytestmark = [
    pytest.mark.anyio,
    pytest.mark.xdist_group(name='beam'),
]

# --- Mock helpers ---

# Patch targets: use the module where the name is *looked up*, not where it's defined.
CHECKPOINT_MODULE = 'pydantic_ai.durable_exec.beam._checkpoint'
MODEL_MODULE = 'pydantic_ai.durable_exec.beam._model'
FUNCTION_TOOLSET_MODULE = 'pydantic_ai.durable_exec.beam._function_toolset'
AGENT_MODULE = 'pydantic_ai.durable_exec.beam._agent'


def make_mock_erlang() -> tuple[dict[str, Any], MagicMock]:
    """Create a mock ``erlang`` module backed by an in-memory dict.

    The mock provides the ``state_get``/``state_set``/``state_delete``/
    ``state_keys`` helpers (ETS state API) as well as ``call`` (for
    ``publish_event`` and similar registered Erlang functions).
    """
    store: dict[str, Any] = {}
    mock_erl = MagicMock()

    def state_get(key: str) -> Any:
        return store.get(key)

    def state_set(key: str, value: Any) -> None:
        store[key] = value

    def state_delete(key: str) -> Any:
        return store.pop(key, None)

    def state_keys(prefix: str = '') -> list[str]:
        return [k for k in store if k.startswith(prefix)]

    def call(fn: str, *args: Any) -> None:
        pass

    mock_erl.state_get = state_get
    mock_erl.state_set = state_set
    mock_erl.state_delete = state_delete
    mock_erl.state_keys = state_keys
    mock_erl.call = call

    return store, mock_erl


# --- 1. TestStepCounter ---


class TestStepCounter:
    def test_deterministic_ids(self) -> None:
        counter = StepCounter()
        assert counter.next('model.request') == 'model.request:0'
        assert counter.next('model.request') == 'model.request:1'
        assert counter.next('tool.search') == 'tool.search:0'
        assert counter.next('model.request') == 'model.request:2'

    def test_reset(self) -> None:
        counter = StepCounter()
        counter.next('model.request')
        counter.next('model.request')
        counter.reset()
        assert counter.next('model.request') == 'model.request:0'

    def test_independent_categories(self) -> None:
        counter = StepCounter()
        assert counter.next('model.request') == 'model.request:0'
        assert counter.next('tool.search') == 'tool.search:0'
        assert counter.next('model.request_stream') == 'model.request_stream:0'
        assert counter.next('model.request') == 'model.request:1'
        assert counter.next('tool.search') == 'tool.search:1'


# --- 2. TestBEAMNotAvailableError ---


class TestBEAMNotAvailableError:
    def test_not_in_beam_worker(self) -> None:
        # Outside BEAM, the `erlang` module is not importable.
        assert in_beam_worker() is False

    def test_erlang_call_raises_outside_beam(self) -> None:
        with pytest.raises(
            BEAMNotAvailableError,
            match=re.escape('The `erlang` module is not available.'),
        ):
            erlang_call('beam_event_publish', 'wf-1', {'event': 'test'})

    def test_is_user_error_subclass(self) -> None:
        assert issubclass(BEAMNotAvailableError, UserError)


# --- 3. TestBEAMAgentValidation ---


class TestBEAMAgentValidation:
    def test_name_required(self) -> None:
        with pytest.raises(
            UserError,
            match=re.escape('An agent needs to have a unique `name` in order to be used with BEAM.'),
        ):
            BEAMAgent(Agent())

    def test_model_required(self) -> None:
        with pytest.raises(
            UserError,
            match=re.escape('An agent needs to have a `model` in order to be used with BEAM'),
        ):
            BEAMAgent(Agent(name='test_agent'))

    def test_valid_creation(self) -> None:
        agent = Agent(name='valid_agent', model=TestModel())
        beam_agent = BEAMAgent(agent)
        assert beam_agent.name == 'valid_agent'


# --- 4. TestBEAMAgentOutsideBEAM ---


class TestBEAMAgentOutsideBEAM:
    async def test_run_delegates_outside_beam(self) -> None:
        """When `in_beam_worker()` returns False, `beam_agent.run()` delegates to the wrapped agent."""
        agent = Agent(name='delegate_agent', model=TestModel())
        beam_agent = BEAMAgent(agent)

        with patch(f'{AGENT_MODULE}.in_beam_worker', return_value=False):
            result = await beam_agent.run('Hello')
        assert isinstance(result.output, str)


# --- 5. TestBEAMCheckpointing ---


class TestBEAMCheckpointing:
    async def test_model_request_checkpointed(self) -> None:
        """First run stores checkpoint via set_checkpoint."""
        store, mock_erl = make_mock_erlang()
        model = TestModel()
        step_counter = StepCounter()
        beam_model = BEAMModel(model, workflow_id='wf-1', step_counter=step_counter)

        with (
            patch(f'{MODEL_MODULE}.in_beam_worker', return_value=True),
            patch(f'{CHECKPOINT_MODULE}.get_erlang', return_value=mock_erl),
        ):
            response = await beam_model.request([], None, ModelRequestParameters())

        assert isinstance(response, ModelResponse)
        # Verify checkpoint was stored
        assert len(store) == 1
        key = next(iter(store))
        assert 'model.request:0' in key

    async def test_model_request_replayed(self) -> None:
        """Pre-populated checkpoint is returned without calling the real model."""
        store, mock_erl = make_mock_erlang()

        # Pre-populate the checkpoint
        original_response = ModelResponse(
            parts=[TextPart(content='cached answer')],
            usage=RequestUsage(),
            model_name='test',
        )
        store['beam_ckpt:wf-2:model.request:0'] = serialize_model_response(original_response)

        model = TestModel()
        step_counter = StepCounter()
        beam_model = BEAMModel(model, workflow_id='wf-2', step_counter=step_counter)

        with (
            patch(f'{MODEL_MODULE}.in_beam_worker', return_value=True),
            patch(f'{CHECKPOINT_MODULE}.get_erlang', return_value=mock_erl),
            patch.object(model, 'request', wraps=model.request) as spy,
        ):
            response = await beam_model.request([], None, ModelRequestParameters())

        # Real model should NOT have been called
        spy.assert_not_called()
        assert response.parts[0].content == 'cached answer'  # type: ignore[union-attr]

    async def test_tool_call_checkpointed(self) -> None:
        """Tool calls are checkpointed to ETS."""
        store, mock_erl = make_mock_erlang()
        step_counter = StepCounter()
        model = TestModel()

        toolset = FunctionToolset[None]()

        @toolset.tool
        async def get_weather(city: str) -> str:
            return f'sunny in {city}'

        beam_toolset = BEAMFunctionToolset(
            wrapped=toolset,
            workflow_id='wf-3',
            step_counter=step_counter,
        )

        ctx = RunContext[None](deps=None, model=model, usage=RunUsage())

        # Get the tool definition to pass to call_tool
        tools = await toolset.get_tools(ctx)
        tool = tools['get_weather']

        with (
            patch(f'{FUNCTION_TOOLSET_MODULE}.in_beam_worker', return_value=True),
            patch(f'{CHECKPOINT_MODULE}.get_erlang', return_value=mock_erl),
        ):
            result = await beam_toolset.call_tool('get_weather', {'city': 'Paris'}, ctx, tool)

        assert result == 'sunny in Paris'
        assert len(store) == 1
        key = next(iter(store))
        assert 'tool.get_weather:0' in key

    async def test_tool_call_replayed(self) -> None:
        """Pre-populated tool checkpoint is returned without executing the tool."""
        store, mock_erl = make_mock_erlang()
        step_counter = StepCounter()
        model = TestModel()

        toolset = FunctionToolset[None]()
        call_count = 0

        @toolset.tool
        async def get_weather(city: str) -> str:
            nonlocal call_count
            call_count += 1
            return f'sunny in {city}'  # pragma: no cover

        beam_toolset = BEAMFunctionToolset(
            wrapped=toolset,
            workflow_id='wf-4',
            step_counter=step_counter,
        )

        # Pre-populate checkpoint
        store['beam_ckpt:wf-4:tool.get_weather:0'] = serialize_tool_result('rainy in London')

        ctx = RunContext[None](deps=None, model=model, usage=RunUsage())

        tools = await toolset.get_tools(ctx)
        tool = tools['get_weather']

        with (
            patch(f'{FUNCTION_TOOLSET_MODULE}.in_beam_worker', return_value=True),
            patch(f'{CHECKPOINT_MODULE}.get_erlang', return_value=mock_erl),
        ):
            result = await beam_toolset.call_tool('get_weather', {'city': 'London'}, ctx, tool)

        assert result == 'rainy in London'
        assert call_count == 0  # Real tool was NOT called


# --- 6. TestBEAMStreaming ---


class TestBEAMStreaming:
    async def test_run_stream_raises_in_beam(self) -> None:
        """When in BEAM worker, `run_stream()` should raise UserError."""
        agent = Agent(name='stream_agent', model=TestModel())
        beam_agent = BEAMAgent(agent)

        with (
            patch(f'{AGENT_MODULE}.in_beam_worker', return_value=True),
            pytest.raises(
                UserError,
                match=re.escape('`agent.run_stream()` cannot be used inside a BEAM workflow.'),
            ),
        ):
            async with beam_agent.run_stream('Hello'):
                pass  # pragma: no cover


# --- 7. TestBEAMToolsetWrapping ---


class TestBEAMToolsetWrapping:
    def test_beamify_function_toolset(self) -> None:
        toolset = FunctionToolset[None]()
        result = beamify_toolset(toolset, workflow_id='wf-5', step_counter=StepCounter())
        assert isinstance(result, BEAMFunctionToolset)

    def test_beamify_unknown_toolset_passthrough(self) -> None:
        """Unknown toolset types are returned as-is."""

        class UnknownToolset:
            pass

        toolset = UnknownToolset()
        result = beamify_toolset(toolset, workflow_id='wf-6', step_counter=StepCounter())  # type: ignore[arg-type]
        assert result is toolset

    def test_visit_and_replace_returns_self(self) -> None:
        """BEAM wrapper toolsets cannot be swapped out after wrapping."""
        toolset = FunctionToolset[None]()
        beam_toolset = BEAMFunctionToolset(
            wrapped=toolset,
            workflow_id='wf-7',
            step_counter=StepCounter(),
        )
        replacement = FunctionToolset[None](id='replaced')
        result = beam_toolset.visit_and_replace(lambda t: replacement)
        assert result is beam_toolset

    def test_id_delegates_to_wrapped(self) -> None:
        toolset = FunctionToolset[None](id='my-tools')
        beam_toolset = BEAMFunctionToolset(
            wrapped=toolset,
            workflow_id='wf-8',
            step_counter=StepCounter(),
        )
        assert beam_toolset.id == 'my-tools'


# --- 8. TestBEAMModelSerialization ---


class TestBEAMModelSerialization:
    def test_model_response_round_trip(self) -> None:
        response = ModelResponse(
            parts=[TextPart(content='Hello, world!')],
            usage=RequestUsage(input_tokens=10, output_tokens=20),
            model_name='test-model',
        )
        serialized = serialize_model_response(response)
        deserialized = deserialize_model_response(serialized)

        assert deserialized.parts[0].content == 'Hello, world!'  # type: ignore[union-attr]
        assert deserialized.usage.input_tokens == 10
        assert deserialized.usage.output_tokens == 20
        assert deserialized.model_name == 'test-model'

    def test_model_response_with_tool_calls(self) -> None:
        response = ModelResponse(
            parts=[
                TextPart(content='Let me check that.'),
                ToolCallPart(tool_name='search', args={'query': 'pydantic ai'}),
            ],
            usage=RequestUsage(),
            model_name='test-model',
        )
        serialized = serialize_model_response(response)
        deserialized = deserialize_model_response(serialized)

        assert len(deserialized.parts) == 2
        assert deserialized.parts[0].content == 'Let me check that.'  # type: ignore[union-attr]
        assert deserialized.parts[1].tool_name == 'search'  # type: ignore[union-attr]

    def test_tool_result_round_trip_string(self) -> None:
        result = 'sunny in Paris'
        serialized = serialize_tool_result(result)
        deserialized = deserialize_tool_result(serialized)
        assert deserialized == 'sunny in Paris'

    def test_tool_result_round_trip_dict(self) -> None:
        result = {'temperature': 22, 'condition': 'sunny', 'tags': ['warm', 'clear']}
        serialized = serialize_tool_result(result)
        deserialized = deserialize_tool_result(serialized)
        assert deserialized == result

    def test_tool_result_round_trip_list(self) -> None:
        result = [1, 2, 3]
        serialized = serialize_tool_result(result)
        deserialized = deserialize_tool_result(serialized)
        assert deserialized == [1, 2, 3]

    def test_serialized_format_is_json_bytes(self) -> None:
        response = ModelResponse(
            parts=[TextPart(content='test')],
            usage=RequestUsage(),
        )
        serialized = serialize_model_response(response)
        assert isinstance(serialized, bytes)
        # Should be valid JSON
        parsed = json.loads(serialized)
        assert isinstance(parsed, dict)

        tool_serialized = serialize_tool_result({'key': 'value'})
        assert isinstance(tool_serialized, bytes)
        assert json.loads(tool_serialized) == {'key': 'value'}
