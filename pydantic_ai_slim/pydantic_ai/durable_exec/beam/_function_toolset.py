from __future__ import annotations

from typing import Any

from pydantic_ai import ToolsetTool
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai.toolsets.function import FunctionToolset

from ._checkpoint import (
    StepCounter,
    deserialize_tool_result,
    get_checkpoint,
    serialize_tool_result,
    set_checkpoint,
)
from ._erlang import in_beam_worker
from ._toolset import BEAMWrapperToolset


class BEAMFunctionToolset(BEAMWrapperToolset[AgentDepsT]):
    """A wrapper for FunctionToolset that checkpoints tool calls to ETS for durable replay."""

    def __init__(
        self,
        wrapped: FunctionToolset[AgentDepsT],
        *,
        workflow_id: str,
        step_counter: StepCounter,
    ):
        super().__init__(wrapped)
        self._workflow_id = workflow_id
        self._step_counter = step_counter

    async def call_tool(
        self,
        name: str,
        tool_args: dict[str, Any],
        ctx: RunContext[AgentDepsT],
        tool: ToolsetTool[AgentDepsT],
    ) -> Any:
        if not in_beam_worker():
            return await super().call_tool(name, tool_args, ctx, tool)

        step_id = self._step_counter.next(f'tool.{name}')

        cached = get_checkpoint(self._workflow_id, step_id)
        if cached is not None:
            return deserialize_tool_result(cached)

        result = await super().call_tool(name, tool_args, ctx, tool)
        set_checkpoint(self._workflow_id, step_id, serialize_tool_result(result))
        return result
