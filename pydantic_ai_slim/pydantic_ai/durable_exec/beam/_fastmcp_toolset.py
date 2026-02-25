from __future__ import annotations

from pydantic_ai import ToolsetTool
from pydantic_ai.tools import AgentDepsT, ToolDefinition
from pydantic_ai.toolsets.fastmcp import FastMCPToolset

from ._checkpoint import StepCounter
from ._mcp_server import BEAMMCPToolset


class BEAMFastMCPToolset(BEAMMCPToolset[AgentDepsT]):
    """A wrapper for FastMCPToolset that checkpoints get_tools and call_tool to ETS for durable replay."""

    def __init__(
        self,
        wrapped: FastMCPToolset[AgentDepsT],
        *,
        workflow_id: str,
        step_counter: StepCounter,
    ):
        super().__init__(
            wrapped,
            workflow_id=workflow_id,
            step_counter=step_counter,
        )

    def tool_for_tool_def(self, tool_def: ToolDefinition) -> ToolsetTool[AgentDepsT]:
        assert isinstance(self.wrapped, FastMCPToolset)
        return self.wrapped.tool_for_tool_def(tool_def)
