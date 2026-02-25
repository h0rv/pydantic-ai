from __future__ import annotations

import dataclasses
import json
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from typing_extensions import Self

from pydantic_ai import AbstractToolset, ToolsetTool
from pydantic_ai.mcp import MCPServer
from pydantic_ai.tools import AgentDepsT, RunContext, ToolDefinition

from ._checkpoint import (
    StepCounter,
    deserialize_tool_result,
    get_checkpoint,
    serialize_tool_result,
    set_checkpoint,
)
from ._erlang import in_beam_worker
from ._toolset import BEAMWrapperToolset

if TYPE_CHECKING:
    from pydantic_ai.mcp import ToolResult


class BEAMMCPToolset(BEAMWrapperToolset[AgentDepsT], ABC):
    """Base class for BEAM-wrapped MCP toolsets with ETS checkpointing on get_tools and call_tool."""

    def __init__(
        self,
        wrapped: AbstractToolset[AgentDepsT],
        *,
        workflow_id: str,
        step_counter: StepCounter,
    ):
        super().__init__(wrapped)
        self._workflow_id = workflow_id
        self._step_counter = step_counter

    @abstractmethod
    def tool_for_tool_def(self, tool_def: ToolDefinition) -> ToolsetTool[AgentDepsT]:
        raise NotImplementedError

    async def __aenter__(self) -> Self:
        # The wrapped MCP toolset enters itself around listing and calling tools
        # so we don't need to enter it here.
        return self

    async def __aexit__(self, *args: object) -> bool | None:
        return None

    async def get_tools(self, ctx: RunContext[AgentDepsT]) -> dict[str, ToolsetTool[AgentDepsT]]:
        if not in_beam_worker():
            return await super().get_tools(ctx)

        step_id = self._step_counter.next('mcp.get_tools')

        cached = get_checkpoint(self._workflow_id, step_id)
        if cached is not None:
            tool_defs: dict[str, ToolDefinition] = {
                name: ToolDefinition(**td) for name, td in json.loads(cached).items()
            }
            return {name: self.tool_for_tool_def(td) for name, td in tool_defs.items()}

        tools = await super().get_tools(ctx)
        serialized = json.dumps({name: dataclasses.asdict(tool.tool_def) for name, tool in tools.items()}).encode()
        set_checkpoint(self._workflow_id, step_id, serialized)
        return {name: self.tool_for_tool_def(tool.tool_def) for name, tool in tools.items()}

    async def call_tool(
        self,
        name: str,
        tool_args: dict[str, Any],
        ctx: RunContext[AgentDepsT],
        tool: ToolsetTool[AgentDepsT],
    ) -> ToolResult:
        if not in_beam_worker():
            return await super().call_tool(name, tool_args, ctx, tool)

        step_id = self._step_counter.next('mcp.call_tool')

        cached = get_checkpoint(self._workflow_id, step_id)
        if cached is not None:
            return deserialize_tool_result(cached)

        result = await super().call_tool(name, tool_args, ctx, tool)
        set_checkpoint(self._workflow_id, step_id, serialize_tool_result(result))
        return result


class BEAMMCPServer(BEAMMCPToolset[AgentDepsT]):
    """A wrapper for MCPServer that checkpoints get_tools and call_tool to ETS for durable replay.

    Tool definitions are cached across steps to avoid redundant MCP server round-trips,
    respecting the wrapped server's `cache_tools` setting.
    """

    def __init__(
        self,
        wrapped: MCPServer,
        *,
        workflow_id: str,
        step_counter: StepCounter,
    ):
        super().__init__(
            wrapped,
            workflow_id=workflow_id,
            step_counter=step_counter,
        )
        self._cached_tool_defs: dict[str, ToolDefinition] | None = None

    @property
    def _server(self) -> MCPServer:
        assert isinstance(self.wrapped, MCPServer)
        return self.wrapped

    def tool_for_tool_def(self, tool_def: ToolDefinition) -> ToolsetTool[AgentDepsT]:
        return self._server.tool_for_tool_def(tool_def)

    async def get_tools(self, ctx: RunContext[AgentDepsT]) -> dict[str, ToolsetTool[AgentDepsT]]:
        if self._server.cache_tools and self._cached_tool_defs is not None:
            return {name: self.tool_for_tool_def(td) for name, td in self._cached_tool_defs.items()}

        result = await super().get_tools(ctx)
        if self._server.cache_tools:
            self._cached_tool_defs = {name: tool.tool_def for name, tool in result.items()}
        return result
