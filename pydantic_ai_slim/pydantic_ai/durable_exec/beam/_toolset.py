from __future__ import annotations

from abc import ABC
from collections.abc import Callable
from typing import Any

from pydantic_ai import AbstractToolset, WrapperToolset
from pydantic_ai.tools import AgentDepsT
from pydantic_ai.toolsets.function import FunctionToolset

from ._checkpoint import StepCounter


class BEAMWrapperToolset(WrapperToolset[AgentDepsT], ABC):
    """Base class for BEAM-wrapped toolsets with ETS checkpointing."""

    @property
    def id(self) -> str | None:
        return self.wrapped.id

    def visit_and_replace(
        self, visitor: Callable[[AbstractToolset[AgentDepsT]], AbstractToolset[AgentDepsT]]
    ) -> AbstractToolset[AgentDepsT]:
        # BEAM-ified toolsets cannot be swapped out after the fact.
        return self


def beamify_toolset(
    toolset: AbstractToolset[Any],
    workflow_id: str,
    step_counter: StepCounter,
) -> AbstractToolset[Any]:
    """Transform a toolset into its BEAM-wrapped equivalent for durable checkpointing."""
    if isinstance(toolset, FunctionToolset):
        from ._function_toolset import BEAMFunctionToolset

        return BEAMFunctionToolset(
            wrapped=toolset,
            workflow_id=workflow_id,
            step_counter=step_counter,
        )

    try:
        from pydantic_ai.mcp import MCPServer

        from ._mcp_server import BEAMMCPServer
    except ImportError:
        pass
    else:
        if isinstance(toolset, MCPServer):
            return BEAMMCPServer(
                wrapped=toolset,
                workflow_id=workflow_id,
                step_counter=step_counter,
            )

    try:
        from pydantic_ai.toolsets.fastmcp import FastMCPToolset

        from ._fastmcp_toolset import BEAMFastMCPToolset
    except ImportError:
        pass
    else:
        if isinstance(toolset, FastMCPToolset):
            return BEAMFastMCPToolset(
                wrapped=toolset,
                workflow_id=workflow_id,
                step_counter=step_counter,
            )

    return toolset
