# Durable Execution with BEAM

[BEAM](https://www.erlang.org/) is the virtual machine that runs Erlang (and Elixir). Using [`erlang_python`](https://github.com/benoitc/erlang-python), Python runs **in-process** inside BEAM workers, giving Pydantic AI agents access to OTP's supervision, checkpointing, and distribution with zero external dependencies.

## Durable Execution

`BEAMAgent` makes your agent **durable** by checkpointing model requests and tool calls to ETS (Erlang Term Storage). If the BEAM process crashes, the OTP supervisor restarts it and the agent automatically resumes from the last completed step.

* **Workflows** wrap the full agent run. On crash, the supervisor restarts the process and replays from checkpoints.
* **Steps** wrap individual model requests and tool calls. Each step's result is checkpointed to ETS before continuing.

Every step output is stored in ETS via `erlang_python`'s built-in shared state API. When a workflow is replayed after a crash, completed steps return their cached results and execution resumes from the first incomplete step.

The diagram below shows the overall architecture of an agentic application on BEAM. `erlang_python` runs Python in-process via a NIF -- there is no network hop between Python and Erlang.

```text
+------------------------------------------------------+
|  Python (BEAM worker via erlang_python)               |
|                                                       |
|  BEAMAgent(agent)                                     |
|    +-- BEAMModel(model)                               |
|    |     +-- request() -> checkpoint to ETS           |
|    +-- BEAMToolset(toolset)                           |
|          +-- call_tool() -> checkpoint to ETS         |
+------------------------------------------------------+
                        |
                        v
+------------------------------------------------------+
|  Erlang (OTP)                                        |
|                                                       |
|  supervisor (simple_one_for_one)                      |
|    +-- workflow process (gen_server per agent run)     |
|          +-- ETS: checkpoint step results             |
|          +-- pg: publish events to subscribers        |
|          +-- on crash: restart -> read ETS -> resume  |
+------------------------------------------------------+
```

See the [`erlang_python` documentation](https://github.com/benoitc/erlang-python) for more information.

## Durable Agent

Any agent can be wrapped in a [`BEAMAgent`][pydantic_ai.durable_exec.beam.BEAMAgent] to get durable execution. `BEAMAgent` automatically:

* Wraps [`Agent.run()`][pydantic_ai.agent.Agent.run] and [`Agent.run_sync()`][pydantic_ai.agent.Agent.run_sync] as BEAM workflows with ETS checkpointing.
* Wraps [model requests](../models/overview.md), [tool calls](../tools.md), and [MCP communication](../mcp/client.md) as checkpointed steps.

The original agent can still be used as normal outside a BEAM worker. When running outside BEAM (e.g. during development or testing), `BEAMAgent` delegates directly to the underlying agent with no overhead.

Here is a simple but complete example of wrapping an agent for durable execution. It requires an Erlang/OTP project with `erlang_python` (see [Erlang setup](#erlang-setup) below) and Pydantic AI installed with the BEAM optional group:

```bash
pip/uv-add pydantic-ai[beam]
```

Or with the slim package:

```bash
pip/uv-add pydantic-ai-slim[beam]
```

!!! note

    There is no PyPI dependency for the BEAM integration -- the `erlang` module is injected at runtime by `erlang_python`. The optional group exists so `pip install pydantic-ai[beam]` works consistently with other integrations.

```python {title="let_it_crash.py" test="skip"}
import asyncio

from pydantic_ai import Agent
from pydantic_ai.durable_exec.beam import BEAMAgent

agent = Agent(
    'openai:gpt-5.2',
    instructions="You're an expert in distributed systems and fault-tolerant architectures.",
    name='let_it_crash',  # (4)!
)

beam_agent = BEAMAgent(agent)  # (1)!


def run_agent(prompt: str) -> str:  # (3)!
    result = asyncio.run(beam_agent.run(prompt))  # (2)!
    return result.output
```

1. Wrapping the agent enables durable execution when running inside a BEAM worker.
2. [`BEAMAgent.run()`][pydantic_ai.durable_exec.beam.BEAMAgent.run] works like [`Agent.run()`][pydantic_ai.agent.Agent.run], but checkpoints model requests and tool calls to ETS.
3. This function is called from Erlang via `py:call(my_agent_module, run_agent, [Prompt])`.
4. The agent's `name` is used to uniquely identify its workflows.

Then from Erlang:

```erlang
{ok, Result} = py:call(my_agent_module, run_agent, [<<"How do BEAM processes achieve fault tolerance?">>]).
```

## BEAM Integration Considerations

There are a few considerations specific to agents and toolsets when using BEAM for durable execution. These are important to understand to ensure that your agents and toolsets work correctly with BEAM's checkpoint and replay model.

### Agent Names and Toolset IDs

To ensure that BEAM can correctly resume workflows after a crash, each agent instance must have a unique [`name`][pydantic_ai.agent.AbstractAgent.name] and each toolset must have a unique [`id`][pydantic_ai.toolsets.AbstractToolset.id]. These fields are normally optional, but are required to be set when using BEAM. They should not be changed once the durable agent has been deployed to production as this would break active workflows.

When `BEAMAgent` dynamically wraps the agent's model requests and toolsets (specifically those that implement their own tool listing and calling, i.e. [`FunctionToolset`][pydantic_ai.toolsets.FunctionToolset] and [`MCPServer`][pydantic_ai.mcp.MCPServer]), their checkpoint keys are derived from the agent's name and the toolset IDs.

All tool calls and MCP communication are automatically checkpointed. Event stream handlers are also wrapped to push events to Erlang subscribers.

Other than that, any agent and toolset will just work!

### Agent Run Context and Dependencies

BEAM checkpoints step outputs to ETS using JSON serialization. This means tool return values must be JSON-serializable. Model responses are serialized via Pydantic's [`TypeAdapter`](https://docs.pydantic.dev/latest/api/type_adapter/).

### Streaming

Because BEAM workflows checkpoint the full model response, [`Agent.run_stream()`][pydantic_ai.agent.Agent.run_stream], [`Agent.run_stream_events()`][pydantic_ai.agent.Agent.run_stream_events], and [`Agent.iter()`][pydantic_ai.agent.Agent.iter] are not supported inside a BEAM workflow.

Instead, you can implement streaming by setting an [`event_stream_handler`][pydantic_ai.agent.EventStreamHandler] on the `Agent` or `BEAMAgent` instance and using [`BEAMAgent.run()`][pydantic_ai.durable_exec.beam.BEAMAgent.run]. The event stream handler function will receive the agent [run context][pydantic_ai.tools.RunContext] and an async iterable of events from the model's streaming response and the agent's execution of tools. For examples, see the [streaming docs](../agent.md#streaming-all-events).

Each event is automatically pushed to pg subscribers, allowing Erlang processes to consume the stream in real time:

```python {title="live_view.py" test="skip"}
from collections.abc import AsyncIterable

from pydantic_ai import Agent
from pydantic_ai.agent import AgentStreamEvent
from pydantic_ai.durable_exec.beam import BEAMAgent
from pydantic_ai.tools import RunContext


async def stream_to_erlang(
    ctx: RunContext[None],
    events: AsyncIterable[AgentStreamEvent],
) -> None:
    async for event in events:
        pass  # Events are published to pg subscribers automatically


agent = Agent(
    'openai:gpt-5.2',
    name='live_view',
    event_stream_handler=stream_to_erlang,
)

beam_agent = BEAMAgent(agent)
```

On the Erlang side, subscribe to workflow events via `pg:join/3`:

```erlang
pg:join(beam_workflows, WorkflowId, self()).

%% In your gen_server handle_info
handle_info({beam_event, _WorkflowId, EventData}, State) ->
    %% Forward to WebSocket, log, etc.
    {noreply, State}.
```

### Parallel Tool Execution

When using `BEAMAgent`, tools are executed in parallel by default. To guarantee deterministic replay, BEAM waits for all parallel tool calls to complete before emitting events in order. It's equivalent to the behavior of [`with agent.parallel_tool_call_execution_mode('parallel_ordered_events')`][pydantic_ai.agent.AbstractAgent.parallel_tool_call_execution_mode].

If you prefer strict ordering, you can configure the agent to run tools sequentially by setting `parallel_execution_mode='sequential'` when initializing the `BEAMAgent`.

## Retries

BEAM relies on OTP's supervision tree for fault tolerance: if a worker process crashes, the supervisor restarts it and the agent automatically resumes from the last checkpointed step. This replaces the need for application-level retry configuration.

On top of OTP's automatic restarts, Pydantic AI and various provider API clients also have their own request retry logic. Enabling these at the same time may cause the request to be retried more often than expected, with improper `Retry-After` handling.

When using BEAM, it's recommended to not use [HTTP Request Retries](../retries.md) and to turn off your provider API client's own retry logic, for example by setting `max_retries=0` on a [custom `OpenAIProvider` API client](../models/openai.md#custom-openai-client).

You can customize restart behavior using the OTP supervisor's `intensity` and `period` settings in your supervision tree configuration.

## Observability with Logfire

Pydantic AI generates telemetry events for each agent run, model request, and tool call. These can be sent to [Pydantic Logfire](../logfire.md) to get a complete picture of what's happening in your application, including inside BEAM workflows.

Since BEAM runs Python in-process, Logfire instrumentation works the same as in any other Python application — configure Logfire as usual and all Pydantic AI spans will be emitted automatically. See the [Logfire documentation](../logfire.md) for setup instructions.

## Worker Pool Sizing

Each reentrant callback consumes a worker from the pool. During an agent run, both the outer `py:call` (running the agent) and inner callbacks (checkpoints) need workers simultaneously.

**Minimum pool size**: `max_concurrent_agent_runs * 2 + 1`

For example, if you expect up to 4 concurrent agent runs, configure at least 9 workers in `sys.config`:

```erlang
[{erlang_python, [
    {num_workers, 9}
]}].
```

## Erlang Setup

### rebar.config

Add `erlang_python` as a dependency:

```erlang
{deps, [
    {erlang_python, "0.3.0"}
]}.
```

### sys.config

Configure the worker pool:

```erlang
[
    {erlang_python, [
        {num_workers, 9},
        {num_executors, 4}
    ]}
].
```

### Application Startup

Start `erlang_python` and add the companion module to your supervision tree:

```erlang
application:ensure_all_started(erlang_python).
```

```erlang
%% In your application's start/2 callback
start(_StartType, _StartArgs) ->
    Children = [
        #{id => agent_durable, start => {agent_durable, start_link, []}}
    ],
    {ok, {#{strategy => one_for_one, intensity => 5, period => 10}, Children}}.
```

### Reference Erlang Module

Checkpoint storage uses `erlang_python`'s built-in shared state API (`state_set`/`state_get` in Python, `py:state_store`/`py:state_fetch` in Erlang). The companion module below registers functions for event publishing and workflow lifecycle:

| Python calls | Erlang registered function | Purpose |
|---|---|---|
| `erlang.call('beam_event_publish', wf_id, data)` | `agent_durable:event_publish/2` | Push event to pg subscribers |
| `erlang.call('beam_workflow_start', wf_id, name)` | `agent_durable:workflow_start/2` | Register workflow |
| `erlang.call('beam_workflow_complete', wf_id)` | `agent_durable:workflow_complete/1` | Mark done, clean up |

```erlang
-module(agent_durable).
-behaviour(gen_server).

-export([start_link/0, start_link/1]).
-export([init/1, handle_call/3, handle_cast/2, handle_info/2]).
-export([workflow_start/2, workflow_complete/1, event_publish/2]).

start_link() ->
    start_link([]).

start_link(Opts) ->
    gen_server:start_link({local, ?MODULE}, ?MODULE, Opts, []).

init(_Opts) ->
    ets:new(beam_workflows, [named_table, set, public, {read_concurrency, true}]),
    py:register_function(beam_workflow_start, ?MODULE, workflow_start),
    py:register_function(beam_workflow_complete, ?MODULE, workflow_complete),
    py:register_function(beam_event_publish, ?MODULE, event_publish),
    {ok, #{}}.

workflow_start(WorkflowId, AgentName) ->
    ets:insert(beam_workflows, {WorkflowId, AgentName, running, erlang:system_time(second)}),
    pg:join(beam_workflows, WorkflowId, self()),
    ok.

workflow_complete(WorkflowId) ->
    ets:insert(beam_workflows, {WorkflowId, undefined, completed, erlang:system_time(second)}),
    cleanup_checkpoints(WorkflowId),
    pg:leave(beam_workflows, WorkflowId, self()),
    ok.

event_publish(WorkflowId, EventData) ->
    lists:foreach(fun(Pid) ->
        Pid ! {beam_event, WorkflowId, EventData}
    end, pg:get_members(beam_workflows, WorkflowId)),
    ok.

cleanup_checkpoints(WorkflowId) ->
    Prefix = <<"beam_ckpt:", (erlang:term_to_binary(WorkflowId))/binary, ":">>,
    lists:foreach(fun(Key) ->
        case binary:match(Key, Prefix) of
            {0, _} -> py:state_delete(Key);
            _ -> ok
        end
    end, py:state_keys()).

handle_call(_Request, _From, State) ->
    {reply, ok, State}.

handle_cast(_Msg, State) ->
    {noreply, State}.

handle_info(_Info, State) ->
    {noreply, State}.
```
