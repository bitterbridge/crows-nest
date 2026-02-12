"""Adapter bridging atomic_agents with the thunk system.

Provides:
- ThunkTool: Expose thunk operations as atomic_agents tools
- AgentAdapter: Wrap AtomicAgent with ThunkContext support
- create_client: Factory for instructor clients from config
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from atomic_agents import AgentConfig, AtomicAgent, BaseTool, BaseToolConfig
from atomic_agents.context.system_prompt_generator import SystemPromptGenerator
from pydantic import BaseModel, Field, create_model

if TYPE_CHECKING:
    import instructor

    from crows_nest.core.context import ThunkContext
    from crows_nest.core.registry import OperationInfo, ThunkRegistry
    from crows_nest.llm.providers import ProviderConfig


InputT = TypeVar("InputT", bound=BaseModel)
OutputT = TypeVar("OutputT", bound=BaseModel)


def create_client(config: ProviderConfig) -> instructor.Instructor:
    """Create an instructor client from a provider configuration."""
    return config.create_client()


class ThunkTool(BaseTool):  # type: ignore[misc]
    """Adapter exposing a thunk operation as an atomic_agents tool.

    Dynamically creates Pydantic input/output schemas from the operation's
    parameter definitions.
    """

    def __init__(
        self,
        operation_info: OperationInfo,
        registry: ThunkRegistry,
        config: BaseToolConfig | None = None,
    ) -> None:
        """Create a tool from a thunk operation.

        Args:
            operation_info: The operation metadata from the registry.
            registry: The thunk registry for execution.
            config: Optional tool configuration override.
        """
        self._operation_info = operation_info
        self._registry = registry

        # Build config from operation if not provided
        if config is None:
            config = BaseToolConfig(
                title=operation_info.name,
                description=operation_info.description or f"Execute {operation_info.name}",
            )

        # Dynamically create input schema from operation parameters
        self.input_schema = self._build_input_schema()
        self.output_schema = self._build_output_schema()

        super().__init__(config)

    def _build_input_schema(self) -> type[BaseModel]:
        """Build a Pydantic model from operation parameters."""
        fields: dict[str, Any] = {}

        for param_name, param in self._operation_info.parameters.items():
            # Map parameter info to Pydantic field
            field_type = self._map_type(param.annotation)
            default = ... if param.required else param.default

            fields[param_name] = (
                field_type,
                Field(default=default, description=f"Parameter: {param_name}"),
            )

        return create_model(
            f"{self._operation_info.name.replace('.', '_')}_Input",
            **fields,
        )

    def _build_output_schema(self) -> type[BaseModel]:
        """Build a generic output schema.

        Since thunk operations can return any type, we use a flexible schema.
        """
        return create_model(
            f"{self._operation_info.name.replace('.', '_')}_Output",
            result=(Any, Field(description="Operation result")),
            success=(bool, Field(default=True, description="Whether operation succeeded")),
            error=(str | None, Field(default=None, description="Error message if failed")),
        )

    def _map_type(self, type_hint: str | None) -> type[Any]:
        """Map a type hint string to a Python type."""
        if type_hint is None:
            return object  # Use object as catch-all for Any

        # Simple type mapping - could be more sophisticated
        type_map: dict[str, type[Any]] = {
            "str": str,
            "int": int,
            "float": float,
            "bool": bool,
            "list": list,
            "dict": dict,
            "Any": object,
        }

        # Handle Optional, List, Dict etc.
        for base, typ in type_map.items():
            if type_hint == base or type_hint.startswith(f"{base}["):
                return typ

        return object  # Use object as catch-all for Any

    def run(self, params: BaseModel) -> BaseModel:
        """Execute the thunk operation synchronously."""
        # Run async operation in sync context
        return asyncio.run(self.run_async(params))

    async def run_async(self, params: BaseModel) -> BaseModel:
        """Execute the thunk operation asynchronously."""
        from crows_nest.core.runtime import ThunkRuntime
        from crows_nest.core.thunk import Thunk

        runtime = ThunkRuntime(registry=self._registry)
        thunk = Thunk.create(
            operation=self._operation_info.name,
            inputs=params.model_dump(),
        )

        result = await runtime.force(thunk)

        output_cls = self.output_schema
        if result.is_success:
            return output_cls(result=result.value, success=True, error=None)
        else:
            error_msg = result.error.message if result.error else "Unknown error"
            return output_cls(result=None, success=False, error=error_msg)


def tools_from_registry(
    registry: ThunkRegistry,
    operations: list[str] | None = None,
    exclude: list[str] | None = None,
) -> list[ThunkTool]:
    """Create tools for thunk operations.

    Args:
        registry: The thunk registry to pull operations from.
        operations: Specific operations to include (None = all).
        exclude: Operations to exclude.

    Returns:
        List of ThunkTool instances.
    """
    exclude = exclude or []
    tools = []

    for op_info in registry.list_operations_info():
        if operations is not None and op_info.name not in operations:
            continue
        if op_info.name in exclude:
            continue

        tools.append(ThunkTool(op_info, registry))

    return tools


@dataclass
class AgentAdapter(Generic[InputT, OutputT]):
    """Adapter wrapping AtomicAgent with ThunkContext support.

    Bridges atomic_agents' execution model with our context system,
    enabling cancellation checking and progress reporting during LLM calls.
    """

    agent: AtomicAgent[InputT, OutputT]
    """The wrapped atomic_agents agent."""

    @classmethod
    def create(
        cls,
        config: ProviderConfig,
        input_schema: type[InputT],
        output_schema: type[OutputT],
        system_prompt: str | None = None,
        tools: list[BaseTool] | None = None,
        **agent_kwargs: Any,
    ) -> AgentAdapter[InputT, OutputT]:
        """Create an agent adapter from a provider configuration.

        Args:
            config: Provider configuration (Anthropic, OpenAI, etc.)
            input_schema: Pydantic model for agent input.
            output_schema: Pydantic model for agent output.
            system_prompt: Optional system prompt.
            tools: Optional list of tools (can include ThunkTools).
            **agent_kwargs: Additional arguments passed to AtomicAgent.

        Returns:
            Configured AgentAdapter.
        """
        client = config.create_client()

        # Build system prompt generator if provided
        system_prompt_gen = None
        if system_prompt:
            system_prompt_gen = SystemPromptGenerator(
                background=[system_prompt],
            )

        agent_config = AgentConfig(
            client=client,
            model=config.model,
            system_prompt_generator=system_prompt_gen,
            **agent_kwargs,
        )

        agent = AtomicAgent[input_schema, output_schema](
            config=agent_config,
            input_schema=input_schema,
            output_schema=output_schema,
            tools=tools or [],
        )

        return cls(agent=agent)

    async def run(
        self,
        user_input: InputT,
        context: ThunkContext | None = None,
    ) -> OutputT:
        """Run the agent with optional context support.

        Args:
            user_input: The input to the agent.
            context: Optional ThunkContext for cancellation/progress.

        Returns:
            The agent's output.

        Raises:
            CancellationError: If context is cancelled.
        """
        # Check for cancellation before starting
        if context:
            context.raise_if_cancelled()

        # Report progress if available
        if context:
            await context.progress.report_progress(0.0, "Starting LLM call")

        # Run the agent
        result: OutputT = await self.agent.run_async(user_input)

        if context:
            await context.progress.report_progress(1.0, "LLM call complete")

        return result

    async def run_stream(
        self,
        user_input: InputT,
        context: ThunkContext | None = None,
    ) -> Any:
        """Run the agent with streaming output.

        Yields partial results and reports progress via context.

        Args:
            user_input: The input to the agent.
            context: Optional ThunkContext for cancellation/progress.

        Yields:
            Partial output objects as they stream in.
        """
        if context:
            context.raise_if_cancelled()
            await context.progress.start()

        chunks_received = 0
        async for partial in self.agent.run_async_stream(user_input):
            # Check cancellation between chunks
            if context:
                context.raise_if_cancelled()
                await context.progress.emit_partial("llm_chunk", {"chunk": chunks_received})

            chunks_received += 1
            yield partial

        if context:
            await context.progress.complete()
