from abc import abstractmethod
from typing import (
    TYPE_CHECKING,
    Any,
    List,
    Optional,
    Type,
    cast,
)

from mcp_agent.logging.logger import get_logger
from mcp_agent.mcp.interfaces import (
    AugmentedLLMProtocol,
    MessageParamT,
    MessageT,
    ModelT,
)
from mcp_agent.mcp.prompt_message_multipart import PromptMessageMultipart
from mcp_agent.workflows.llm.sampling_format_converter import (
    BasicFormatConverter,
    ProviderFormatConverter,
)

# Forward reference for type annotations
if TYPE_CHECKING:
    from mcp_agent.agents.agent import Agent
    from mcp_agent.context import Context
    from mcp_agent.mcp.prompt_message_multipart import PromptMessageMultipart


from mcp.types import (
    CallToolRequest,
    CallToolResult,
    GetPromptResult,
    PromptMessage,
    TextContent,
)
from rich.text import Text

from mcp_agent.context_dependent import ContextDependent
from mcp_agent.core.exceptions import ModelConfigError, PromptExitError
from mcp_agent.core.request_params import RequestParams
from mcp_agent.event_progress import ProgressAction
from mcp_agent.mcp.mcp_aggregator import MCPAggregator
from mcp_agent.ui.console_display import ConsoleDisplay
from mcp_agent.workflows.llm.memory import Memory, SimpleMemory

# TODO -- move this to a constant
HUMAN_INPUT_TOOL_NAME = "__human_input__"


class AugmentedLLM(ContextDependent, AugmentedLLMProtocol[MessageParamT, MessageT]):
    """
    The basic building block of agentic systems is an LLM enhanced with augmentations
    such as retrieval, tools, and memory provided from a collection of MCP servers.
    Our current models can actively use these capabilities—generating their own search queries,
    selecting appropriate tools, and determining what information to retain.
    """

    provider: str | None = None

    def __init__(
        self,
        agent: Optional["Agent"] = None,
        server_names: List[str] | None = None,
        instruction: str | None = None,
        name: str | None = None,
        request_params: RequestParams | None = None,
        type_converter: Type[ProviderFormatConverter[MessageParamT, MessageT]] = BasicFormatConverter,
        context: Optional["Context"] = None,
        **kwargs: dict[str, Any],
    ) -> None:
        """
        Initialize the LLM with a list of server names and an instruction.
        If a name is provided, it will be used to identify the LLM.
        If an agent is provided, all other properties are optional
        """
        if context is None and agent is not None:
            context = agent.context
        # Extract request_params before super() call
        self._init_request_params = request_params
        super().__init__(context=context, **kwargs)
        self.logger = get_logger(__name__)
        self.executor = self.context.executor
        self.aggregator = agent if agent is not None else MCPAggregator(server_names or [])
        self.name = name or (agent.name if agent else None)
        self.instruction = instruction or (agent.instruction if agent and isinstance(agent.instruction, str) else None)
        self.history: Memory[MessageParamT] = SimpleMemory[MessageParamT]()

        # Initialize the display component
        self.display = ConsoleDisplay(config=self.context.config)

        # Initialize default parameters
        self.default_request_params = self._initialize_default_params(kwargs)

        # Merge with provided params if any
        if self._init_request_params:
            self.default_request_params = self._merge_request_params(self.default_request_params, self._init_request_params)

        self.type_converter = type_converter
        self.verb = kwargs.get("verb")

    @abstractmethod
    async def generate(
        self,
        message: str | MessageParamT | List[MessageParamT],
        request_params: RequestParams | None = None,
    ) -> List[MessageT]:
        """Request an LLM generation, which may run multiple iterations, and return the result"""

    @abstractmethod
    async def generate_str(
        self,
        message: str | MessageParamT | List[MessageParamT],
        request_params: RequestParams | None = None,
    ) -> str:
        """Request an LLM generation and return the string representation of the result"""

    @abstractmethod
    async def generate_structured(
        self,
        message: str | MessageParamT | List[MessageParamT],
        response_model: Type[ModelT],
        request_params: RequestParams | None = None,
    ) -> ModelT:
        """Request a structured LLM generation and return the result as a Pydantic model."""

    async def select_model(self, request_params: RequestParams | None = None) -> str | None:
        """
        Return the configured model (legacy support)
        """
        if request_params and request_params.model:
            return request_params.model

        raise ModelConfigError("Internal Error: Model is not configured correctly")

    def _initialize_default_params(self, kwargs: dict) -> RequestParams:
        """Initialize default parameters for the LLM.
        Should be overridden by provider implementations to set provider-specific defaults."""
        return RequestParams(
            systemPrompt=self.instruction,
            parallel_tool_calls=True,
            max_iterations=10,
            use_history=True,
        )

    def _merge_request_params(self, default_params: RequestParams, provided_params: RequestParams) -> RequestParams:
        """Merge default and provided request parameters"""

        merged = default_params.model_dump()
        merged.update(provided_params.model_dump(exclude_unset=True))
        final_params = RequestParams(**merged)

        return final_params

    def get_request_params(
        self,
        request_params: RequestParams | None = None,
        default: RequestParams | None = None,
    ) -> RequestParams:
        """
        Get request parameters with merged-in defaults and overrides.
        Args:
            request_params: The request parameters to use as overrides.
            default: The default request parameters to use as the base.
                If unspecified, self.default_request_params will be used.
        """
        # Start with the defaults
        default_request_params = default or self.default_request_params

        if not default_request_params:
            default_request_params = self._initialize_default_params({})

        # If user provides overrides, merge them with defaults
        if request_params:
            return self._merge_request_params(default_request_params, request_params)

        return default_request_params

    @classmethod
    def convert_message_to_message_param(cls, message: MessageT, **kwargs: dict[str, Any]) -> MessageParamT:
        """Convert a response object to an input parameter object to allow LLM calls to be chained."""
        # Many LLM implementations will allow the same type for input and output messages
        return cast("MessageParamT", message)

    async def get_last_message(self) -> MessageParamT | None:
        """
        Return the last message generated by the LLM or None if history is empty.
        This is useful for prompt chaining workflows where the last message from one LLM is used as input to another.
        """
        history = self.history.get()
        return history[-1] if history else None

    async def get_last_message_str(self) -> str | None:
        """Return the string representation of the last message generated by the LLM or None if history is empty."""
        last_message = await self.get_last_message()
        return self.message_param_str(last_message) if last_message else None

    def show_tool_result(self, result: CallToolResult) -> None:
        """Display a tool result in a formatted panel."""
        self.display.show_tool_result(result)

    def show_oai_tool_result(self, result: str) -> None:
        """Display a tool result in a formatted panel."""
        self.display.show_oai_tool_result(result)

    def show_tool_call(self, available_tools, tool_name, tool_args) -> None:
        """Display a tool call in a formatted panel."""
        self.display.show_tool_call(available_tools, tool_name, tool_args)

    async def show_assistant_message(
        self,
        message_text: str | Text,
        highlight_namespaced_tool: str = "",
        title: str = "ASSISTANT",
    ) -> None:
        """Display an assistant message in a formatted panel."""
        await self.display.show_assistant_message(
            message_text,
            aggregator=self.aggregator,
            highlight_namespaced_tool=highlight_namespaced_tool,
            title=title,
            name=self.name,
        )

    def show_user_message(self, message, model: str | None, chat_turn: int) -> None:
        """Display a user message in a formatted panel."""
        self.display.show_user_message(message, model, chat_turn, name=self.name)

    async def pre_tool_call(self, tool_call_id: str | None, request: CallToolRequest) -> CallToolRequest | bool:
        """Called before a tool is executed. Return False to prevent execution."""
        return request

    async def post_tool_call(self, tool_call_id: str | None, request: CallToolRequest, result: CallToolResult) -> CallToolResult:
        """Called after a tool execution. Can modify the result before it's returned."""
        return result

    async def call_tool(
        self,
        request: CallToolRequest,
        tool_call_id: str | None = None,
    ) -> CallToolResult:
        """Call a tool with the given parameters and optional ID"""

        try:
            preprocess = await self.pre_tool_call(
                tool_call_id=tool_call_id,
                request=request,
            )

            if isinstance(preprocess, bool):
                if not preprocess:
                    return CallToolResult(
                        isError=True,
                        content=[
                            TextContent(
                                type="text",
                                text=f"Error: Tool '{request.params.name}' was not allowed to run.",
                            )
                        ],
                    )
            else:
                request = preprocess

            tool_name = request.params.name
            tool_args = request.params.arguments
            result = await self.aggregator.call_tool(tool_name, tool_args)

            postprocess = await self.post_tool_call(tool_call_id=tool_call_id, request=request, result=result)

            if isinstance(postprocess, CallToolResult):
                result = postprocess

            return result
        except PromptExitError:
            raise
        except Exception as e:
            return CallToolResult(
                isError=True,
                content=[
                    TextContent(
                        type="text",
                        text=f"Error executing tool '{request.params.name}': {str(e)}",
                    )
                ],
            )

    def message_param_str(self, message: MessageParamT) -> str:
        """
        Convert an input message to a string representation.
        Tries to extract just the content when possible.
        """
        if isinstance(message, dict):
            # For dictionary format messages
            if "content" in message:
                content = message["content"]
                # Handle both string and structured content formats
                if isinstance(content, str):
                    return content
                elif isinstance(content, list) and content:
                    # Try to extract text from content parts
                    text_parts = []
                    for part in content:
                        if isinstance(part, dict) and "text" in part:
                            text_parts.append(part["text"])
                        elif hasattr(part, "text"):
                            text_parts.append(part.text)  # type: ignore
                    if text_parts:
                        return "\n".join(text_parts)

        # For objects with content attribute
        if hasattr(message, "content"):
            content = message.content  # type: ignore
            if isinstance(content, str):
                return content
            elif hasattr(content, "text"):
                return content.text

        # Default fallback
        return str(message)

    def message_str(self, message: MessageT) -> str:
        """
        Convert an output message to a string representation.
        Tries to extract just the content when possible.
        """
        # First try to use the same method for consistency
        result = self.message_param_str(message)  # type: ignore
        if result != str(message):
            return result

        # Additional handling for output-specific formats
        if hasattr(message, "content"):
            content = getattr(message, "content")
            if isinstance(content, list):
                # Extract text from content blocks
                text_parts = []
                for block in content:
                    if hasattr(block, "text") and block.text:
                        text_parts.append(block.text)
                if text_parts:
                    return "\n".join(text_parts)

        # Default fallback
        return str(message)

    def _log_chat_progress(self, chat_turn: Optional[int] = None, model: Optional[str] = None) -> None:
        """Log a chat progress event"""
        # Determine action type based on verb
        if hasattr(self, "verb") and self.verb:
            # Use verb directly regardless of type
            act = self.verb
        else:
            act = ProgressAction.CHATTING

        data = {
            "progress_action": act,
            "model": model,
            "agent_name": self.name,
            "chat_turn": chat_turn if chat_turn is not None else None,
        }
        self.logger.debug("Chat in progress", data=data)

    def _log_chat_finished(self, model: Optional[str] = None) -> None:
        """Log a chat finished event"""
        data = {
            "progress_action": ProgressAction.READY,
            "model": model,
            "agent_name": self.name,
        }
        self.logger.debug("Chat finished", data=data)

    def _convert_prompt_messages(self, prompt_messages: List[PromptMessage]) -> List[MessageParamT]:
        """
        Convert prompt messages to this LLM's specific message format.
        To be implemented by concrete LLM classes.
        """
        raise NotImplementedError("Must be implemented by subclass")

    async def show_prompt_loaded(
        self,
        prompt_name: str,
        description: Optional[str] = None,
        message_count: int = 0,
        arguments: Optional[dict[str, str]] = None,
    ) -> None:
        """
        Display information about a loaded prompt template.

        Args:
            prompt_name: The name of the prompt
            description: Optional description of the prompt
            message_count: Number of messages in the prompt
            arguments: Optional dictionary of arguments passed to the prompt
        """
        await self.display.show_prompt_loaded(
            prompt_name=prompt_name,
            description=description,
            message_count=message_count,
            agent_name=self.name,
            aggregator=self.aggregator,
            arguments=arguments,
        )

    async def apply_prompt_template(self, prompt_result: GetPromptResult, prompt_name: str) -> str:
        """
        Apply a prompt template by adding it to the conversation history.
        If the last message in the prompt is from a user, automatically
        generate an assistant response.

        Args:
            prompt_result: The GetPromptResult containing prompt messages
            prompt_name: The name of the prompt being applied

        Returns:
            String representation of the assistant's response if generated,
            or the last assistant message in the prompt
        """
        from mcp_agent.mcp.prompt_message_multipart import PromptMessageMultipart

        # Check if we have any messages
        if not prompt_result.messages:
            return "Prompt contains no messages"

        # Extract arguments if they were stored in the result
        arguments = getattr(prompt_result, "arguments", None)

        # Display information about the loaded prompt
        await self.show_prompt_loaded(
            prompt_name=prompt_name,
            description=prompt_result.description,
            message_count=len(prompt_result.messages),
            arguments=arguments,
        )

        # Convert to PromptMessageMultipart objects
        multipart_messages = PromptMessageMultipart.parse_get_prompt_result(prompt_result)

        # Delegate to the provider-specific implementation
        return await self._apply_prompt_template_provider_specific(multipart_messages, None)

    async def apply_prompt(
        self,
        multipart_messages: List["PromptMessageMultipart"],
        request_params: RequestParams | None = None,
    ) -> str:
        """
        Apply a list of PromptMessageMultipart messages directly to the LLM.
        This is a cleaner interface to _apply_prompt_template_provider_specific.

        Args:
            multipart_messages: List of PromptMessageMultipart objects
            request_params: Optional parameters to configure the LLM request

        Returns:
            String representation of the assistant's response
        """
        # Delegate to the provider-specific implementation
        return await self._apply_prompt_template_provider_specific(multipart_messages, request_params)

    # this shouln't need to be very big...
    async def _apply_prompt_template_provider_specific(
        self,
        multipart_messages: List["PromptMessageMultipart"],
        request_params: RequestParams | None = None,
    ) -> str:
        """
        Provider-specific implementation of apply_prompt_template.
        This default implementation handles basic text content for any LLM type.
        Provider-specific subclasses should override this method to handle
        multimodal content appropriately.

        Args:
            multipart_messages: List of PromptMessageMultipart objects parsed from the prompt template

        Returns:
            String representation of the assistant's response if generated,
            or the last assistant message in the prompt
        """
        # Check the last message role
        last_message = multipart_messages[-1]

        if last_message.role == "user":
            # For user messages: Add all previous messages to history, then generate response to the last one
            self.logger.debug("Last message in prompt is from user, generating assistant response")

            # Add all but the last message to history
            if len(multipart_messages) > 1:
                previous_messages = multipart_messages[:-1]
                converted = []

                # Fallback generic method for all LLM types
                for msg in previous_messages:
                    # Convert each PromptMessageMultipart to individual PromptMessages
                    prompt_messages = msg.from_multipart()
                    for prompt_msg in prompt_messages:
                        converted.append(self.type_converter.from_prompt_message(prompt_msg))

                self.history.extend(converted, is_prompt=True)

            # For generic LLMs, extract text and describe non-text content
            user_text_parts = []
            for content in last_message.content:
                if content.type == "text":
                    user_text_parts.append(content.text)
                elif content.type == "resource" and getattr(content, "resource", None) is not None:
                    if hasattr(content.resource, "text"):
                        user_text_parts.append(content.resource.text)  # type: ignore
                elif content.type == "image":
                    # Add a placeholder for images
                    mime_type = getattr(content, "mimeType", "image/unknown")
                    user_text_parts.append(f"[Image: {mime_type}]")

            user_text = "\n".join(user_text_parts) if user_text_parts else ""
            if not user_text:
                # Fallback to original method if we couldn't extract text
                user_text = str(last_message.content)

            return await self.generate_str(user_text)
        else:
            # For assistant messages: Add all messages to history and return the last one
            self.logger.debug("Last message in prompt is from assistant, returning it directly")

            # Convert and add all messages to history
            converted = []

            # Fallback to the original method for all LLM types
            for msg in multipart_messages:
                # Convert each PromptMessageMultipart to individual PromptMessages
                prompt_messages = msg.from_multipart()
                for prompt_msg in prompt_messages:
                    converted.append(self.type_converter.from_prompt_message(prompt_msg))

            self.history.extend(converted, is_prompt=True)

            # Return the assistant's message with proper handling of different content types
            assistant_text_parts = []
            has_non_text_content = False

            for content in last_message.content:
                if content.type == "text":
                    assistant_text_parts.append(content.text)
                elif content.type == "resource" and hasattr(content.resource, "text"):
                    # Add resource text with metadata
                    mime_type = getattr(content.resource, "mimeType", "text/plain")
                    uri = getattr(content.resource, "uri", "")
                    if uri:
                        assistant_text_parts.append(
                            f"[Resource: {uri}, Type: {mime_type}]\n{content.resource.text}"  # ignore # type: ignore
                        )
                    else:
                        assistant_text_parts.append(
                            f"[Resource Type: {mime_type}]\n{content.resource.text}"  # type ignore # type: ignore
                        )
                elif content.type == "image":
                    # Note the presence of images
                    mime_type = getattr(content, "mimeType", "image/unknown")
                    assistant_text_parts.append(f"[Image: {mime_type}]")
                    has_non_text_content = True
                else:
                    # Other content types
                    assistant_text_parts.append(f"[Content of type: {content.type}]")
                    has_non_text_content = True

            # Join all parts with double newlines for better readability
            result = "\n\n".join(assistant_text_parts) if assistant_text_parts else str(last_message.content)

            # Add a note if non-text content was present
            if has_non_text_content:
                result += "\n\n[Note: This message contained non-text content that may not be fully represented in text format]"

            return result


#####################################
### NEW INTERFACE METHODS BELOW   ###
#####################################
