import os
from typing import List, Type, TYPE_CHECKING

from mcp_agent.workflows.llm.providers.multipart_converter_anthropic import (
    AnthropicConverter,
)
from mcp_agent.workflows.llm.providers.sampling_converter_anthropic import (
    AnthropicSamplingConverter,
)

if TYPE_CHECKING:
    from mcp_agent.mcp.prompt_message_multipart import PromptMessageMultipart


from anthropic import Anthropic, AuthenticationError
from anthropic.types import (
    Message,
    MessageParam,
    TextBlock,
    TextBlockParam,
    ToolParam,
    ToolUseBlockParam,
)
from mcp.types import (
    CallToolRequestParams,
    CallToolRequest,
)
from pydantic_core import from_json

from mcp_agent.workflows.llm.augmented_llm import (
    AugmentedLLM,
    ModelT,
    RequestParams,
)
from mcp_agent.core.exceptions import ProviderKeyError
from rich.text import Text

from mcp_agent.logging.logger import get_logger

DEFAULT_ANTHROPIC_MODEL = "claude-3-7-sonnet-latest"


class AnthropicAugmentedLLM(AugmentedLLM[MessageParam, Message]):
    """
    The basic building block of agentic systems is an LLM enhanced with augmentations
    such as retrieval, tools, and memory provided from a collection of MCP servers.
    Our current models can actively use these capabilities—generating their own search queries,
    selecting appropriate tools, and determining what information to retain.
    """

    def __init__(self, *args, **kwargs):
        self.provider = "Anthropic"
        # Initialize logger - keep it simple without name reference
        self.logger = get_logger(__name__)

        # Now call super().__init__
        super().__init__(*args, type_converter=AnthropicSamplingConverter, **kwargs)

    def _initialize_default_params(self, kwargs: dict) -> RequestParams:
        """Initialize Anthropic-specific default parameters"""
        return RequestParams(
            model=kwargs.get("model", DEFAULT_ANTHROPIC_MODEL),
            modelPreferences=self.model_preferences,
            maxTokens=4096,  # default haiku3
            systemPrompt=self.instruction,
            parallel_tool_calls=True,
            max_iterations=10,
            use_history=True,
        )

    async def generate(
        self,
        message,
        request_params: RequestParams | None = None,
    ):
        """
        Process a query using an LLM and available tools.
        Override this method to use a different LLM.
        """

        api_key = self._api_key(self.context.config)
        try:
            anthropic = Anthropic(api_key=api_key)
            messages: List[MessageParam] = []
            params = self.get_request_params(request_params)
        except AuthenticationError as e:
            raise ProviderKeyError(
                "Invalid Anthropic API key",
                "The configured Anthropic API key was rejected.\n"
                "Please check that your API key is valid and not expired.",
            ) from e

        # Always include prompt messages, but only include conversation history
        # if use_history is True
        messages.extend(self.history.get(include_history=params.use_history))

        if isinstance(message, str):
            messages.append({"role": "user", "content": message})
        elif isinstance(message, list):
            messages.extend(message)
        else:
            messages.append(message)

        response = await self.aggregator.list_tools()
        available_tools: List[ToolParam] = [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.inputSchema,
            }
            for tool in response.tools
        ]

        responses: List[Message] = []
        model = await self.select_model(params)
        chat_turn = (len(messages) + 1) // 2
        self.show_user_message(str(message), model, chat_turn)

        for i in range(params.max_iterations):
            chat_turn = (len(messages) + 1) // 2
            self._log_chat_progress(chat_turn, model=model)
            arguments = {
                "model": model,
                "messages": messages,
                "system": self.instruction or params.systemPrompt,
                "stop_sequences": params.stopSequences,
                "tools": available_tools,
            }

            if params.maxTokens is not None:
                arguments["max_tokens"] = params.maxTokens

            if params.metadata:
                arguments = {**arguments, **params.metadata}

            self.logger.debug(f"{arguments}")

            executor_result = await self.executor.execute(
                anthropic.messages.create, **arguments
            )

            response = executor_result[0]

            if isinstance(response, AuthenticationError):
                raise ProviderKeyError(
                    "Invalid Anthropic API key",
                    "The configured Anthropic API key was rejected.\n"
                    "Please check that your API key is valid and not expired.",
                ) from response
            elif isinstance(response, BaseException):
                error_details = str(response)
                self.logger.error(f"Error: {error_details}", data=executor_result)

                # Try to extract more useful information for API errors
                if hasattr(response, "status_code") and hasattr(response, "response"):
                    try:
                        error_json = response.response.json()
                        error_details = (
                            f"Error code: {response.status_code} - {error_json}"
                        )
                    except:  # noqa: E722
                        error_details = (
                            f"Error code: {response.status_code} - {str(response)}"
                        )

                # Convert other errors to text response
                error_message = f"Error during generation: {error_details}"
                response = Message(
                    id="error",  # Required field
                    model="error",  # Required field
                    role="assistant",
                    type="message",
                    content=[TextBlock(type="text", text=error_message)],
                    stop_reason="end_turn",  # Must be one of the allowed values
                    usage={"input_tokens": 0, "output_tokens": 0},  # Required field
                )

            self.logger.debug(
                f"{model} response:",
                data=response,
            )

            response_as_message = self.convert_message_to_message_param(response)
            messages.append(response_as_message)
            responses.append(response)

            if response.stop_reason == "end_turn":
                message_text = ""
                for block in response_as_message["content"]:
                    if isinstance(block, dict) and block.get("type") == "text":
                        message_text += block.get("text", "")
                    elif hasattr(block, "type") and block.type == "text":
                        message_text += block.text

                await self.show_assistant_message(message_text)

                self.logger.debug(
                    f"Iteration {i}: Stopping because finish_reason is 'end_turn'"
                )
                break
            elif response.stop_reason == "stop_sequence":
                # We have reached a stop sequence
                self.logger.debug(
                    f"Iteration {i}: Stopping because finish_reason is 'stop_sequence'"
                )
                break
            elif response.stop_reason == "max_tokens":
                # We have reached the max tokens limit

                self.logger.debug(
                    f"Iteration {i}: Stopping because finish_reason is 'max_tokens'"
                )
                if params.maxTokens is not None:
                    message_text = Text(
                        f"the assistant has reached the maximum token limit ({params.maxTokens})",
                        style="dim green italic",
                    )
                else:
                    message_text = Text(
                        "the assistant has reached the maximum token limit",
                        style="dim green italic",
                    )

                await self.show_assistant_message(message_text)

                break
            else:
                message_text = ""
                for block in response_as_message["content"]:
                    if isinstance(block, dict) and block.get("type") == "text":
                        message_text += block.get("text", "")
                    elif hasattr(block, "type") and block.type == "text":
                        message_text += block.text

                # response.stop_reason == "tool_use":
                # First, collect all tool uses in this turn
                tool_uses = [c for c in response.content if c.type == "tool_use"]

                if tool_uses:
                    if message_text == "":
                        message_text = Text(
                            "the assistant requested tool calls",
                            style="dim green italic",
                        )

                    # Process all tool calls and collect results
                    tool_results = []
                    for i, content in enumerate(tool_uses):
                        tool_name = content.name
                        tool_args = content.input
                        tool_use_id = content.id

                        if i == 0:  # Only show message for first tool use
                            await self.show_assistant_message(message_text, tool_name)

                        self.show_tool_call(available_tools, tool_name, tool_args)
                        tool_call_request = CallToolRequest(
                            method="tools/call",
                            params=CallToolRequestParams(
                                name=tool_name, arguments=tool_args
                            ),
                        )
                        # TODO -- support MCP isError etc.
                        result = await self.call_tool(
                            request=tool_call_request, tool_call_id=tool_use_id
                        )
                        self.show_tool_result(result)

                        # Add each result to our collection
                        tool_results.append((tool_use_id, result))

                    messages.append(
                        AnthropicConverter.create_tool_results_message(tool_results)
                    )

        # Only save the new conversation messages to history if use_history is true
        # Keep the prompt messages separate
        if params.use_history:
            # Get current prompt messages
            prompt_messages = self.history.get(include_history=False)

            # Calculate new conversation messages (excluding prompts)
            new_messages = messages[len(prompt_messages) :]

            # Update conversation history
            self.history.set(new_messages)

        self._log_chat_finished(model=model)

        return responses

    def _api_key(self, config):
        api_key = None

        if hasattr(config, "anthropic") and config.anthropic:
            api_key = config.anthropic.api_key
            if api_key == "<your-api-key-here>":
                api_key = None

        if api_key is None:
            api_key = os.getenv("ANTHROPIC_API_KEY")

        if not api_key:
            raise ProviderKeyError(
                "Anthropic API key not configured",
                "The Anthropic API key is required but not set.\n"
                "Add it to your configuration file under anthropic.api_key "
                "or set the ANTHROPIC_API_KEY environment variable.",
            )

        return api_key

    async def generate_str(
        self,
        message,
        request_params: RequestParams | None = None,
    ) -> str:
        """
        Process a query using an LLM and available tools.
        The default implementation uses Claude as the LLM.
        Override this method to use a different LLM.

        Special commands:
        - "***SAVE_HISTORY <filename.md>" - Saves the conversation history to the specified file
          in MCP prompt format with user/assistant delimiters.
        """
        # Check if this is a special command to save history
        if isinstance(message, str) and message.startswith("***SAVE_HISTORY "):
            return await self._save_history_to_file(message)

        responses: List[Message] = await self.generate(
            message=message,
            request_params=request_params,
        )

        final_text: List[str] = []

        # Process all responses and collect all text content
        for response in responses:
            # Extract text content from each message
            message_text = ""
            for content in response.content:
                if content.type == "text":
                    # Extract text from text blocks
                    message_text += content.text

            # Only append non-empty text
            if message_text:
                final_text.append(message_text)

        # TODO -- make tool detail inclusion behaviour configurable
        # Join all collected text
        return "\n".join(final_text)

    async def generate_prompt(
        self, prompt: "PromptMessageMultipart", request_params: RequestParams | None
    ) -> str:
        return await self.generate_str(
            AnthropicConverter.convert_to_anthropic(prompt), request_params
        )

    async def _apply_prompt_template_provider_specific(
        self, multipart_messages: List["PromptMessageMultipart"]
    ) -> str:
        """
        Anthropic-specific implementation of apply_prompt_template that handles
        multimodal content natively.

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
            self.logger.debug(
                "Last message in prompt is from user, generating assistant response"
            )

            # Add all but the last message to history
            if len(multipart_messages) > 1:
                previous_messages = multipart_messages[:-1]
                converted = []

                # Convert all previous messages to Anthropic format
                for msg in previous_messages:
                    converted.append(AnthropicConverter.convert_to_anthropic(msg))

                self.history.extend(converted, is_prompt=True)

            # Convert the last message to Anthropic format and generate a response
            message_param = AnthropicConverter.convert_to_anthropic
            (last_message)
            return await self.generate_str(message_param)
        else:
            # For assistant messages: Add all messages to history and return the last one
            self.logger.debug(
                "Last message in prompt is from assistant, returning it directly"
            )

            # Convert and add all messages to history
            converted = []

            # Convert all messages to Anthropic format
            for msg in multipart_messages:
                converted.append(AnthropicConverter.convert_to_anthropic(msg))

            self.history.extend(converted, is_prompt=True)

            # Process the last message content for display
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
                            f"[Resource: {uri}, Type: {mime_type}]\n{content.resource.text}"
                        )
                    else:
                        assistant_text_parts.append(
                            f"[Resource Type: {mime_type}]\n{content.resource.text}"
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
            result = (
                "\n\n".join(assistant_text_parts)
                if assistant_text_parts
                else str(last_message.content)
            )

            # Add a note if non-text content was present
            if has_non_text_content:
                result += "\n\n[Note: This message contained non-text content that may not be fully represented in text format]"

            return result

    async def _save_history_to_file(self, command: str) -> str:
        """
        Save the conversation history to a file in MCP prompt format.

        Args:
            command: The command string, expected format: "***SAVE_HISTORY <filename.md>"

        Returns:
            Success or error message
        """
        try:
            # Extract the filename from the command
            parts = command.split(" ", 1)
            if len(parts) != 2 or not parts[1].strip():
                return "Error: Invalid format. Expected '***SAVE_HISTORY <filename.md>'"

            filename = parts[1].strip()

            # Get all messages from history
            messages = self.history.get(include_history=True)

            # Import required utilities
            from mcp_agent.workflows.llm.anthropic_utils import (
                anthropic_message_param_to_prompt_message_multipart,
            )
            from mcp_agent.mcp.prompt_serialization import (
                multipart_messages_to_delimited_format,
            )

            # Convert message params to PromptMessageMultipart objects
            multipart_messages = []
            for msg in messages:
                multipart_messages.append(
                    anthropic_message_param_to_prompt_message_multipart(msg)
                )

            # Convert to delimited format
            delimited_content = multipart_messages_to_delimited_format(
                multipart_messages,
                user_delimiter="---USER",
                assistant_delimiter="---ASSISTANT",
            )

            # Write to file
            with open(filename, "w", encoding="utf-8") as f:
                f.write("\n\n".join(delimited_content))

            self.logger.info(f"Saved conversation history to {filename}")
            return f"Done. Saved conversation history to {filename}"

        except Exception as e:
            self.logger.error(f"Error saving history: {str(e)}")
            return f"Error saving history: {str(e)}"

    async def generate_structured(
        self,
        message,
        response_model: Type[ModelT],
        request_params: RequestParams | None = None,
    ) -> ModelT:
        # TODO -- simiar to the OAI version, we should create a tool call for the expected schema
        response = await self.generate_str(
            message=message,
            request_params=request_params,
        )
        # Don't try to parse if we got no response
        if not response:
            self.logger.error("No response from generate_str")
            return None

        return response_model.model_validate(from_json(response, allow_partial=True))

    @classmethod
    def convert_message_to_message_param(
        cls, message: Message, **kwargs
    ) -> MessageParam:
        """Convert a response object to an input parameter object to allow LLM calls to be chained."""
        content = []

        for content_block in message.content:
            if content_block.type == "text":
                content.append(TextBlockParam(type="text", text=content_block.text))
            elif content_block.type == "tool_use":
                content.append(
                    ToolUseBlockParam(
                        type="tool_use",
                        name=content_block.name,
                        input=content_block.input,
                        id=content_block.id,
                    )
                )

        return MessageParam(role="assistant", content=content, **kwargs)

    def message_param_str(self, message: MessageParam) -> str:
        """Convert an input message to a string representation."""

        if message.get("content"):
            content = message["content"]
            if isinstance(content, str):
                return content
            else:
                final_text: List[str] = []
                for block in content:
                    if block.text:
                        final_text.append(str(block.text))
                    else:
                        final_text.append(str(block))

                return "\n".join(final_text)

        return str(message)

    def message_str(self, message: Message) -> str:
        """Convert an output message to a string representation."""
        content = message.content

        if content:
            if isinstance(content, list):
                final_text: List[str] = []
                for block in content:
                    if block.text:
                        final_text.append(str(block.text))
                    else:
                        final_text.append(str(block))

                return "\n".join(final_text)
            else:
                return str(content)

        return str(message)
