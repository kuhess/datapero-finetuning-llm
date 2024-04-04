from __future__ import annotations

from typing import TYPE_CHECKING

from clients.ollm.models.create_chat_completion_request import (
    CreateChatCompletionRequest,
)
from octoai.chat import TEXT_DEFAULT_ENDPOINT, TEXT_SECURELINK_ENDPOINT, TextModel

if TYPE_CHECKING:
    from clients.ollm.models.chat_completion_response_format import (
        ChatCompletionResponseFormat,
    )
    from clients.ollm.models.chat_message import ChatMessage
    from octoai.client import Client, InferenceFuture


class CompletionsAsync:
    """Text completions API."""

    client: Client
    endpoint: str = TEXT_DEFAULT_ENDPOINT

    def __init__(self, client: Client) -> None:
        self.client = client

        if self.client.secure_link:
            self.endpoint = TEXT_SECURELINK_ENDPOINT

    async def create(
        self,
        *,
        messages: list[ChatMessage | dict[str, str]],
        model: str | TextModel,
        frequency_penalty: float | None = 0.0,
        max_tokens: int | None = None,
        presence_penalty: float | None = 0.0,
        response_format: ChatCompletionResponseFormat | None = None,
        stop: str | None = None,
        temperature: float | None = 1.0,
        top_p: float | None = 1.0,
    ) -> InferenceFuture:
        """Create a chat completion with a text generation model.

        :param messages: Required. A list of messages to use as context for the
            completion.
        :param model: Required. The model to use for the completion. Supported models
            are listed in the `octoai.chat.TextModel` enum.
        :param frequency_penalty: Positive values make it less likely that the model
            repeats tokens several times in the completion. Valid values are between
            -2.0 and 2.0.
        :param max_tokens: The maximum number of tokens to generate.
        :param presence_penalty: Positive values make it less likely that the model
            repeats tokens in the completion. Valid values are between -2.0 and 2.0.
        :param response_format: An object specifying the format that the model must
            output.
        :param stop: A list of sequences where the model stops generating tokens.
        :param stream: Whether to return a generator that yields partial message
            deltas as they become available, instead of waiting to return the entire
            response.
        :param temperature: Sampling temperature. A value between 0 and 2. Higher values
            make the model more creative by sampling less likely tokens.
        :param top_p: The cumulative probability of the most likely tokens to use. Use
            `temperature` or `top_p` but not both.
        """
        request = CreateChatCompletionRequest(
            messages=messages,
            model=model.value if isinstance(model, TextModel) else model,
            frequency_penalty=frequency_penalty,
            function_call=None,
            functions=None,
            logit_bias=None,
            max_tokens=max_tokens,
            n=1,
            presence_penalty=presence_penalty,
            response_format=response_format,
            stop=stop,
            temperature=temperature,
            top_p=top_p,
            user=None,
        )

        inputs = request.to_dict()

        return self.client.infer_async(self.endpoint, inputs)


class ChatAsync:
    """Chat API for text generation models."""

    completions: CompletionsAsync

    def __init__(self, client: Client) -> None:
        self.completions = CompletionsAsync(client)
