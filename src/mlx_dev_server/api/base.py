import asyncio
import json
import logging
import time
import uuid
from abc import ABC, abstractmethod
from typing import Any, Iterable, Optional

from aiohttp import web

from ..models import ModelResponseWrapper


# A list of CORS headers
_CORS_HEADERS = {
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Allow-Methods': '*',
    'Access-Control-Allow-Headers': '*',
}


class ResponseMixin:
    """Mixin for API response classes to provide common functionality.

    Constants:
        _RESPONSE_ID_PREFIX (str): A prefix for the unique identifier of the response.
    """
    _RESPONSE_ID_PREFIX = ''

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the object of the base class.

        Args:
            *args: Positional arguments to pass to the base class.
            **kwargs: Keyword arguments to pass to the base class.
        """
        super().__init__(*args, **kwargs)

        self.content_type = 'application/json'
        self.headers.extend(_CORS_HEADERS)  # type: ignore

        self._response_id = f'{self._RESPONSE_ID_PREFIX}{uuid.uuid4()}'
        self._created = int(time.time())

        self._logger = logging.getLogger(__name__)

    @property
    def response_id(self) -> str:
        """Return the unique identifier of the response."""
        return self._response_id

    @property
    def created(self) -> int:
        """Return the UNIX timestamp of the response creation."""
        return self._created


class Response(ResponseMixin, web.Response, ABC):
    """Base class for API responses."""
    def __init__(
        self,
        stream: Iterable[ModelResponseWrapper],
        model_id: Optional[str] = '',
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Initialize the Response object.

        Args:
            stream (Iterable[ModelResponseWrapper]): The stream of responses to be sent.
            model_id (Optional[str]): The model identifier.
            *args: Positional arguments to pass to the ``web.Response`` class.
            **kwargs: Keyword arguments to pass to the ``web.Response`` class.
        """
        super().__init__(*args, **kwargs)

        self._stream = stream
        self._model_id = model_id

    def _join_items_sync(self) -> tuple[str, str | None, int, int, str]:
        """Join the text chunks of the stream response."""
        text = ''
        finish_reason = None
        prompt_tokens = 0
        generation_tokens = 0
        system_fingerprint = ''

        for item in self._stream:
            text += item.text
            finish_reason = item.finish_reason
            prompt_tokens = item.prompt_tokens
            generation_tokens = item.generation_tokens
            system_fingerprint = item.system_fingerprint

        return text, finish_reason, prompt_tokens, generation_tokens, system_fingerprint

    @abstractmethod
    def _prepare_response_object(
        self,
        choices: list[str],
        finish_reason: Optional[str],
        prompt_tokens: int,
        completion_tokens: int,
        system_fingerprint: str,
    ) -> dict[str, Any]:
        """Prepare the object for the API response.

        Args:
            choices (list[str]): The list of generated responses.
            finish_reason (Optional[str]): The finish reason for the response.
            prompt_tokens (int): The number of tokens used in the prompt.
            completion_tokens (int): The number of tokens generated.
            system_fingerprint (str): The system fingerprint for the response.
        """
        pass

    async def prepare_json(self) -> None:
        """Prepare the JSON response and assign it to the ``text`` attribute."""
        loop = asyncio.get_event_loop()
        future = loop.run_in_executor(None, self._join_items_sync)
        text, finish_reason, prompt_tokens, completion_tokens, system_fingerprint = await future

        response = self._prepare_response_object(
            [text],
            finish_reason,
            prompt_tokens,
            completion_tokens,
            system_fingerprint,
        )
        self.text = json.dumps(response, ensure_ascii=False)


class StreamResponse(ResponseMixin, web.StreamResponse, ABC):
    """Base class for streaming API responses."""
    def __init__(
        self,
        model_id: Optional[str] = '',
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Initialize the StreamResponse object.

        Args:
            *args: Positional arguments to pass to the ``web.StreamResponse`` class.
            **kwargs: Keyword arguments to pass to the ``web.StreamResponse`` class.
        """
        super().__init__(*args, **kwargs)

        self.content_type = 'text/event-stream'
        self.headers['Cache-Control'] = 'no-cache'

        self._model_id = model_id

    @abstractmethod
    def _prepare_chunk_object(
        self,
        text: str,
        finish_reason: Optional[str],
        system_fingerprint: str,
    ) -> dict[str, Any]:
        """Prepare the chunk object for the response stream.

        Args:
            text (str): The generated text.
            finish_reason (Optional[str]): The finish reason for the response chunk.
            system_fingerprint (str): The system fingerprint for the response chunk.
        """
        pass

    def _write_json_sync(self, stream: Iterable[Any], loop: asyncio.AbstractEventLoop) -> None:
        """Syncronously write the JSON stream to the response.

        This function is intended to run in executor to avoid blocking the event loop.

        Args:
            stream (Iterable[Any]): The stream of generated items to write.
            loop (asyncio.AbstractEventLoop): The event loop to use.
        """
        for item in stream:
            response = self._prepare_chunk_object(item.text, item.finish_reason, item.system_fingerprint)
            chunk = b'data: ' + json.dumps(response, ensure_ascii=False).encode('utf-8') + b'\n\n'
            asyncio.run_coroutine_threadsafe(self.write(chunk), loop).result()

        asyncio.run_coroutine_threadsafe(self.write(b'data: [DONE]\n'), loop).result()

    def write_json(self, stream: Iterable[Any]) -> asyncio.Future:
        """Write the JSON stream to the response.

        Args:
            stream (Iterable[Any]): The stream of generated items to write.
        """
        loop = asyncio.get_event_loop()
        return loop.run_in_executor(None, self._write_json_sync, stream, loop)
