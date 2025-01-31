from typing import Any, Optional

from .base import Response, StreamResponse


class CompletionResponse(Response):
    """A class representing a completion response."""
    _RESPONSE_ID_PREFIX = 'cmpl-'

    def _prepare_response_object(
        self,
        choices: list[str],
        finish_reason: Optional[str],
        prompt_tokens: int,
        completion_tokens: int,
        system_fingerprint: str,
    ) -> dict[str, Any]:
        """Prepare the completion object for the API response.

        Args:
            choices (list[str]): The list of generated responses.
            finish_reason (Optional[str]): The finish reason for the response.
            prompt_tokens (int): The number of tokens used in the prompt.
            completion_tokens (int): The number of tokens generated.
            system_fingerprint (str): The system fingerprint for the response.
        """
        response = {
            'id': self.response_id,
            'object': 'text_completion',
            'created': self.created,
            'model': self._model_id,
            'system_fingerprint': system_fingerprint,
            'choices': [],
            'usage': {
                'prompt_tokens': prompt_tokens,
                'completion_tokens': completion_tokens,
                'total_tokens': prompt_tokens + completion_tokens,
            },
        }
        for index, text in enumerate(choices):
            response['choices'].append({
                'text': text,
                'index': index,
                'logprobs': None,
                'finish_reason': finish_reason,
            })

        return response


class StreamCompletionResponse(StreamResponse):
    """A class representing a stream completion response."""
    _RESPONSE_ID_PREFIX = 'cmpl-'

    def _prepare_chunk_object(
        self,
        text: str,
        finish_reason: Optional[str],
        system_fingerprint: str,
    ) -> dict[str, Any]:
        """Prepare the completion object for the response stream.

        Args:
            text (str): The generated text.
            finish_reason (Optional[str]): The finish reason for the response chunk.
            system_fingerprint (str): The system fingerprint for the response chunk.
        """
        return {
            'id': self.response_id,
            'object': 'text_completion',
            'created': self.created,
            'choices': [{
                'text': text,
                'index': 0,
                'logprobs': None,
                'finish_reason': finish_reason,
            }],
            'model': self._model_id,
            'system_fingerprint': system_fingerprint,
        }
