import asyncio
import json
import logging

from aiohttp import web
from huggingface_hub import scan_cache_dir

from .api import (
    ChatCompletionResponse,
    CompletionResponse,
    StreamChatCompletionResponse,
    StreamCompletionResponse,
)
from .models import ModelProvider, ModelNotFoundError


__all__ = ('APIHandler', )


class APIHandler:
    """Handle endpoints compatible with OpenAI API.

    API Reference: https://platform.openai.com/docs/api-reference
    """
    def __init__(self, model_provider: ModelProvider) -> None:
        """Initialize the APIHandler object.

        Args:
            model_provider (ModelProvider): Model provider that loads and unloads models on demand.
        """
        self._model_provider = model_provider

        self._logger = logging.getLogger(__name__)

    async def handle_models(self, _: web.Request) -> web.Response:
        """List models in the local cache.

        Args:
            _ (web.Request): Request object (not used).
        """
        hf_cache_info = scan_cache_dir()
        repos = sorted(hf_cache_info.repos, key=lambda r: r.last_modified, reverse=True)
        data = [
            {
                'id': repo.repo_id,
                'object': repo.repo_type,
                'created': int(repo.last_modified),
                'owned_by': 'local',
            } for repo in repos
            if repo.repo_type == 'model'
        ]
        return web.json_response({
            'object': 'list',
            'data': data,
        })

    async def handle_completions(self, request: web.Request) -> web.StreamResponse:
        """Handle text completion request.

        Args:
            request (web.Request): Request object.

        Raises:
            web.HTTPBadRequest: Raised if the request is invalid.
            web.HTTPNotFound: Raised if the requested model is not found.
        """
        data = await request.json()

        # Get required parameters
        model_id = data.get('model')
        if model_id is None:
            err = {
                'error': {
                    'message': 'you must provide a model parameter',
                    'type': 'invalid_request_error',
                    'param': None,
                    'code': None,
                },
            }
            raise web.HTTPBadRequest(text=json.dumps(err), content_type='application/json')
        elif not isinstance(model_id, str):
            err = {
                'error': {
                    'message': 'model parameter must be a string',
                    'type': 'invalid_request_error',
                    'param': None,
                    'code': None,
                },
            }
            raise web.HTTPBadRequest(text=json.dumps(err), content_type='application/json')

        prompt = data.get('prompt')
        if prompt is None:
            err = {
                'error': {
                    'message': "Missing required parameter: 'prompt'.",
                    'type': 'invalid_request_error',
                    'param': 'prompt',
                    'code': 'missing_required_parameter',
                },
            }
            raise web.HTTPBadRequest(text=json.dumps(err), content_type='application/json')
        elif not isinstance(prompt, (str, list)):
            err = {
                'error': {
                    'message': f"Invalid type for 'prompt': expected one of a string or array of strings, integers, or integer arrays, but got {type(prompt).__name__} instead.",
                    'type': 'invalid_request_error',
                    'param': 'prompt',
                    'code': 'invalid_type',
                },
            }
            raise web.HTTPBadRequest(text=json.dumps(err), content_type='application/json')
        elif isinstance(prompt, list):
            indexes = [i for i, x in enumerate(prompt) if not isinstance(x, (str, int, list))]
            if indexes:
                err = {
                    'error': {
                        'message': f"Invalid type for 'prompt[{indexes[0]}]': expected strings, integers, or integer arrays, but got {type(prompt[indexes[0]]).__name__} instead.",
                        'type': 'invalid_request_error',
                        'param': f'prompt[{indexes[0]}]',
                        'code': 'invalid_type',
                    },
                }
                raise web.HTTPBadRequest(text=json.dumps(err), content_type='application/json')

        # Filter supported parameters
        supported_params = {
            # 'best_of',
            # 'echo',
            'frequency_penalty',
            'logit_bias',
            # 'logprobs',
            'max_tokens',
            # 'n',
            # 'presence_penalty',
            # 'seed',
            # 'stop',
            # 'stream_options',
            # 'suffix',
            'temperature',
            'top_p',
            # 'user',
        }
        params = {k: data[k] for k in supported_params if k in data}

        # Load the model
        future = asyncio.get_event_loop().run_in_executor(None, self._model_provider.load, model_id)
        try:
            model = await asyncio.shield(future)
        except ModelNotFoundError:
            err = {
                'error': {
                    'message': f"The model `{model_id}` does not exist or you do not have access to it.",
                    'type': 'invalid_request_error',
                    'param': None,
                    'code': 'model_not_found',
                },
            }
            raise web.HTTPNotFound(text=json.dumps(err), content_type='application/json')

        # Generate the response
        stream = model.stream_generate(prompt, request.task.cancelled, **params)
        if data.get('stream', False):
            response = StreamCompletionResponse(model_id)
            await response.prepare(request)
            await response.write_json(stream)
        else:
            response = CompletionResponse(stream, model_id)
            await response.prepare_json()

        return response

    async def handle_chat_completions(self, request: web.Request) -> web.StreamResponse:
        """Handle chat completions request.

        Args:
            request (web.Request): Request object.

        Raises:
            web.HTTPBadRequest: Raised if the request is invalid.
            web.HTTPNotFound: Raised if the requested model is not found.
        """
        data = await request.json()

        # Get required parameters
        model_id = data.get('model')
        if model_id is None:
            err = {
                'error': {
                    'message': 'you must provide a model parameter',
                    'type': 'invalid_request_error',
                    'param': None,
                    'code': None,
                },
            }
            raise web.HTTPBadRequest(text=json.dumps(err), content_type='application/json')
        elif not isinstance(model_id, str):
            err = {
                'error': {
                    'message': 'model parameter must be a string',
                    'type': 'invalid_request_error',
                    'param': None,
                    'code': None,
                },
            }
            raise web.HTTPBadRequest(text=json.dumps(err), content_type='application/json')

        messages = data.get('messages')
        if messages is None:
            err = {
                'error': {
                    'message': "Missing required parameter: 'messages'.",
                    'type': 'invalid_request_error',
                    'param': 'messages',
                    'code': 'missing_required_parameter',
                },
            }
            raise web.HTTPBadRequest(text=json.dumps(err), content_type='application/json')
        elif not isinstance(messages, list):
            err = {
                'error': {
                    'message': f"Invalid type for 'messages': expected an array of objects, but got {type(messages).__name__} instead.",
                    'type': 'invalid_request_error',
                    'param': 'messages',
                    'code': 'invalid_type',
                },
            }
            raise web.HTTPBadRequest(text=json.dumps(err), content_type='application/json')
        else:
            for i, m in enumerate(messages):
                if not isinstance(m, dict):
                    err = {
                        'error': {
                            'message': f"Invalid type for 'messages[{i}]': expected an object, but got {type(m).__name__} instead.",
                            'type': 'invalid_request_error',
                            'param': f'messages[{i}]',
                            'code': 'invalid_type',
                        },
                    }
                    raise web.HTTPBadRequest(text=json.dumps(err), content_type='application/json')
                else:
                    for k in ('role', 'content'):
                        if k not in m:
                            err = {
                                'error': {
                                    'message': f"Missing required parameter: 'messages[{i}].{k}'.",
                                    'type': 'invalid_request_error',
                                    'param': f'messages[{i}].{k}',
                                    'code': 'missing_required_parameter',
                                },
                            }
                            raise web.HTTPBadRequest(text=json.dumps(err), content_type='application/json')
                        elif not isinstance(m[k], str):
                            err_messages = {
                                'role': f"expected one of 'system', 'assistant', 'user', 'function', 'tool', or 'developer', but got {type(m[k]).__name__} instead.",
                                'content': f'expected one of a string or array of objects, but got {type(m[k]).__name__} instead.',
                            }
                            err = {
                                'error': {
                                    'message': f"Invalid type for 'messages[{i}].{k}': {err_messages[k]}",
                                    'type': 'invalid_request_error',
                                    'param': f'messages[{i}].{k}',
                                    'code': 'invalid_type',
                                },
                            }
                            raise web.HTTPBadRequest(text=json.dumps(err), content_type='application/json')

                        if m['role'] not in ('system', 'assistant', 'user', 'function', 'tool', 'developer'):
                            err = {
                                'error': {
                                    'message': f"Invalid value: '{m['role']}'. Supported values are: 'system', 'assistant', 'user', 'function', 'tool', and 'developer'.",
                                    'type': 'invalid_request_error',
                                    'param': f'messages[{i}].role',
                                    'code': 'invalid_value',
                                },
                            }
                            raise web.HTTPBadRequest(text=json.dumps(err), content_type='application/json')

        # Filter supported parameters
        supported_params = {
            # 'store',
            # 'reasoning_effort',
            # 'metadata',
            'frequency_penalty',
            'logit_bias',
            # 'logprobs',
            'top_logprobs',
            'max_tokens',
            # 'max_completion_tokens',
            # 'modalities',
            # 'prediction',
            # 'audio',
            # 'presence_penalty',
            # 'response_format',
            # 'seed',
            # 'service_tier',
            # 'stop',
            # 'stream_options',
            'temperature',
            'top_p',
            # 'tools',
            # 'tool_choice',
            # 'parallel_tool_calls',
            # 'user',
            # 'function_call',
            # 'functions',
        }
        params = {k: data[k] for k in supported_params if k in data}

        # Load the model
        future = asyncio.get_event_loop().run_in_executor(None, self._model_provider.load, model_id)
        try:
            model = await asyncio.shield(future)
        except ModelNotFoundError:
            err = {
                'error': {
                    'message': f"The model `{model_id}` does not exist or you do not have access to it.",
                    'type': 'invalid_request_error',
                    'param': None,
                    'code': 'model_not_found',
                },
            }
            raise web.HTTPNotFound(text=json.dumps(err), content_type='application/json')

        # Prepare the prompt
        prompt = model.apply_chat_template(messages)

        # Generate the response
        stream = model.stream_generate(prompt, request.task.cancelled, **params)
        if data.get('stream', False):
            response = StreamChatCompletionResponse(model_id)
            await response.prepare(request)
            await response.write_json(stream)
        else:
            response = ChatCompletionResponse(stream, model_id)
            await response.prepare_json()

        return response
