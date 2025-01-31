import json
import time
from collections import namedtuple
from dataclasses import dataclass
from typing import Any, Optional

import asyncio
import pytest
from aiohttp.test_utils import make_mocked_request
from mlx_lm.utils import ModelNotFoundError
from pytest_unordered import unordered

from mlx_dev_server.app import make_app
from mlx_dev_server.models import ModelProvider


@dataclass
class MockGenerationResponse:
    text: str
    prompt_tokens: int = 0
    generation_tokens: int = 0
    finish_reason: Optional[str] = None
    system_fingerprint: str = 'fp_test'


class MockLoad:
    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        self.args = ()
        self.kwargs = {}

    def __call__(self, model_id: str, *args, **kwargs) -> tuple[Any, Any]:
        self.args = (model_id, *args)
        self.kwargs = dict(kwargs)

        if self.model_id != model_id:
            raise ModelNotFoundError('model not found')
        return None, None


class MockStreamGenerate:
    def __init__(self, num_responses: int, prefix: str = 'chunk '):
        self._num_responses = num_responses
        self._prefix = prefix
        self.args = ()
        self.kwargs = {}

    def __call__(self, *args, prompt_progress_callback=None, **kwargs):
        self.args = args
        self.kwargs = dict(kwargs, prompt_progress_callback=prompt_progress_callback)

        # Prompt processing step
        if prompt_progress_callback is not None:
            for i in range(1, 9):
                prompt_progress_callback(128 * i, 1024)

        # Generation step
        prompt_tokens = 2
        generation_tokens = 4
        for i in range(1, self._num_responses):
            yield MockGenerationResponse(f'{self._prefix}{i} ', prompt_tokens, generation_tokens)
            prompt_tokens += 2
            generation_tokens += 4
        yield MockGenerationResponse(f'{self._prefix}{self._num_responses}', prompt_tokens, generation_tokens, 'stop')

    @property
    def expected_text_parts(self) -> list[str]:
        parts = [f'{self._prefix}{i} ' for i in range(1, self._num_responses)]
        parts.append(f'{self._prefix}{self._num_responses}')
        return parts

    @property
    def expected_text_full(self) -> str:
        return ''.join(self.expected_text_parts)


@pytest.fixture
def app():
    return make_app(ModelProvider())

@pytest.fixture
async def client(aiohttp_client, app):
    return await aiohttp_client(app)


class TestApiHandlerModels:
    async def test_models(self, client, monkeypatch):
        def mock_scan_cache_dir():
            Result = namedtuple('Result', ['repos'])
            Repo = namedtuple('Repo', ['repo_id', 'repo_type', 'last_modified'])
            return Result([
                Repo('test/model-1', 'model', 1682534400),
                Repo('test/model-2', 'model', 1712534400),
                Repo('test/dataset-1', 'dataset', 1713534400),
                Repo('test/dataset-2', 'dataset', 1710574400),
            ])

        monkeypatch.setattr('mlx_dev_server.handlers.scan_cache_dir', mock_scan_cache_dir)

        resp = await client.get('/v1/models')
        assert resp.status == 200

        actual = await resp.json()
        assert actual['object'] == 'list'

        expected_data = [
            {
                'id': 'test/model-1',
                'object': 'model',
                'created': 1682534400,
                'owned_by': 'local',
            }, {
                'id': 'test/model-2',
                'object': 'model',
                'created': 1712534400,
                'owned_by': 'local',
            },
        ]
        assert actual['data'] == unordered(expected_data)


class TestApiHandlerCompletions:
    # Constants for MockLoad()
    _MODEL_ID = 'test/model-1'

    # Constants for MockStreamGenerate()
    _NUM_RESPONSES = 3
    _PREFIX = 'chunk '

    @pytest.fixture
    def mock_load(self, monkeypatch):
        load = MockLoad(self._MODEL_ID)
        monkeypatch.setattr('mlx_dev_server.models.load', load)
        return load

    @pytest.fixture
    def mock_stream_generate(self, monkeypatch):
        stream_generate = MockStreamGenerate(self._NUM_RESPONSES, self._PREFIX)
        monkeypatch.setattr('mlx_dev_server.models.stream_generate', stream_generate)
        return stream_generate

    @staticmethod
    def _assert_cors_headers(headers):
        assert headers['access-control-allow-origin'] == '*'
        assert headers['access-control-allow-methods'] == '*'
        assert headers['access-control-allow-headers'] == '*'

    @staticmethod
    def _assert_response(resp, model_id, text):
        assert resp['id'].startswith('cmpl-')
        assert resp['object'] == 'text_completion'
        assert resp['created'] > time.time() - 1000
        assert resp['model'] == model_id
        assert resp['system_fingerprint'].startswith('fp_')

        choices = resp['choices']
        assert len(choices) == 1
        assert choices[0]['text'] == text
        assert choices[0]['index'] == 0
        assert choices[0]['logprobs'] is None
        assert choices[0]['finish_reason'] == 'stop'

        usage = resp['usage']
        assert usage['prompt_tokens'] == 2 * 3
        assert usage['completion_tokens'] == 4 * 3
        assert usage['total_tokens'] == (2 + 4) * 3

    @staticmethod
    def _assert_response_chunk(chunk, model_id, text, finish_reason):
        assert chunk['id'].startswith('cmpl-')
        assert chunk['object'] == 'text_completion'
        assert chunk['created'] > time.time() - 1000
        assert chunk['model'] == model_id
        assert chunk['system_fingerprint'].startswith('fp_')

        choices = chunk['choices']
        assert len(choices) == 1
        assert choices[0]['text'] == text
        assert choices[0]['index'] == 0
        assert choices[0]['logprobs'] is None
        assert choices[0]['finish_reason'] == finish_reason

    async def test_completions_str(self, client, mock_load, mock_stream_generate):
        resp = await client.post('/v1/completions', json={
            'model': self._MODEL_ID,
            'prompt': 'test prompt',
            'max_tokens': 123,
        })
        assert resp.status == 200
        self._assert_cors_headers(resp.headers)
        self._assert_response(await resp.json(), self._MODEL_ID, mock_stream_generate.expected_text_full)

        assert mock_load.args[0] == self._MODEL_ID
        assert mock_stream_generate.kwargs['max_tokens'] == 123

    async def test_completions_list_of_int(self, client, mock_load, mock_stream_generate):
        resp = await client.post('/v1/completions', json={
            'model': self._MODEL_ID,
            'prompt': [1, 2, 3, 4, 5],
            'max_tokens': 123,
        })
        assert resp.status == 200
        self._assert_cors_headers(resp.headers)
        self._assert_response(await resp.json(), self._MODEL_ID, mock_stream_generate.expected_text_full)

        assert mock_load.args[0] == self._MODEL_ID
        assert mock_stream_generate.kwargs['max_tokens'] == 123

    async def test_completions_list_of_str(self, client, mock_load, mock_stream_generate):
        resp = await client.post('/v1/completions', json={
            'model': self._MODEL_ID,
            'prompt': ['test prompt 1', 'test prompt 2', 'test prompt 3'],
            'max_tokens': 123,
        })
        assert resp.status == 200
        self._assert_cors_headers(resp.headers)
        self._assert_response(await resp.json(), self._MODEL_ID, mock_stream_generate.expected_text_full)

        assert mock_load.args[0] == self._MODEL_ID
        assert mock_stream_generate.kwargs['max_tokens'] == 123

    async def test_completions_prompt_processing_cancel(self, app, mock_load, mock_stream_generate):
        async def req_text():
            return json.dumps({
                'model': self._MODEL_ID,
                'prompt': 'test prompt',
                'max_tokens': 123,
            })

        req = make_mocked_request(
            'POST',
            '/v1/completions',
            {'Content-Type': 'application/json'},
            app=app,
        )
        req.text = lambda: req_text()
        req._task = asyncio.create_task(asyncio.sleep(0))
        req.task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await app._handle(req)

        assert mock_load.args[0] == self._MODEL_ID
        assert mock_stream_generate.kwargs['max_tokens'] == 123

    async def test_completions_stream(self, client, mock_load, mock_stream_generate):
        resp = await client.post('/v1/completions', json={
            'model': self._MODEL_ID,
            'prompt': 'test prompt',
            'max_tokens': 123,
            'stream': True,
        })
        assert resp.status == 200
        assert resp.content_type == 'text/event-stream'
        assert resp.headers['cache-control'] == 'no-cache'
        self._assert_cors_headers(resp.headers)

        i = 1
        async for chunk in resp.content:
            if chunk != b'\n':
                assert chunk.startswith(b'data: ')
                if not chunk.endswith(b'[DONE]\n'):
                    data = json.loads(chunk[5:])  # skip "data: " prefix
                    expected_text = mock_stream_generate.expected_text_parts[i - 1]
                    expected_finish_reason = None if i < len(mock_stream_generate.expected_text_parts) else 'stop'
                    self._assert_response_chunk(data, self._MODEL_ID, expected_text, expected_finish_reason)
                    i += 1

        assert mock_load.args[0] == self._MODEL_ID
        assert mock_stream_generate.kwargs['max_tokens'] == 123

    async def test_completions_stream_generation_cancel(self, app, mock_load, mock_stream_generate, monkeypatch):
        async def req_text():
            return json.dumps({
                'model': self._MODEL_ID,
                'prompt': 'test prompt',
                'max_tokens': 123,
                'stream': True,
            })

        async def req_task():
            while True:
                await asyncio.sleep(0)

        req = make_mocked_request(
            'POST',
            '/v1/completions',
            {'Content-Type': 'application/json'},
            app=app,
        )
        req.text = lambda: req_text()
        req._task = asyncio.create_task(req_task())

        i = 1
        async def resp_write(_, chunk):
            nonlocal i
            if i <= 2:
                assert chunk.startswith(b'data: ')
                data = json.loads(chunk[5:])  # skip "data: " prefix
                expected_text = mock_stream_generate.expected_text_parts[i - 1]
                expected_finish_reason = None if i < len(mock_stream_generate.expected_text_parts) else 'stop'
                self._assert_response_chunk(data, self._MODEL_ID, expected_text, expected_finish_reason)
                if i == 2:
                    req.task.cancel()
                    await asyncio.sleep(0)
            else:
                assert i == 3
                assert chunk == b'data: [DONE]\n'
            i += 1

        monkeypatch.setattr('aiohttp.web.StreamResponse.write', resp_write)

        resp = await app._handle(req)
        assert resp.status == 200
        assert resp.content_type == 'text/event-stream'
        assert resp.headers['cache-control'] == 'no-cache'
        self._assert_cors_headers(resp.headers)

        assert mock_load.args[0] == self._MODEL_ID
        assert mock_stream_generate.kwargs['max_tokens'] == 123

    async def test_missing_model(self, client):
        resp = await client.post('/v1/completions', json={
            'prompt': 'test prompt',
        })
        assert resp.status == 400

        actual = await resp.json()
        expected = {
            'error': {
                'message': 'you must provide a model parameter',
                'type': 'invalid_request_error',
                'param': None,
                'code': None,
            },
        }
        assert actual == expected

    async def test_model_is_not_str(self, client):
        resp = await client.post('/v1/completions', json={
            'model': 123,
            'prompt': 'test prompt',
        })
        assert resp.status == 400

        actual = await resp.json()
        expected = {
            'error': {
                'message': 'model parameter must be a string',
                'type': 'invalid_request_error',
                'param': None,
                'code': None,
            },
        }
        assert actual == expected

    async def test_missing_prompt(self, client):
        resp = await client.post('/v1/completions', json={
            'model': self._MODEL_ID,
        })
        assert resp.status == 400

        actual = await resp.json()
        expected = {
            'error': {
                'message': "Missing required parameter: 'prompt'.",
                'type': 'invalid_request_error',
                'param': 'prompt',
                'code': 'missing_required_parameter',
            },
        }
        assert actual == expected

    async def test_prompt_is_not_str_or_list(self, client):
        resp = await client.post('/v1/completions', json={
            'model': self._MODEL_ID,
            'prompt': 123,
        })
        assert resp.status == 400

        actual = await resp.json()
        expected = {
            'error': {
                'message': "Invalid type for 'prompt': expected one of a string or array of strings, integers, or integer arrays, but got int instead.",
                'type': 'invalid_request_error',
                'param': 'prompt',
                'code': 'invalid_type',
            },
        }
        assert actual == expected

    async def test_prompt_is_list_with_invalid_type(self, client):
        resp = await client.post('/v1/completions', json={
            'model': self._MODEL_ID,
            'prompt': [
                'test prompt',
                {},
            ],
        })
        assert resp.status == 400

        actual = await resp.json()
        expected = {
            'error': {
                'message': "Invalid type for 'prompt[1]': expected strings, integers, or integer arrays, but got dict instead.",
                'type': 'invalid_request_error',
                'param': 'prompt[1]',
                'code': 'invalid_type',
            },
        }
        assert actual == expected

    async def test_model_not_found(self, client, mock_load):
        resp = await client.post('/v1/completions', json={
            'model': 'test/non-existent-model',
            'prompt': 'test prompt',
        })
        assert resp.status == 404

        actual = await resp.json()
        expected = {
            'error': {
                'message': "The model `test/non-existent-model` does not exist or you do not have access to it.",
                'type': 'invalid_request_error',
                'param': None,
                'code': 'model_not_found',
            },
        }
        assert actual == expected

        assert mock_load.args[0] == 'test/non-existent-model'


class TestApiHandlerChatCompletions:
    # Constants for MockLoad()
    _MODEL_ID = 'test/model-1'

    # Constants for MockStreamGenerate()
    _NUM_RESPONSES = 3
    _PREFIX = 'chunk '

    @pytest.fixture
    def mock_load(self, monkeypatch):
        load = MockLoad(self._MODEL_ID)
        monkeypatch.setattr('mlx_dev_server.models.load', load)
        return load

    @pytest.fixture
    def mock_stream_generate(self, monkeypatch):
        stream_generate = MockStreamGenerate(self._NUM_RESPONSES, self._PREFIX)
        monkeypatch.setattr('mlx_dev_server.models.stream_generate', stream_generate)
        return stream_generate

    @staticmethod
    def _assert_cors_headers(headers):
        assert headers['access-control-allow-origin'] == '*'
        assert headers['access-control-allow-methods'] == '*'
        assert headers['access-control-allow-headers'] == '*'

    @staticmethod
    def _assert_response(resp, model_id, text):
        assert resp['id'].startswith('chatcmpl-')
        assert resp['object'] == 'chat.completion'
        assert resp['created'] > time.time() - 1000
        assert resp['model'] == model_id
        assert resp['system_fingerprint'].startswith('fp_')

        choices = resp['choices']
        assert len(choices) == 1
        assert choices[0]['index'] == 0
        assert choices[0]['message']['role'] == 'assistant'
        assert choices[0]['message']['content'] == text
        assert choices[0]['logprobs'] is None
        assert choices[0]['finish_reason'] == 'stop'

        usage = resp['usage']
        assert usage['prompt_tokens'] == 2 * 3
        assert usage['completion_tokens'] == 4 * 3
        assert usage['total_tokens'] == (2 + 4) * 3

        details = usage['completion_tokens_details']
        assert details['reasoning_tokens'] == 0
        assert details['accepted_prediction_tokens'] == 0
        assert details['rejected_prediction_tokens'] == 0

    @staticmethod
    def _assert_response_chunk(chunk, model_id, text, finish_reason):
        assert chunk['id'].startswith('chatcmpl-')
        assert chunk['object'] == 'chat.completion.chunk'
        assert chunk['created'] > time.time() - 1000
        assert chunk['model'] == model_id
        assert chunk['system_fingerprint'].startswith('fp_')

        choices = chunk['choices']
        assert len(choices) == 1
        assert choices[0]['index'] == 0
        assert choices[0]['delta']['role'] == 'assistant'
        assert choices[0]['delta']['content'] == text
        assert choices[0]['logprobs'] is None
        assert choices[0]['finish_reason'] == finish_reason

    async def test_chat_completions(self, client, mock_load, mock_stream_generate):
        resp = await client.post('/v1/chat/completions', json={
            'model': self._MODEL_ID,
            'messages': [{
                'role': 'user',
                'content': 'test prompt',
            }],
            'max_tokens': 123,
        })
        assert resp.status == 200
        self._assert_cors_headers(resp.headers)
        self._assert_response(await resp.json(), self._MODEL_ID, mock_stream_generate.expected_text_full)

        assert mock_load.args[0] == self._MODEL_ID
        assert mock_stream_generate.kwargs['max_tokens'] == 123

    async def test_chat_completions_stream(self, client, mock_load, mock_stream_generate):
        resp = await client.post('/v1/chat/completions', json={
            'model': self._MODEL_ID,
            'messages': [{
                'role': 'user',
                'content': 'test prompt',
            }],
            'max_tokens': 123,
            'stream': True,
        })
        assert resp.status == 200
        assert resp.content_type == 'text/event-stream'
        assert resp.headers['cache-control'] == 'no-cache'
        self._assert_cors_headers(resp.headers)

        i = 1
        async for chunk in resp.content:
            if chunk != b'\n':
                assert chunk.startswith(b'data: ')
                if not chunk.endswith(b'[DONE]\n'):
                    data = json.loads(chunk[5:])  # skip "data: " prefix
                    expected_text = mock_stream_generate.expected_text_parts[i - 1]
                    expected_finish_reason = None if i < len(mock_stream_generate.expected_text_parts) else 'stop'
                    self._assert_response_chunk(data, self._MODEL_ID, expected_text, expected_finish_reason)
                    i += 1

        assert mock_load.args[0] == self._MODEL_ID
        assert mock_stream_generate.kwargs['max_tokens'] == 123

    async def test_missing_model(self, client):
        resp = await client.post('/v1/chat/completions', json={
            'messages': [{
                'role': 'user',
                'content': 'test prompt',
            }],
        })
        assert resp.status == 400

        actual = await resp.json()
        expected = {
            'error': {
                'message': 'you must provide a model parameter',
                'type': 'invalid_request_error',
                'param': None,
                'code': None,
            },
        }
        assert actual == expected

    async def test_model_is_not_str(self, client):
        resp = await client.post('/v1/chat/completions', json={
            'model': 123,
            'messages': [{
                'role': 'user',
                'content': 'test prompt',
            }],
        })
        assert resp.status == 400

        actual = await resp.json()
        expected = {
            'error': {
                'message': 'model parameter must be a string',
                'type': 'invalid_request_error',
                'param': None,
                'code': None,
            },
        }
        assert actual == expected

    async def test_missing_messages(self, client):
        resp = await client.post('/v1/chat/completions', json={
            'model': self._MODEL_ID,
        })
        assert resp.status == 400

        actual = await resp.json()
        expected = {
            'error': {
                'message': "Missing required parameter: 'messages'.",
                'type': 'invalid_request_error',
                'param': 'messages',
                'code': 'missing_required_parameter',
            },
        }
        assert actual == expected

    async def test_messages_invalid_type(self, client):
        resp = await client.post('/v1/chat/completions', json={
            'model': self._MODEL_ID,
            'messages': {
                'role': 'user',
                'content': 'test prompt',
            },
        })
        assert resp.status == 400

        actual = await resp.json()
        expected = {
            'error': {
                'message': "Invalid type for 'messages': expected an array of objects, but got dict instead.",
                'type': 'invalid_request_error',
                'param': 'messages',
                'code': 'invalid_type',
            },
        }
        assert actual == expected

    async def test_messages_contains_invalid_type(self, client):
        resp = await client.post('/v1/chat/completions', json={
            'model': self._MODEL_ID,
            'messages': [
                {
                    'role': 'user',
                    'content': 'test prompt',
                },
                123,
            ],
        })
        assert resp.status == 400

        actual = await resp.json()
        expected = {
            'error': {
                'message': "Invalid type for 'messages[1]': expected an object, but got int instead.",
                'type': 'invalid_request_error',
                'param': 'messages[1]',
                'code': 'invalid_type',
            },
        }
        assert actual == expected

    async def test_messages_contains_object_without_role(self, client):
        resp = await client.post('/v1/chat/completions', json={
            'model': self._MODEL_ID,
            'messages': [{
                'myrole': 'user',
                'content': 'test prompt',
            }],
        })
        assert resp.status == 400

        actual = await resp.json()
        expected = {
            'error': {
                'message': "Missing required parameter: 'messages[0].role'.",
                'type': 'invalid_request_error',
                'param': 'messages[0].role',
                'code': 'missing_required_parameter',
            },
        }
        assert actual == expected

    async def test_messages_contains_object_without_content(self, client):
        resp = await client.post('/v1/chat/completions', json={
            'model': self._MODEL_ID,
            'messages': [{
                'role': 'user',
                'mycontent': 'test prompt',
            }],
        })
        assert resp.status == 400

        actual = await resp.json()
        expected = {
            'error': {
                'message': "Missing required parameter: 'messages[0].content'.",
                'type': 'invalid_request_error',
                'param': 'messages[0].content',
                'code': 'missing_required_parameter',
            },
        }
        assert actual == expected

    async def test_messages_contains_object_with_invalid_role_type(self, client):
        resp = await client.post('/v1/chat/completions', json={
            'model': self._MODEL_ID,
            'messages': [{
                'role': 123,
                'content': 'test prompt',
            }],
        })
        assert resp.status == 400

        actual = await resp.json()
        expected = {
            'error': {
                'message': "Invalid type for 'messages[0].role': expected one of 'system', 'assistant', 'user', 'function', 'tool', or 'developer', but got int instead.",
                'type': 'invalid_request_error',
                'param': 'messages[0].role',
                'code': 'invalid_type',
            },
        }
        assert actual == expected

    async def test_messages_contains_object_with_invalid_content_type(self, client):
        resp = await client.post('/v1/chat/completions', json={
            'model': self._MODEL_ID,
            'messages': [{
                'role': 'user',
                'content': 123,
            }],
        })
        assert resp.status == 400

        actual = await resp.json()
        expected = {
            'error': {
                'message': "Invalid type for 'messages[0].content': expected one of a string or array of objects, but got int instead.",
                'type': 'invalid_request_error',
                'param': 'messages[0].content',
                'code': 'invalid_type'
            }
        }
        assert actual == expected

    async def test_messages_contains_object_with_invalid_role_value(self, client):
        resp = await client.post('/v1/chat/completions', json={
            'model': self._MODEL_ID,
            'messages': [{
                'role': 'test',
                'content': 'test prompt',
            }],
        })
        assert resp.status == 400

        actual = await resp.json()
        expected = {
            'error': {
                'message': "Invalid value: 'test'. Supported values are: 'system', 'assistant', 'user', 'function', 'tool', and 'developer'.",
                'type': 'invalid_request_error',
                'param': 'messages[0].role',
                'code': 'invalid_value',
            },
        }
        assert actual == expected

    async def test_model_not_found(self, client, mock_load):
        resp = await client.post('/v1/chat/completions', json={
            'model': 'test/non-existent-model',
            'messages': [{
                'role': 'user',
                'content': 'test prompt',
            }],
        })
        assert resp.status == 404

        actual = await resp.json()
        expected = {
            'error': {
                'message': "The model `test/non-existent-model` does not exist or you do not have access to it.",
                'type': 'invalid_request_error',
                'param': None,
                'code': 'model_not_found',
            },
        }
        assert actual == expected

        assert mock_load.args[0] == 'test/non-existent-model'
