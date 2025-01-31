from aiohttp import web

from .handlers import APIHandler
from .models import ModelProvider


def make_app(model_provider: ModelProvider) -> web.Application:
    """Return prepared application.

    Args:
        model_provider (ModelProvider): The model provider instance.
    """
    handler = APIHandler(model_provider)
    app = web.Application()
    app.add_routes([
        web.get('/v1/models', handler.handle_models),
        web.post('/v1/completions', handler.handle_completions),
        web.post('/v1/chat/completions', handler.handle_chat_completions),
    ])
    return app
