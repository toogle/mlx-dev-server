#!/usr/bin/env python3

import asyncio
import contextlib
import logging
import sys
from argparse import ArgumentParser

import colorlog
from aiohttp import web

from ..app import make_app
from ..models import ModelProvider


def main() -> None:
    # Parse command line arguments
    parser = ArgumentParser(sys.argv[0], description='Local OpenAI-compatible API server')
    parser.add_argument(
        '-l',
        '--log-level',
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        help='verbosity level (default is INFO)',
    )
    parser.add_argument(
        '-p',
        '--port',
        type=int,
        default=8080,
        help='port to listen (default is 8080)',
    )
    parser.add_argument(
        '-k',
        '--keep-alive',
        type=int,
        default=5 * 60,
        help='number of seconds that models stay loaded in memory (default is 300 = 5 minutes)',
    )
    parser.add_argument(
        '-m',
        '--max-loaded-models',
        type=int,
        default=2,
        help='maximum number of models to keep loaded (default is 2)',
    )
    parser.add_argument(
        '--host',
        type=str,
        default=None,
        help='host to listen (default is localhost)',
    )
    parser.add_argument(
        '--max-tokens',
        type=int,
        default=4096,
        help='maximum number of tokens to generate if not specified in the request (default 4096)',
    )
    parser.add_argument(
        '--max-kv-size',
        type=int,
        default=4096,
        help='maximum size of the key-value cache (default is 4096)',
    )
    parser.add_argument(
        '--kv-bits',
        type=int,
        default=8,
        help='number of bits to use for key-value cache quantization (default is 8)',
    )
    parser.add_argument(
        '--prefill-step-size',
        type=int,
        default=128,
        help='step size for processing the prompt (default is 128)',
    )

    args = parser.parse_args()

    # Configure logging
    ch = colorlog.StreamHandler()
    fmt = colorlog.ColoredFormatter('%(log_color)s%(asctime)s %(name)s[%(process)d] %(levelname)s %(message)s')
    ch.setFormatter(fmt)

    logging.basicConfig(level=getattr(logging, args.log_level), handlers=[ch])

    # Initialize the model provider
    model_provider = ModelProvider(
        keep_alive=args.keep_alive,
        max_loaded_models=args.max_loaded_models,
        max_tokens=args.max_tokens,
        max_kv_size=args.max_kv_size,
        kv_bits=args.kv_bits,
        prefill_step_size=args.prefill_step_size,
    )

    # Start background task to clean up unused models
    async def background_tasks(app):
        app['models_cleaner'] = asyncio.create_task(model_provider.run_cleaner())

        yield

        app['models_cleaner'].cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await app['models_cleaner']

    # Initialize the application and start the server
    app = make_app(model_provider)
    app.cleanup_ctx.append(background_tasks)
    web.run_app(
        app,
        host=args.host,
        port=args.port,
        print=None,
        access_log_format='%a "%r" %s %b "%{Referer}i" "%{User-Agent}i" %Tfs',
        handler_cancellation=True,
    )


if __name__ == '__main__':
    main()
