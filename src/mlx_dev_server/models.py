import asyncio
import gc
import logging
import threading
import time
from collections import OrderedDict
from hashlib import md5
from typing import Any, Callable, Generator, Optional

from mlx.nn import Module
from mlx_lm.sample_utils import make_logits_processors, make_sampler
from mlx_lm.tokenizer_utils import TokenizerWrapper
from mlx_lm.utils import GenerationResponse, ModelNotFoundError as NotFoundError, load, stream_generate


__all__ = (
    'Model',
    'ModelNotFoundError',
    'ModelProvider',
    'ModelResponseWrapper',
)


class ModelResponseWrapper:
    """Wrapper for the model response to provide system fingerprint."""
    def __init__(
        self,
        response: GenerationResponse,
        model_params: Optional[dict[str, Any]] = None,
    ) -> None:
        """Initialize the ModelResponseWrapper object.

        Args:
            response (GenerationResponse): The original response from the model.
            model_params (Optional[dict[str, Any]]): The model parameters. If provided, a system fingerprint
                is generated based on the parameters.
        """
        self._response = response
        self._system_fingerprint = 'fp_' + md5(repr(model_params).encode()).hexdigest()[:12]

    def __getattribute__(self, name: str) -> Any:
        """Provide attributes from the original response.

        Args:
            name (str): The name of the attribute to return.
        """
        try:
            return super().__getattribute__(name)
        except AttributeError:
            return getattr(self._response, name)

    @property
    def system_fingerprint(self) -> str:
        """Return the system fingerprint."""
        return self._system_fingerprint


class Model:
    """A class representing a model."""
    def __init__(
        self,
        model: Module,
        tokenizer: TokenizerWrapper,
        *,
        max_tokens: Optional[int] = None,
        max_kv_size: Optional[int] = None,
        kv_bits: Optional[int] = None,
        prefill_step_size: Optional[int] = None,
    ) -> None:
        """Initialize the Model object.

        Args:
            model (mlx.nn.Module): The underlying model.
            tokenizer (mlx_lm.tokenizer_utils.TokenizerWrapper): The tokenizer for the model.
            max_tokens (Optional[int]): The maximum number of tokens to generate.
            max_kv_size (Optional[int]): The maximum size of the key-value cache.
            kv_bits (Optional[int]): The number of bits to use for key-value cache quantization.
            prefill_step_size (Optional[int]): The step size for processing the prompt.
        """
        self._model = model
        self._tokenizer = tokenizer
        self._max_tokens = max_tokens
        self._max_kv_size = max_kv_size
        self._kv_bits = kv_bits
        self._prefill_step_size = prefill_step_size

        self._last_used_time = time.time()
        self._logger = logging.getLogger(__name__)

    @property
    def last_used_time(self):
        """Returns the UNIX timestamp of the last usage of the model."""
        return self._last_used_time

    def apply_chat_template(self, messages: list[dict[str, str]]) -> str:
        """Apply a chat template to the conversation.

        Args:
            messages (list[dict[str, str]]): The list of messages.
        """
        self._last_used_time = time.time()

        if hasattr(self._tokenizer, 'apply_chat_template'):
            return self._tokenizer.apply_chat_template(  # type: ignore
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            self._logger.warning('Model does not support chat template')
            return '\n'.join(f"{m['role'].upper()}: {m['content']}" for m in messages)

    def stream_generate(
        self,
        prompt: str | list[int],
        cancelled: Optional[Callable] = None,
        *,
        temperature: float = 0,
        top_p: float = 0,
        min_p: float = 0,
        top_k: int = -1,
        logit_bias: Optional[dict[int, float]] = None,
        frequency_penalty: Optional[float] = None,
        **kwargs: Any,
    ) -> Generator[ModelResponseWrapper, Any, Any]:
        """Generate text from a prompt using the model and provided parameters.

        Args:
            prompt (str | list[int]): The input prompt.
            cancelled (Optional[Callable]): A callback function to check if the generation should be cancelled.
              Default is ``None`` meaning the generation will not be cancelled.
            temperature (float): The temperature for sampling, if 0 the argmax is used. Default is ``0``.
            top_p (float): Nulceus sampling, higher means model considers
              more less likely words. Default is ``0``.
            min_p (float): The minimum value (scaled by the top token's probability) that a token probability
              must have to be considered. Default is ``0``.
            top_k (int): The top k tokens ranked by probability to constrain
              the sampling to. Default is ``-1``
            logit_bias (Optional[dict[int, float]]): Additive logit bias. Default is ``None``.
            frequency_penalty (Optional[float]): The penalty factor for repeating
              tokens. Default is ``None``.
            **kwargs (Any): Additional keyword arguments to pass to ``mlx_lm.utils.stream_generate()``.
        """
        self._last_used_time = time.time()

        model_params = OrderedDict(sorted(kwargs.items(), key=lambda x: x[0]))
        model_params.update(
            model=self._model,
            tokenizer=self._tokenizer,
            temp=temperature,
            top_p=top_p,
            min_p=min_p,
            top_k=top_k,
            logit_bias=logit_bias,
            frequency_penalty=frequency_penalty,
        )

        if cancelled is not None:
            def prompt_cb(processed: int, total: int) -> None:
                if cancelled():
                    self._logger.warning(
                        'Client disconnected, cancelling prompt processing after %d of %d tokens',
                        processed,
                        total,
                    )
                    raise asyncio.CancelledError('Client disconnected')

            kwargs['prompt_progress_callback'] = prompt_cb

        kwargs['sampler'] = make_sampler(
            temp=temperature,
            top_p=top_p,
            min_p=min_p,
            top_k=top_k,
        )
        kwargs['logits_processors'] = make_logits_processors(
            logit_bias=logit_bias,
            repetition_penalty=frequency_penalty,
        )
        stream = stream_generate(
            self._model,
            self._tokenizer,
            prompt,
            max_tokens=kwargs.pop('max_tokens', self._max_tokens),
            max_kv_size=kwargs.pop('max_kv_size', self._max_kv_size),
            kv_bits=kwargs.pop('kv_bits', self._kv_bits),
            prefill_step_size=kwargs.pop('prefill_step_size', self._prefill_step_size),
            **kwargs,
        )
        for item in stream:
            self._last_used_time = time.time()

            if cancelled is not None and cancelled():
                self._logger.warning(
                    'Client disconnected, cancelling generation after %d tokens',
                    item.generation_tokens
                )
                break

            yield ModelResponseWrapper(item, model_params)


class ModelNotFoundError(NotFoundError):
    """Error raised when the requested model is not found."""


class ModelProvider:
    """Model provider that loads and unloads models on demand."""
    def __init__(
        self,
        keep_alive: int = 5 * 60,
        max_loaded_models: int = -1,
        **kwargs: Any,
    ) -> None:
        """Initialize the ModelProvider object.

        Args:
            keep_alive (int): Time to keep a loaded model alive after the last use. Default is ``5 * 60`` seconds.
            max_loaded_models (int): Maximum number of simultaneously loaded models. Default is ``-1`` (unlimited).
            **kwargs: Default keyword arguments for model's ``mlx_lm.utils.stream_generate()``.
        """
        self._keep_alive = keep_alive
        self._max_loaded_models = max_loaded_models
        self._default_kwargs = kwargs or {}

        self._models: dict[str, Model] = {}
        self._load_locks: dict[str, threading.Lock] = {}
        self._logger = logging.getLogger(__name__)

    def load(self, model_id: str) -> Model:
        """Load a model by the path or repository name.

        Args:
            model_id (str): Path or repository name of the model.
        """
        lock = self._load_locks.setdefault(model_id, threading.Lock())
        with lock:
            if model_id not in self._models:
                if self._max_loaded_models > 0 and len(self._models) >= self._max_loaded_models:
                    self.unload_oldest()

                self._logger.info('Loading model %s', model_id)
                try:
                    model, tokinizer = load(model_id)
                except NotFoundError as e:
                    raise ModelNotFoundError(e.message)
                self._models[model_id] = Model(model, tokinizer, **self._default_kwargs)
            return self._models[model_id]

    def unload(self, model_id: str) -> None:
        """Unload a model by its identifier.

        Args:
            model_id (str): Path or repository name of the model.
        """
        lock = self._load_locks.setdefault(model_id, threading.Lock())
        with lock:
            if model_id in self._models:
                self._logger.info('Unloading model %s', model_id)
                del self._models[model_id]
                gc.collect()

    def unload_oldest(self) -> None:
        """Unload the oldest loaded model."""
        if self._models:
            model_id = min(self._models.items(), key=lambda x: x[1].last_used_time)[0]
            self.unload(model_id)

    async def run_cleaner(self):
        """Periodically clean up unused models."""
        while True:
            now = time.time()
            for model_id in list(self._models.keys()):
                if now - self._models[model_id].last_used_time > self._keep_alive:
                    self.unload(model_id)

            await asyncio.sleep(10)
