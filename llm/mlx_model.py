"""MLX Language Model wrapper

This module provides a lightweight wrapper around MLX-LM inference
API for loading a quantized language model, formatting prompts using the
model's chat template, and generating both standard and streaming
responses.
"""

import logging

from mlx_lm import load, generate, stream_generate
from mlx_lm.sample_utils import make_sampler

logger = logging.getLogger(__name__)


class MLXModel:
    """Wrapper for running inference with an MLX language model.

    The class manages model loading, prompt formatting, sampling
    configuration, and response generation using the MLX-LM framework.
    Both complete and token-streaming generation modes are supported for
    integration with the RAG pipeline.
    """

    def __init__(
        self,
        model_name: str = "mlx-community/Mistral-7B-Instruct-v0.3-4bit",
        max_tokens: int = 700,
    ):
        self.model_name = model_name
        self.max_tokens = max_tokens

        self.sampler = make_sampler(temp=0.5, top_k=20)

        logger.info("Loading MLX model: %s", self.model_name)
        self.model, self.tokenizer = load(self.model_name)

    def _build_chat_prompt(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]

        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def generate_response(self, prompt: str) -> str:
        chat_prompt = self._build_chat_prompt(prompt)

        return generate(
            self.model,
            self.tokenizer,
            prompt=chat_prompt,
            max_tokens=self.max_tokens,
            sampler=self.sampler,
            verbose=False,
        )

    def stream_response(self, prompt: str):
        chat_prompt = self._build_chat_prompt(prompt)

        for response in stream_generate(
            self.model,
            self.tokenizer,
            prompt=chat_prompt,
            max_tokens=self.max_tokens,
            sampler=self.sampler,
        ):
            yield response.text
