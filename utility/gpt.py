import os
import concurrent.futures
import json
from dataclasses import dataclass
from typing import Annotated, Any, Literal, Sequence

from retry import retry
from openai import (
    OpenAI as OpenAIClient,
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    RateLimitError,
    BadRequestError,
)

# ---------------------------------------------------------------------
# Model constants
# ---------------------------------------------------------------------

MODEL_STRONG = "gpt-5.4"
MODEL_CHEAP = "gpt-5.4-mini"


# ---------------------------------------------------------------------
# Options & pricing
# ---------------------------------------------------------------------


@dataclass
class LMOptions:
    """Generation options for GPT-5.1 / GPT-5-mini."""

    # Max completion tokens (visible output, not counting prompt).
    # Hard upper bound from docs is 128,000 for these models.
    
    # max_completion_tokens: int = 10000
    
    max_completion_tokens: int = 100000000
    # max_completion_tokens: int = 1000

    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0

    # Default reasoning effort is **medium** (per your request).
    # Valid values: "none", "low", "medium", "high".
    reasoning_effort: Literal["none", "low", "medium", "high"] = "medium"


# Standard-tier pricing, per 1M tokens (approximate).
PRICING_PER_M_TOKEN: dict[str, dict[str, float]] = {
    MODEL_STRONG: {
        "input": 1.25,
        "cached_input": 0.125,
        "output": 10.00,
    },
    MODEL_CHEAP: {
        "input": 0.25,
        "cached_input": 0.025,
        "output": 2.00,
    },
}


@dataclass
class GptResponse:
    """Normalized wrapper around an OpenAI chat completion."""

    content: str
    model: str

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

    reasoning_tokens: int = 0
    cached_input_tokens: int = 0
    cost: float = 0.0  # USD

    @classmethod
    def from_chat_completion(cls, response: Any) -> "GptResponse":
        """Builds a GptResponse from `client.chat.completions.create(...)`."""
        usage = getattr(response, "usage", None)

        prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
        completion_tokens = getattr(usage, "completion_tokens", 0) or 0
        total_tokens = getattr(usage, "total_tokens", prompt_tokens + completion_tokens) or 0

        # Cached input tokens (discounted price).
        prompt_details = getattr(usage, "prompt_tokens_details", None)
        if prompt_details is not None:
            cached_input_tokens = getattr(prompt_details, "cached_tokens", 0) or 0
        else:
            cached_input_tokens = 0

        # Reasoning tokens (subset of completion_tokens, not extra).
        completion_details = getattr(usage, "completion_tokens_details", None)
        if completion_details is not None:
            reasoning_tokens = getattr(completion_details, "reasoning_tokens", 0) or 0
        else:
            reasoning_tokens = 0

        inst = cls(
            content=response.choices[0].message.content,
            model=response.model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            reasoning_tokens=reasoning_tokens,
            cached_input_tokens=cached_input_tokens,
        )
        inst.cost = inst._calculate_cost()
        return inst

    def _base_model_name(self) -> str:
        # Handles suffixes like `gpt-5.1-2025-08-07` or `gpt-5-mini:whatever`
        return self.model.split(":", 1)[0]

    def _calculate_cost(self) -> float:
        """Approximate cost (USD) using standard-tier pricing."""
        model_name = self._base_model_name()
        pricing = None
        for prefix, row in PRICING_PER_M_TOKEN.items():
            if model_name.startswith(prefix):
                pricing = row
                break

        if pricing is None:
            return 0.0

        per_m_input = pricing["input"]
        per_m_cached_input = pricing["cached_input"]
        per_m_output = pricing["output"]

        input_price = per_m_input / 1_000_000.0
        cached_input_price = per_m_cached_input / 1_000_000.0
        output_price = per_m_output / 1_000_000.0

        uncached_input = max(self.prompt_tokens - self.cached_input_tokens, 0)

        return (
            uncached_input * input_price
            + self.cached_input_tokens * cached_input_price
            + self.completion_tokens * output_price
        )


# ---------------------------------------------------------------------
# OpenAI wrapper (only gpt-5.1 and gpt-5-mini)
# ---------------------------------------------------------------------


class OpenAI:
    """Thin wrapper over OpenAI chat API with batching + cost estimation."""

    _RETRY_EXCEPTIONS = (APIConnectionError, APITimeoutError, InternalServerError, RateLimitError, BadRequestError)

    def __init__(
        self,
        model,
        max_workers: int = 6,
        api_key: str | None = None,
    ) -> None:
        self.model = model
        self.max_workers = max_workers
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is not set.")
        self._client = OpenAIClient(api_key=self.api_key)

    def _get_request_args(self, options: LMOptions) -> dict[str, Any]:
        """Translate LMOptions -> chat.completions kwargs."""
        # Enforce a hard upper bound for safety.
        max_tokens = min(options.max_completion_tokens, 128_000)
        return dict(
            model=self.model,
            max_completion_tokens=max_tokens,
            top_p=options.top_p,
            frequency_penalty=options.frequency_penalty,
            presence_penalty=options.presence_penalty,
            reasoning_effort=options.reasoning_effort,  # default is "medium"
        )

    # ------------- single-call helpers -------------

    @retry(_RETRY_EXCEPTIONS, delay=1, backoff=2, max_delay=4)
    def chat(
        self,
        prompt: str,
        options: LMOptions | None = None,
    ) -> GptResponse:
        """Simple text-in, text-out call."""
        if options is None:
            options = LMOptions()

        response = self._client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            **self._get_request_args(options),
        )
        return GptResponse.from_chat_completion(response)

    @retry(_RETRY_EXCEPTIONS, delay=1, backoff=2, max_delay=4)
    def chat_messages(
        self,
        messages: list[dict[str, str]],
        options: LMOptions | None = None,
    ) -> GptResponse:
        """Call with a pre-built messages list."""
        if options is None:
            options = LMOptions()

        response = self._client.chat.completions.create(
            messages=messages,
            **self._get_request_args(options),
        )
        return GptResponse.from_chat_completion(response)

    # ------------- batch helpers -------------

    def chat_batch(
        self,
        prompts: Sequence[str],
        options: LMOptions | None = None,
    ) -> list[GptResponse]:
        """Run multiple string prompts in parallel."""
        if not prompts:
            return []
        if options is None:
            options = LMOptions()

        max_workers = min(self.max_workers, len(prompts))

        def _run(p: str) -> GptResponse:
            return self.chat(p, options=options)

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            return list(executor.map(_run, prompts))

    def chat_batch_messages(
        self,
        messages_list: Sequence[list[dict[str, str]]],
        options: LMOptions | None = None,
    ) -> list[GptResponse]:
        """Run multiple message lists in parallel."""
        if not messages_list:
            return []
        if options is None:
            options = LMOptions()

        max_workers = min(self.max_workers, len(messages_list))

        def _run(msgs: list[dict[str, str]]) -> GptResponse:
            return self.chat_messages(msgs, options=options)

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            return list(executor.map(_run, messages_list))


# ---------------------------------------------------------------------
# Simple factory helpers (optional but convenient)
# ---------------------------------------------------------------------


# def StrongModel() -> OpenAI:
#     """High-quality default model (GPT-5.1)."""
#     return OpenAI(model=MODEL_STRONG)


# def CheapModel() -> OpenAI:
#     """Cheaper workhorse (GPT-5-mini)."""
#     return OpenAI(model=MODEL_CHEAP)
