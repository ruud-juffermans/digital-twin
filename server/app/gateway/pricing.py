"""Flat USD-per-token rates per provider, for the meta event's costUsd.

Cache discounts aren't metered — cached input is billed at the full input
rate here, so the figure errs on the safe (over-counting) side. DeepSeek
uses its peak rate for the same reason. Rates checked August 2026; update
alongside the model defaults in config.py.
"""

# (input, output) USD per token.
PROVIDER_RATES: dict[str, tuple[float, float]] = {
    "gemini": (0.25 / 1_000_000, 1.50 / 1_000_000),  # gemini-3.1-flash-lite
    "openai": (0.20 / 1_000_000, 1.20 / 1_000_000),  # gpt-5.6-luna
    "deepseek": (0.44 / 1_000_000, 1.32 / 1_000_000),  # deepseek-v4-flash, peak
    "kimi": (0.95 / 1_000_000, 4.00 / 1_000_000),  # kimi-k2.6
}


def cost_usd(provider: str, input_tokens: int, output_tokens: int) -> float:
    rate_in, rate_out = PROVIDER_RATES.get(provider, (0.0, 0.0))
    return input_tokens * rate_in + output_tokens * rate_out
