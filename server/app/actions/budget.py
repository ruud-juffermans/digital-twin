"""Daily cost budget for the agent, fail-closed.

Plain deterministic Python: the graph checks `exceeded` before spending any
tokens and adds each run's cost after verification. In-memory and resetting
at midnight; Phase 9 seeds today's spend from a ledger at startup via
`restore`, making the cap restart-proof.
"""

import time


class BudgetTracker:
    def __init__(self, daily_limit_usd: float):
        self.daily_limit_usd = daily_limit_usd
        self._day: str | None = None
        self._spent = 0.0

    def _roll_day(self) -> None:
        today = time.strftime("%Y-%m-%d")
        if today != self._day:
            self._day = today
            self._spent = 0.0

    @property
    def spent_usd(self) -> float:
        self._roll_day()
        return round(self._spent, 5)

    @property
    def exceeded(self) -> bool:
        self._roll_day()
        return self.daily_limit_usd > 0 and self._spent >= self.daily_limit_usd

    def add(self, cost_usd: float) -> None:
        self._roll_day()
        self._spent += cost_usd

    def restore(self, spent_usd: float) -> None:
        """Seed today's spend from a durable ledger (startup only)."""
        self._day = time.strftime("%Y-%m-%d")
        self._spent = spent_usd
