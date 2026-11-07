"""Eval history: one row per eval-suite run and per case, so regressions show
up as history instead of overwriting report.json."""

from typing import Any

from sqlalchemy.ext.asyncio import async_sessionmaker

from ..storage.models import EvalCaseRow, EvalRunRow


async def record_eval_run(sessions: async_sessionmaker, report: dict[str, Any]) -> int:
    """Persist one eval-suite report: an eval_runs row plus one eval_cases row
    per case. Returns the run id. Raises on failure — an eval run whose
    history can't be written should fail loudly, unlike chat telemetry."""
    datasets = report.get("datasets") or []
    run = EvalRunRow(
        provider=report["provider"],
        model=report["model"],
        met=all(summary["met"] for summary in datasets) if datasets else False,
        cost_usd=round(sum(summary.get("costUsd") or 0.0 for summary in datasets), 5),
        summary=[
            {key: value for key, value in summary.items() if key != "results"}
            for summary in datasets
        ],
        failure_injection=report.get("failure_injection"),
    )
    async with sessions() as session:
        session.add(run)
        await session.flush()  # assigns run.id for the case rows
        for summary in datasets:
            for row in summary.get("results") or []:
                session.add(
                    EvalCaseRow(
                        run_id=run.id,
                        dataset=summary["dataset"],
                        case_id=row["id"],
                        passed=row["passed"],
                        detail=row.get("detail"),
                        tools=row.get("tools") or [],
                        errors=row.get("errors") or [],
                        cost_usd=row.get("costUsd") or 0.0,
                        latency_ms=row.get("latencyMs") or 0,
                    )
                )
        await session.commit()
    return run.id
