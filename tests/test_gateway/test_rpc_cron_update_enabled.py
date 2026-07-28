"""cron.update ``enabled`` toggles revive auto-stopped jobs and keep sibling fields."""

from __future__ import annotations

from pathlib import Path

import pytest

from opensquilla.gateway.rpc import RpcContext
from opensquilla.gateway.rpc_cron import _handle_cron_update, _job_to_wire
from opensquilla.scheduler.engine import SchedulerEngine
from opensquilla.scheduler.jobs import apply_result
from opensquilla.scheduler.payloads import make_agent_turn_payload, payload_text
from opensquilla.scheduler.persistence import JobStore
from opensquilla.scheduler.types import JobExecution, JobStatus, ScheduleKind, SessionTarget


async def _make_engine(tmp_path: Path) -> tuple[SchedulerEngine, JobStore]:
    store = JobStore(str(tmp_path / "cron.db"))
    await store.open()
    return SchedulerEngine(store), store


async def _add_recurring_job(engine: SchedulerEngine):
    return await engine.add_job(
        name="daily-report",
        schedule_kind=ScheduleKind.CRON,
        schedule_value="0 9 * * *",
        handler_key="agent_run",
        payload=make_agent_turn_payload("old prompt"),
        session_target=SessionTarget.ISOLATED,
        jitter_seconds=0,
    )


def _ctx(engine: SchedulerEngine) -> RpcContext:
    return RpcContext(conn_id="test", cron_scheduler=engine)


async def test_update_enabled_true_revives_auto_disabled_job(tmp_path: Path) -> None:
    engine, store = await _make_engine(tmp_path)
    try:
        job = await _add_recurring_job(engine)
        failure = JobExecution(job_id=job.id, success=False, error="provider returned 403")
        await apply_result(job, failure, store)
        disabled = await store.get(job.id)
        assert disabled is not None and disabled.status == JobStatus.DISABLED

        await _handle_cron_update({"id": job.id, "enabled": True}, _ctx(engine))

        after = await store.get(job.id)
        assert after is not None
        assert after.enabled is True
        assert after.status != JobStatus.DISABLED
    finally:
        await store.close()


async def test_update_enabled_true_revives_failed_job(tmp_path: Path) -> None:
    engine, store = await _make_engine(tmp_path)
    try:
        job = await _add_recurring_job(engine)
        for _ in range(5):
            current = await store.get(job.id)
            assert current is not None
            failure = JobExecution(job_id=job.id, success=False, error="network unreachable")
            await apply_result(current, failure, store)
        failed = await store.get(job.id)
        assert failed is not None and failed.status == JobStatus.FAILED

        await _handle_cron_update({"id": job.id, "enabled": True}, _ctx(engine))

        after = await store.get(job.id)
        assert after is not None
        assert after.status != JobStatus.FAILED
        assert after.next_run_at is not None
    finally:
        await store.close()


async def test_update_enabled_true_applies_sibling_fields(tmp_path: Path) -> None:
    engine, store = await _make_engine(tmp_path)
    try:
        job = await _add_recurring_job(engine)
        await engine.pause_job(job.id)

        await _handle_cron_update(
            {"id": job.id, "enabled": True, "text": "new prompt"}, _ctx(engine)
        )

        after = await store.get(job.id)
        assert after is not None
        assert after.status == JobStatus.PENDING
        assert payload_text(after.payload, after.session_target) == "new prompt"
    finally:
        await store.close()


async def test_update_enabled_false_applies_sibling_fields(tmp_path: Path) -> None:
    engine, store = await _make_engine(tmp_path)
    try:
        job = await _add_recurring_job(engine)

        await _handle_cron_update(
            {"id": job.id, "enabled": False, "text": "new prompt"}, _ctx(engine)
        )

        after = await store.get(job.id)
        assert after is not None
        assert after.status == JobStatus.PAUSED
        assert payload_text(after.payload, after.session_target) == "new prompt"
    finally:
        await store.close()


async def test_update_workspace_round_trips_to_wire(tmp_path: Path) -> None:
    engine, store = await _make_engine(tmp_path)
    workspace = tmp_path / "project"
    workspace.mkdir()
    try:
        job = await _add_recurring_job(engine)

        await _handle_cron_update(
            {"id": job.id, "workspaceDir": str(workspace)}, _ctx(engine)
        )

        after = await store.get(job.id)
        assert after is not None
        assert after.workspace_dir == str(workspace.resolve())
        assert _job_to_wire(after)["workspaceDir"] == str(workspace.resolve())
    finally:
        await store.close()


async def test_update_workspace_rejects_missing_directory(tmp_path: Path) -> None:
    engine, store = await _make_engine(tmp_path)
    try:
        job = await _add_recurring_job(engine)

        with pytest.raises(ValueError, match="does not exist"):
            await _handle_cron_update(
                {"id": job.id, "workspaceDir": str(tmp_path / "missing")},
                _ctx(engine),
            )
    finally:
        await store.close()
