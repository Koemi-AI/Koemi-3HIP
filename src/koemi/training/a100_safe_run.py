"""Conservative A100 launcher for a small, resumable code-focused run.

The launcher deliberately keeps the corpus bounded and does not embed data or
the runner source in a notebook. ``plan`` is local-only, ``preflight`` runs a
small real-device forward/backward check, and ``train`` requires an explicit
budget confirmation before opening the remote dataset streams.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
import time
from typing import Any

import torch

from koemi.model.execution import ExecutionMode
from koemi.model.network import KoemiModel
from koemi.training import a100_run as canonical
from koemi.training.objective import calculate_training_objective


SAFE_RUN_FORMAT_VERSION = 1
COST_PER_HOUR_USD = 6.33
DEFAULT_BUDGET_HOURS = 188.0
DEFAULT_SESSION_HOURS = 7.5
DEFAULT_SEQUENCE_LENGTH = 256
DEFAULT_NUM_WORKERS = 2
DEFAULT_CHECKPOINT_MINUTES = 10
DEFAULT_LOG_INTERVAL_STEPS = 20
DEFAULT_EVALUATION_BATCHES = 32


@dataclass(frozen=True)
class SafeA100Plan:
    results_directory: Path
    budget_hours: float = DEFAULT_BUDGET_HOURS
    session_hours: float = DEFAULT_SESSION_HOURS
    data_seed: int = 20260916
    model_seed: int = 1337
    sequence_length: int = DEFAULT_SEQUENCE_LENGTH
    quotas: canonical.CorpusQuotas = field(
        default_factory=lambda: canonical.CorpusQuotas(
            opencode_priority=10_000,
            opencode_general=15_000,
            codefeedback=10_000,
            magicoder=8_000,
            openr1_math=2_000,
        )
    )
    opencode_scan_limit: int = 120_000
    source_scan_limit: int = 40_000
    shuffle_buffer_size: int = 4_096
    num_workers: int = DEFAULT_NUM_WORKERS
    checkpoint_interval_minutes: int = DEFAULT_CHECKPOINT_MINUTES
    log_interval_steps: int = DEFAULT_LOG_INTERVAL_STEPS
    evaluation_batches: int = DEFAULT_EVALUATION_BATCHES

    def __post_init__(self) -> None:
        if not isinstance(self.results_directory, Path):
            raise TypeError("results_directory must be a Path")
        for value, name in (
            (self.budget_hours, "budget_hours"),
            (self.session_hours, "session_hours"),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a number")
            if not math.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if self.session_hours > self.budget_hours:
            raise ValueError("session_hours cannot exceed budget_hours")
        if self.sequence_length < 2:
            raise ValueError("sequence_length must be at least 2")
        for value, name in (
            (self.opencode_scan_limit, "opencode_scan_limit"),
            (self.source_scan_limit, "source_scan_limit"),
            (self.shuffle_buffer_size, "shuffle_buffer_size"),
            (self.checkpoint_interval_minutes, "checkpoint_interval_minutes"),
            (self.log_interval_steps, "log_interval_steps"),
            (self.evaluation_batches, "evaluation_batches"),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if isinstance(self.num_workers, bool) or not isinstance(self.num_workers, int) or self.num_workers < 0:
            raise ValueError("num_workers must be a non-negative integer")
        for name, value in asdict(self.quotas).items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"quota {name} must be a positive integer")

    @property
    def requested_cost_usd(self) -> float:
        return self.session_hours * COST_PER_HOUR_USD

    @property
    def budget_cost_usd(self) -> float:
        return self.budget_hours * COST_PER_HOUR_USD

    @property
    def maximum_sessions(self) -> int:
        return math.floor(self.budget_hours / self.session_hours)

    @property
    def target_record_count(self) -> int:
        return sum(asdict(self.quotas).values())

    def to_dict(self) -> dict[str, Any]:
        return {
            "safe_run_format_version": SAFE_RUN_FORMAT_VERSION,
            "results_directory": str(self.results_directory),
            "budget_hours": self.budget_hours,
            "session_hours": self.session_hours,
            "requested_session_cost_usd": round(self.requested_cost_usd, 2),
            "budget_cost_usd": round(self.budget_cost_usd, 2),
            "maximum_sessions": self.maximum_sessions,
            "data_seed": self.data_seed,
            "model_seed": self.model_seed,
            "sequence_length": self.sequence_length,
            "quotas": self.quotas.to_dict(),
            "target_record_count": self.target_record_count,
            "opencode_scan_limit": self.opencode_scan_limit,
            "source_scan_limit": self.source_scan_limit,
            "shuffle_buffer_size": self.shuffle_buffer_size,
            "num_workers": self.num_workers,
            "checkpoint_interval_minutes": self.checkpoint_interval_minutes,
            "log_interval_steps": self.log_interval_steps,
            "evaluation_batches": self.evaluation_batches,
            "model_settings": canonical.model_settings().to_dict(),
        }


def build_run_configuration(plan: SafeA100Plan) -> canonical.RunConfiguration:
    return canonical.RunConfiguration(
        results_directory=plan.results_directory,
        session_seconds=round(plan.session_hours * 60 * 60),
        data_seed=plan.data_seed,
        model_seed=plan.model_seed,
        sequence_length=plan.sequence_length,
        quotas=plan.quotas,
        opencode_scan_limit=plan.opencode_scan_limit,
        source_scan_limit=plan.source_scan_limit,
        shuffle_buffer_size=plan.shuffle_buffer_size,
        num_workers=plan.num_workers,
        checkpoint_interval_seconds=plan.checkpoint_interval_minutes * 60,
        log_interval_steps=plan.log_interval_steps,
        evaluation_batches=plan.evaluation_batches,
    )


def _budget_path(plan: SafeA100Plan) -> Path:
    return plan.results_directory / "safe_budget_state.json"


def _load_budget_state(plan: SafeA100Plan) -> dict[str, Any]:
    path = _budget_path(plan)
    if not path.exists():
        return {
            "safe_run_format_version": SAFE_RUN_FORMAT_VERSION,
            "budget_hours": plan.budget_hours,
            "reserved_hours": 0.0,
            "consumed_hours": 0.0,
            "sessions": [],
        }
    state = canonical.read_json_object(path)
    if state.get("safe_run_format_version") != SAFE_RUN_FORMAT_VERSION:
        raise ValueError("safe budget state format is invalid")
    if state.get("budget_hours") != plan.budget_hours:
        raise ValueError("safe budget state belongs to a different budget")
    for name in ("reserved_hours", "consumed_hours"):
        value = state.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or value < 0:
            raise ValueError(f"safe budget state field {name} is invalid")
    if not isinstance(state.get("sessions"), list):
        raise ValueError("safe budget state sessions is invalid")
    return state


def reserve_budget(plan: SafeA100Plan) -> dict[str, Any]:
    """Reserve one session before remote work; a crash leaves the reservation conservative."""
    plan.results_directory.mkdir(parents=True, exist_ok=True)
    state = _load_budget_state(plan)
    available = plan.budget_hours - float(state["consumed_hours"]) - float(state["reserved_hours"])
    if plan.session_hours > available + 1e-9:
        raise RuntimeError(
            f"safe budget exhausted: requested={plan.session_hours:.2f}h available={max(available, 0.0):.2f}h"
        )
    state["reserved_hours"] = float(state["reserved_hours"]) + plan.session_hours
    canonical.atomic_write_json(_budget_path(plan), state)
    return state


def settle_budget(plan: SafeA100Plan, actual_hours: float, status: str) -> dict[str, Any]:
    if status not in {"completed", "failed"}:
        raise ValueError("budget settlement status is invalid")
    if isinstance(actual_hours, bool) or not isinstance(actual_hours, (int, float)):
        raise TypeError("actual_hours must be a number")
    actual_hours = max(0.0, float(actual_hours))
    state = _load_budget_state(plan)
    state["reserved_hours"] = max(0.0, float(state["reserved_hours"]) - plan.session_hours)
    state["consumed_hours"] = min(
        plan.budget_hours,
        float(state["consumed_hours"]) + actual_hours,
    )
    state["sessions"].append(
        {
            "status": status,
            "requested_hours": plan.session_hours,
            "actual_hours": actual_hours,
            "cost_usd": round(actual_hours * COST_PER_HOUR_USD, 2),
            "settled_at_unix": time.time(),
        }
    )
    canonical.atomic_write_json(_budget_path(plan), state)
    return state


def run_a100_preflight(plan: SafeA100Plan) -> dict[str, Any]:
    """Run one small real A100 forward/backward probe without opening datasets."""
    internal_report = canonical.run_internal_contract_tests()
    device, environment = canonical.configure_a100()
    torch.manual_seed(plan.model_seed)
    torch.cuda.manual_seed_all(plan.model_seed)
    model = KoemiModel(canonical.model_settings()).to(device)
    canonical_report = canonical.run_cuda_preflight(model, device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.01)
    input_ids = torch.randint(0, 256, (2, 32), device=device, dtype=torch.long)
    target_ids = torch.randint(0, 256, (2, 32), device=device, dtype=torch.long)
    thinking_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    torch.cuda.reset_peak_memory_stats(device)
    started_at = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output = model(input_ids, execution_mode=ExecutionMode.PARALLEL)
        objective = calculate_training_objective(output, target_ids, thinking_mask, 0.5)
    if not bool(torch.isfinite(objective.total_loss)):
        raise FloatingPointError("A100 preflight produced a non-finite loss")
    objective.total_loss.backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    if not bool(torch.isfinite(gradient_norm)):
        raise FloatingPointError("A100 preflight produced a non-finite gradient")
    optimizer.step()
    torch.cuda.synchronize(device)
    elapsed_seconds = time.perf_counter() - started_at
    report = {
        "plan": plan.to_dict(),
        "internal_contracts": internal_report,
        "environment": environment,
        "cuda_contract": canonical_report,
        "training_probe": {
            "batch_size": 2,
            "sequence_length": 32,
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "loss": float(objective.total_loss.detach().cpu()),
            "gradient_norm": float(gradient_norm.detach().cpu()),
            "elapsed_seconds": elapsed_seconds,
            "peak_memory_bytes": torch.cuda.max_memory_allocated(device),
        },
    }
    del optimizer, model, input_ids, target_ids, thinking_mask
    torch.cuda.empty_cache()
    return report


def run_safe_training(plan: SafeA100Plan) -> dict[str, Any]:
    reservation_started = time.perf_counter()
    reserve_budget(plan)
    configuration = build_run_configuration(plan)
    try:
        plan.results_directory.mkdir(parents=True, exist_ok=True)
        canonical.atomic_write_json(plan.results_directory / "safe_plan.json", plan.to_dict())
        internal_report = canonical.run_internal_contract_tests()
        device, environment = canonical.configure_a100()
        canonical.atomic_write_json(plan.results_directory / "environment.json", environment)
        canonical.atomic_write_json(
            plan.results_directory / "internal_contract_tests.json", internal_report
        )
        records, corpus_manifest = canonical.build_or_load_corpus(configuration)
        training_records, validation_records = canonical.split_records(
            records, configuration.data_seed
        )
        training_dataset = canonical.MaterializedCausalByteDataset(
            training_records, configuration.sequence_length
        )
        validation_dataset = canonical.MaterializedCausalByteDataset(
            validation_records, configuration.sequence_length
        )
        dataset_report = {
            "records": len(records),
            "training_records": len(training_records),
            "validation_records": len(validation_records),
            "training_chunks": len(training_dataset),
            "validation_chunks": len(validation_dataset),
            "sequence_length": configuration.sequence_length,
            "corpus_manifest": corpus_manifest,
        }
        canonical.atomic_write_json(plan.results_directory / "dataset_report.json", dataset_report)
        session_report = canonical.run_training(
            configuration,
            corpus_manifest,
            training_dataset,
            validation_dataset,
            device,
            environment,
        )
        final_report = {
            "safe_plan": plan.to_dict(),
            "environment": environment,
            "dataset": dataset_report,
            "session": session_report,
        }
        canonical.atomic_write_json(plan.results_directory / "final_report.json", final_report)
    except Exception:
        settle_budget(plan, (time.perf_counter() - reservation_started) / 3600.0, "failed")
        raise
    actual_hours = (time.perf_counter() - reservation_started) / 3600.0
    settle_budget(plan, actual_hours, "completed")
    return final_report


def parse_arguments(argv: list[str] | None = None) -> tuple[str, SafeA100Plan, float | None]:
    parser = argparse.ArgumentParser(description="Conservative Koemi A100 code-training launcher")
    parser.add_argument("--mode", choices=("plan", "preflight", "train"), default="plan")
    parser.add_argument("--results-dir", default="koemi-a100-safe-v1")
    parser.add_argument("--budget-hours", type=float, default=DEFAULT_BUDGET_HOURS)
    parser.add_argument("--session-hours", type=float, default=DEFAULT_SESSION_HOURS)
    parser.add_argument("--confirm-budget-hours", type=float, default=None)
    arguments = parser.parse_args(argv)
    plan = SafeA100Plan(
        results_directory=Path(arguments.results_dir).expanduser().resolve(),
        budget_hours=arguments.budget_hours,
        session_hours=arguments.session_hours,
    )
    return arguments.mode, plan, arguments.confirm_budget_hours


def main(argv: list[str] | None = None) -> None:
    mode, plan, confirmed_budget_hours = parse_arguments(argv)
    if mode == "plan":
        print(json.dumps(plan.to_dict(), indent=2, sort_keys=True))
        return
    if mode == "preflight":
        report = run_a100_preflight(plan)
        plan.results_directory.mkdir(parents=True, exist_ok=True)
        canonical.atomic_write_json(plan.results_directory / "preflight_report.json", report)
        print(json.dumps(report, indent=2, sort_keys=True))
        return
    if confirmed_budget_hours != plan.budget_hours:
        raise ValueError(
            "train requires --confirm-budget-hours equal to --budget-hours; this prevents accidental paid runs"
        )
    report = run_safe_training(plan)
    print(json.dumps(report, indent=2, sort_keys=True))


__all__ = [
    "COST_PER_HOUR_USD",
    "DEFAULT_BUDGET_HOURS",
    "DEFAULT_SESSION_HOURS",
    "SafeA100Plan",
    "build_run_configuration",
    "main",
    "parse_arguments",
    "reserve_budget",
    "run_a100_preflight",
    "run_safe_training",
    "settle_budget",
]


if __name__ == "__main__":
    main()
