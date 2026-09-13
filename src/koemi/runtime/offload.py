from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

import torch
from torch import Tensor, nn


ACCELERATOR_TIER = "accelerator"
HOST_TIER = "host"
DISK_TIER = "disk"
TIER_ORDER = (ACCELERATOR_TIER, HOST_TIER, DISK_TIER)
SAFE_STORE_NAME = re.compile(r"^[A-Za-z0-9_]+$")
ELEMENTWISE_FLOPS_PER_ELEMENT = 1
MATMUL_FLOPS_PER_ELEMENT = 2


class OffloadError(RuntimeError):
    pass


@dataclass(frozen=True)
class ModuleTraffic:
    name: str
    rows: int
    calls: int
    parameter_bytes: int
    flops: int

    @property
    def intensity(self) -> float:
        return self.flops / self.parameter_bytes if self.parameter_bytes else 0.0


@dataclass(frozen=True)
class TierBudget:
    accelerator_bytes: int | None = None
    host_bytes: int | None = None

    def __post_init__(self) -> None:
        for value in (self.accelerator_bytes, self.host_bytes):
            if value is not None and value < 0:
                raise ValueError("tier budgets must be non-negative or unlimited")


@dataclass(frozen=True)
class ModulePlacement:
    name: str
    tier: str
    parameter_bytes: int
    intensity: float

    def __post_init__(self) -> None:
        if self.tier not in TIER_ORDER:
            raise ValueError(f"tier must be one of {TIER_ORDER}")


@dataclass(frozen=True)
class PlacementPlan:
    placements: tuple[ModulePlacement, ...]

    def tier_of(self, name: str) -> str:
        for placement in self.placements:
            if placement.name == name:
                return placement.tier
        raise KeyError(name)

    def bytes_by_tier(self) -> dict[str, int]:
        totals = {tier: 0 for tier in TIER_ORDER}
        for placement in self.placements:
            totals[placement.tier] += placement.parameter_bytes
        return totals

    def names_by_tier(self, tier: str) -> tuple[str, ...]:
        return tuple(placement.name for placement in self.placements if placement.tier == tier)

    def to_dict(self) -> dict[str, object]:
        return {
            "bytes_by_tier": self.bytes_by_tier(),
            "placements": [
                {
                    "name": placement.name,
                    "tier": placement.tier,
                    "parameter_bytes": placement.parameter_bytes,
                    "intensity": placement.intensity,
                }
                for placement in self.placements
            ],
        }


@dataclass
class OffloadStatistics:
    bytes_by_tier: dict[str, int] = field(default_factory=dict)
    host_borrows: int = 0
    host_transferred_bytes: int = 0
    disk_materializations: int = 0
    disk_read_bytes: int = 0
    disk_read_seconds: float = 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "bytes_by_tier": dict(self.bytes_by_tier),
            "host_borrows": self.host_borrows,
            "host_transferred_bytes": self.host_transferred_bytes,
            "disk_materializations": self.disk_materializations,
            "disk_read_bytes": self.disk_read_bytes,
            "disk_read_seconds": self.disk_read_seconds,
        }


def parameter_bytes_of(module: nn.Module) -> int:
    return sum(
        parameter.numel() * parameter.element_size()
        for parameter in module.parameters(recurse=False)
    )


def module_flops(module: nn.Module, rows: int) -> int:
    if isinstance(module, nn.Embedding):
        return 0
    if isinstance(module, nn.Linear):
        return MATMUL_FLOPS_PER_ELEMENT * rows * module.weight.numel()
    owned = sum(parameter.numel() for parameter in module.parameters(recurse=False))
    return ELEMENTWISE_FLOPS_PER_ELEMENT * rows * owned


def rows_of(module: nn.Module, inputs: tuple[object, ...]) -> int:
    for value in inputs:
        if isinstance(value, Tensor) and value.ndim >= 1:
            width = value.shape[-1]
            return value.numel() // width if width else 0
    return 0


def owning_modules(model: nn.Module) -> Iterable[tuple[str, nn.Module]]:
    for name, module in model.named_modules():
        if name and any(True for _ in module.parameters(recurse=False)):
            yield name, module


def calibrate_traffic(model: nn.Module, run_forward: Callable[[], object]) -> tuple[ModuleTraffic, ...]:
    observed: dict[str, list[int]] = {}
    handles = []

    def make_hook(name: str):
        def hook(module: nn.Module, inputs: tuple[object, ...]) -> None:
            observed.setdefault(name, [0, 0])
            observed[name][0] += rows_of(module, inputs)
            observed[name][1] += 1

        return hook

    for name, module in owning_modules(model):
        handles.append(module.register_forward_pre_hook(make_hook(name)))
    try:
        run_forward()
    finally:
        for handle in handles:
            handle.remove()
    traffic = []
    for name, module in owning_modules(model):
        rows, calls = observed.get(name, [0, 0])
        traffic.append(
            ModuleTraffic(
                name=name,
                rows=rows,
                calls=calls,
                parameter_bytes=parameter_bytes_of(module),
                flops=module_flops(module, rows),
            )
        )
    return tuple(traffic)


def plan_placement(traffic: Iterable[ModuleTraffic], budget: TierBudget) -> PlacementPlan:
    ranked = sorted(traffic, key=lambda item: (-item.intensity, -item.parameter_bytes, item.name))
    remaining = {
        ACCELERATOR_TIER: budget.accelerator_bytes,
        HOST_TIER: budget.host_bytes,
    }
    placements = []
    for item in ranked:
        placements.append(
            ModulePlacement(
                name=item.name,
                tier=select_tier(item.parameter_bytes, remaining),
                parameter_bytes=item.parameter_bytes,
                intensity=item.intensity,
            )
        )
    return PlacementPlan(tuple(placements))


def select_tier(parameter_bytes: int, remaining: dict[str, int | None]) -> str:
    for tier in (ACCELERATOR_TIER, HOST_TIER):
        allowance = remaining[tier]
        if allowance is None:
            return tier
        if parameter_bytes <= allowance:
            remaining[tier] = allowance - parameter_bytes
            return tier
    return DISK_TIER


@dataclass(frozen=True)
class OffloadRequest:
    accelerator_bytes: int | None = None
    host_bytes: int | None = None
    store_directory: str | None = None

    @property
    def requested(self) -> bool:
        return (
            self.accelerator_bytes is not None
            or self.host_bytes is not None
            or self.store_directory is not None
        )

    def budget(self) -> TierBudget:
        return TierBudget(accelerator_bytes=self.accelerator_bytes, host_bytes=self.host_bytes)


class DiskParameterStore:
    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory).expanduser().resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.read_bytes = 0
        self.read_seconds = 0.0
        self.reads = 0

    def path_for(self, name: str) -> Path:
        safe_name = name.replace(".", "__")
        if not SAFE_STORE_NAME.match(safe_name):
            raise OffloadError(f"parameter name is not safe for a store file: {name}")
        target = (self.directory / f"{safe_name}.pt").resolve()
        if target.parent != self.directory:
            raise OffloadError(f"parameter name escapes the store directory: {name}")
        return target

    def write(self, name: str, tensor: Tensor) -> None:
        torch.save(tensor.detach().to("cpu").clone(), self.path_for(name))

    def read(self, name: str, device: torch.device) -> Tensor:
        start = time.perf_counter()
        tensor = self.restore(name, device)
        self.read_seconds += time.perf_counter() - start
        self.read_bytes += tensor.numel() * tensor.element_size()
        self.reads += 1
        return tensor

    def restore(self, name: str, device: torch.device) -> Tensor:
        target = self.path_for(name)
        if not target.is_file():
            raise OffloadError(f"store entry is missing: {name}")
        return torch.load(target, map_location=device, weights_only=True)

    def clear(self) -> int:
        removed = 0
        for entry in self.directory.glob("*.pt"):
            entry.unlink()
            removed += 1
        return removed


class OffloadEngine:
    def __init__(
        self,
        model: nn.Module,
        plan: PlacementPlan,
        compute_device: str | torch.device,
        store: DiskParameterStore | None = None,
    ) -> None:
        self.model = model
        self.plan = plan
        self.compute_device = torch.device(compute_device)
        self.store = store
        self.statistics = OffloadStatistics(bytes_by_tier=plan.bytes_by_tier())
        self.handles: list[torch.utils.hooks.RemovableHandle] = []
        self.attached = False
        self.evicted_shapes: dict[str, tuple[int, ...]] = {}
        self.borrowed: dict[str, list[tuple[nn.Module, str, nn.Parameter]]] = {}

    def attach(self) -> None:
        if self.attached:
            raise OffloadError("the offload engine is already attached")
        modules = dict(owning_modules(self.model))
        disk_names = self.plan.names_by_tier(DISK_TIER)
        if disk_names and self.store is None:
            raise OffloadError("a disk tier placement requires a parameter store")
        for placement in self.plan.placements:
            module = modules.get(placement.name)
            if module is None:
                raise OffloadError(f"the plan names a module the model does not own: {placement.name}")
            if placement.tier == ACCELERATOR_TIER:
                self.move_parameters(module, self.compute_device)
                continue
            if placement.tier == HOST_TIER:
                self.move_parameters(module, torch.device("cpu"))
                self.register_borrow_hooks(placement.name, module, HOST_TIER)
                continue
            self.evict_to_disk(placement.name, module)
            self.register_borrow_hooks(placement.name, module, DISK_TIER)
        self.attached = True

    def detach(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.attached = False
        modules = dict(owning_modules(self.model))
        for placement in self.plan.placements:
            if placement.tier != DISK_TIER:
                continue
            module = modules[placement.name]
            for parameter_name, parameter in module.named_parameters(recurse=False):
                stored = f"{placement.name}.{parameter_name}"
                parameter.data = self.store.restore(stored, self.compute_device)
                self.evicted_shapes.pop(stored, None)

    @staticmethod
    def move_parameters(module: nn.Module, device: torch.device) -> None:
        for parameter in module.parameters(recurse=False):
            parameter.data = parameter.data.to(device)

    def evict_to_disk(self, name: str, module: nn.Module) -> None:
        for parameter_name, parameter in module.named_parameters(recurse=False):
            if parameter.requires_grad:
                raise OffloadError(
                    f"{name}.{parameter_name} is trainable, so it cannot live on the disk tier; "
                    "freeze it or give it a host budget"
                )
            stored = f"{name}.{parameter_name}"
            self.store.write(stored, parameter.data)
            self.evicted_shapes[stored] = tuple(parameter.shape)
            parameter.data = torch.empty(0, dtype=parameter.dtype, device=parameter.device)

    def register_borrow_hooks(self, name: str, module: nn.Module, tier: str) -> None:
        def borrow(target: nn.Module, inputs: tuple[object, ...]) -> None:
            self.borrow_parameters(name, target, tier)

        def give_back(target: nn.Module, inputs: tuple[object, ...], output: object) -> None:
            self.restore_parameters(name, target)

        self.handles.append(module.register_forward_pre_hook(borrow))
        self.handles.append(module.register_forward_hook(give_back))

    def borrow_parameters(self, name: str, module: nn.Module, tier: str) -> None:
        if name in self.borrowed:
            return
        held = []
        for parameter_name, parameter in list(module.named_parameters(recurse=False)):
            stored = f"{name}.{parameter_name}"
            if tier == DISK_TIER:
                materialized = self.store.read(stored, self.compute_device)
                self.statistics.disk_materializations += 1
            else:
                materialized = self.lend_to_device(parameter)
                self.statistics.host_borrows += 1
            held.append((module, parameter_name, parameter))
            del module._parameters[parameter_name]
            setattr(module, parameter_name, materialized)
        self.borrowed[name] = held

    def lend_to_device(self, parameter: nn.Parameter) -> Tensor:
        moved = parameter.to(self.compute_device)
        if moved is parameter:
            return parameter.view_as(parameter)
        self.statistics.host_transferred_bytes += parameter.numel() * parameter.element_size()
        return moved

    def restore_parameters(self, name: str, module: nn.Module) -> None:
        held = self.borrowed.pop(name, None)
        if held is None:
            return
        for owner, parameter_name, parameter in held:
            delattr(owner, parameter_name)
            owner.register_parameter(parameter_name, parameter)

    def refresh_statistics(self) -> OffloadStatistics:
        if self.store is not None:
            self.statistics.disk_read_bytes = self.store.read_bytes
            self.statistics.disk_read_seconds = self.store.read_seconds
        return self.statistics


def prepare_offload(
    model: nn.Module,
    run_forward: Callable[[], object],
    request: OffloadRequest,
    compute_device: str | torch.device,
) -> OffloadEngine:
    traffic = calibrate_traffic(model, run_forward)
    plan = plan_placement(traffic, request.budget())
    store = DiskParameterStore(request.store_directory) if request.store_directory else None
    if plan.names_by_tier(DISK_TIER) and store is None:
        raise OffloadError(
            "the budget pushed modules to the disk tier, so --offload-store is required"
        )
    engine = OffloadEngine(model, plan, compute_device, store)
    engine.attach()
    return engine
