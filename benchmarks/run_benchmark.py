from __future__ import annotations

import argparse
import json
import logging
import random
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional
from torch.utils.data import DataLoader

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from koemi.configuration.settings import ModelSettings, PAD_TOKEN_ID, TrainingSettings
from koemi.data.contracts import DatasetRecord
from koemi.model.network import KoemiModel
from koemi.training.dataset import CausalByteDataset, IGNORE_TARGET_ID, create_training_loader
from koemi.training.objective import token_cross_entropy
from koemi.training.trainer import Trainer

MODEL_NAMES = ("koemi", "gru", "lstm")
TASK_NAMES = ("bytes", "recall")
NATS_PER_BIT = 0.6931471805599453
VOCABULARY_SIZE = PAD_TOKEN_ID + 1

SUBJECTS = ("the queue", "the stack", "the buffer", "the cache", "the parser")
VERBS = ("removes", "stores", "returns", "rejects", "accepts", "replaces")
OBJECTS = ("the oldest item", "the newest item", "an invalid record", "a padded token", "the first byte")
REASONS = ("because arrival order decides", "because the window is small", "because the contract requires it")


@dataclass(frozen=True)
class BenchmarkReport:
    model_name: str
    task_name: str
    parameter_count: int
    parameter_bytes: int
    state_bytes_per_sequence: int
    train_seconds: float
    train_tokens_per_second: float
    train_loss_nats: float
    evaluation_loss_nats: float
    evaluation_supervised_tokens: int
    evaluation_loss_standard_error_nats: float
    bits_per_byte: float
    bits_per_byte_standard_error: float
    evaluation_tokens_per_second: float
    peak_resident_bytes: int | None
    expert_activation_counts: tuple[int, ...] | None


class RecurrentBaseline(nn.Module):
    def __init__(self, cell_name: str, embedding_size: int, hidden_size: int) -> None:
        super().__init__()
        self.cell_name = cell_name
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(VOCABULARY_SIZE, embedding_size, padding_idx=PAD_TOKEN_ID)
        cell_type = nn.GRU if cell_name == "gru" else nn.LSTM
        self.recurrent_cell = cell_type(embedding_size, hidden_size, batch_first=True)
        self.token_predictor = nn.Linear(hidden_size, VOCABULARY_SIZE)

    def forward(self, input_ids: Tensor) -> Tensor:
        hidden_states, _ = self.recurrent_cell(self.embedding(input_ids))
        return self.token_predictor(hidden_states)

    def state_bytes_per_sequence(self) -> int:
        state_tensors = 2 if self.cell_name == "lstm" else 1
        return state_tensors * self.hidden_size * 4


def peak_resident_bytes() -> int | None:
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        current_process = ctypes.windll.kernel32.GetCurrentProcess
        current_process.restype = wintypes.HANDLE
        query_memory = ctypes.windll.psapi.GetProcessMemoryInfo
        query_memory.argtypes = [wintypes.HANDLE, ctypes.POINTER(ProcessMemoryCounters), wintypes.DWORD]
        query_memory.restype = wintypes.BOOL
        if not query_memory(current_process(), ctypes.byref(counters), counters.cb):
            return None
        return int(counters.PeakWorkingSetSize)
    import resource

    maximum_resident = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(maximum_resident) if sys.platform == "darwin" else int(maximum_resident) * 1024


def build_byte_records(generator: random.Random, record_count: int, prefix: str) -> tuple[DatasetRecord, ...]:
    records = []
    for index in range(record_count):
        subject = generator.choice(SUBJECTS)
        verb = generator.choice(VERBS)
        target = generator.choice(OBJECTS)
        reason = generator.choice(REASONS)
        question = f"What does {subject} do with {target}?"
        answer = f"{subject} {verb} {target} {reason}."
        records.append(DatasetRecord(f"{prefix}-{index}", question, None, answer, {}))
    return tuple(records)


def build_recall_records(generator: random.Random, record_count: int, pair_count: int, prefix: str) -> tuple[DatasetRecord, ...]:
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    records = []
    for index in range(record_count):
        keys = generator.sample(alphabet, pair_count)
        values = ["".join(generator.choice(alphabet) for _ in range(3)) for _ in keys]
        pairs = " ".join(f"{key}={value}" for key, value in zip(keys, values, strict=True))
        query_position = generator.randrange(pair_count)
        question = f"{pairs} ? {keys[query_position]}="
        records.append(DatasetRecord(f"{prefix}-{index}", question, None, values[query_position], {}))
    return tuple(records)


def build_records(task_name: str, generator: random.Random, record_count: int, prefix: str) -> tuple[DatasetRecord, ...]:
    if task_name == "recall":
        return build_recall_records(generator, record_count, pair_count=6, prefix=prefix)
    return build_byte_records(generator, record_count, prefix=prefix)


def count_parameter_bytes(model: nn.Module) -> tuple[int, int]:
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    parameter_bytes = sum(parameter.numel() * parameter.element_size() for parameter in model.parameters())
    return parameter_count, parameter_bytes


def match_hidden_size(cell_name: str, embedding_size: int, target_parameter_count: int) -> int:
    best_hidden_size = 8
    best_distance = None
    for hidden_size in range(8, 513, 4):
        candidate = RecurrentBaseline(cell_name, embedding_size, hidden_size)
        parameter_count, _ = count_parameter_bytes(candidate)
        distance = abs(parameter_count - target_parameter_count)
        if best_distance is None or distance < best_distance:
            best_distance = distance
            best_hidden_size = hidden_size
    return best_hidden_size


def koemi_state_bytes(settings: ModelSettings) -> int:
    embedding_size = settings.embedding_size
    associative_scalars = embedding_size * settings.memory_features + settings.memory_features
    state_scalars = embedding_size + 2 * associative_scalars
    local_bytes = 2 * settings.local_memory_size * embedding_size * 4
    salient_bytes = 2 * settings.salience_memory_size * embedding_size * 4
    valid_bytes = settings.local_memory_size + settings.salience_memory_size
    last_token_bytes = 8
    return state_scalars * 4 + local_bytes + salient_bytes + valid_bytes + last_token_bytes


def evaluation_error(values: list[float]) -> float:
    if len(values) < 2:
        return float("nan")
    values_tensor = torch.tensor(values, dtype=torch.float64)
    return float(values_tensor.std(unbiased=True) / len(values) ** 0.5)


def evaluate_baseline(model: RecurrentBaseline, loader: DataLoader) -> tuple[float, float, int, float]:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    token_losses: list[float] = []
    start_time = time.perf_counter()
    with torch.no_grad():
        for batch in loader:
            target_ids = batch["target_ids"]
            supervised_mask = target_ids != IGNORE_TARGET_ID
            supervised_count = int(supervised_mask.sum())
            if supervised_count == 0:
                continue
            logits = model(batch["input_ids"])
            token_loss = token_cross_entropy(logits, target_ids)
            total_loss += float((token_loss * supervised_mask).sum())
            token_losses.extend(token_loss.masked_select(supervised_mask).tolist())
            total_tokens += supervised_count
    elapsed_seconds = time.perf_counter() - start_time
    return total_loss / total_tokens, elapsed_seconds, total_tokens, evaluation_error(token_losses)


def evaluate_koemi(model: KoemiModel, loader: DataLoader) -> tuple[float, float, int, float, tuple[int, ...]]:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    token_losses: list[float] = []
    activation_counts: list[int] = []
    start_time = time.perf_counter()
    with torch.no_grad():
        for batch in loader:
            target_ids = batch["target_ids"]
            supervised_mask = target_ids != IGNORE_TARGET_ID
            supervised_count = int(supervised_mask.sum())
            if supervised_count == 0:
                continue
            output = model(batch["input_ids"])
            token_loss = token_cross_entropy(output.logits, target_ids)
            total_loss += float((token_loss * supervised_mask).sum())
            token_losses.extend(token_loss.masked_select(supervised_mask).tolist())
            total_tokens += supervised_count
            counts = output.expert_activation_counts
            if len(activation_counts) < len(counts):
                activation_counts.extend([0] * (len(counts) - len(activation_counts)))
            for index, count in enumerate(counts):
                activation_counts[index] += count
    elapsed_seconds = time.perf_counter() - start_time
    return total_loss / total_tokens, elapsed_seconds, total_tokens, evaluation_error(token_losses), tuple(activation_counts)


def train_baseline(model: RecurrentBaseline, loader: DataLoader, arguments: argparse.Namespace) -> tuple[float, float, int]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=arguments.learning_rate)
    model.train()
    weighted_loss = 0.0
    supervised_total = 0
    start_time = time.perf_counter()
    for _ in range(arguments.epochs):
        for batch in loader:
            target_ids = batch["target_ids"]
            supervised_mask = target_ids != IGNORE_TARGET_ID
            supervised_count = int(supervised_mask.sum())
            if supervised_count == 0:
                continue
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch["input_ids"])
            loss = functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                target_ids.reshape(-1),
                ignore_index=IGNORE_TARGET_ID,
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            weighted_loss += float(loss.detach()) * supervised_count
            supervised_total += supervised_count
    elapsed_seconds = time.perf_counter() - start_time
    return weighted_loss / supervised_total, elapsed_seconds, supervised_total


def run_single_model(arguments: argparse.Namespace) -> BenchmarkReport:
    torch.manual_seed(arguments.seed)
    generator = random.Random(arguments.seed)
    train_records = build_records(arguments.task, generator, arguments.train_records, "train")
    evaluation_records = build_records(arguments.task, generator, arguments.evaluation_records, "eval")
    train_dataset = CausalByteDataset(train_records, arguments.sequence_length)
    evaluation_dataset = CausalByteDataset(evaluation_records, arguments.sequence_length)
    train_loader = create_training_loader(train_dataset, arguments.batch_size, torch.Generator().manual_seed(arguments.seed))
    evaluation_loader = create_training_loader(
        evaluation_dataset, arguments.batch_size, torch.Generator().manual_seed(arguments.seed)
    )
    model_settings = ModelSettings(
        embedding_size=arguments.embedding_size,
        memory_features=arguments.memory_features,
        local_memory_size=arguments.local_memory_size,
        salience_memory_size=arguments.salience_memory_size,
        salience_threshold=arguments.salience_threshold,
        expert_count=arguments.expert_count,
        expert_top_k=arguments.expert_top_k,
        ablation=arguments.ablation,
    )
    koemi_parameter_count, koemi_parameter_bytes = count_parameter_bytes(KoemiModel(model_settings))
    torch.manual_seed(arguments.seed)
    if arguments.model == "koemi":
        model = KoemiModel(model_settings)
        training_settings = TrainingSettings(
            sequence_length=arguments.sequence_length,
            batch_size=arguments.batch_size,
            epochs=arguments.epochs,
            learning_rate=arguments.learning_rate,
            device="cpu",
        )
        logger = logging.getLogger("koemi-benchmark")
        logger.addHandler(logging.NullHandler())
        result = Trainer(logger).train(model, train_loader, training_settings)
        loss, evaluation_seconds, evaluation_tokens, loss_standard_error, activations = evaluate_koemi(model, evaluation_loader)
        return BenchmarkReport(
            model_name="koemi",
            task_name=arguments.task,
            parameter_count=koemi_parameter_count,
            parameter_bytes=koemi_parameter_bytes,
            state_bytes_per_sequence=koemi_state_bytes(model_settings),
            train_seconds=result.elapsed_seconds,
            train_tokens_per_second=result.supervised_token_count / result.elapsed_seconds,
            train_loss_nats=result.mean_loss,
            evaluation_loss_nats=loss,
            evaluation_supervised_tokens=evaluation_tokens,
            evaluation_loss_standard_error_nats=loss_standard_error,
            bits_per_byte=loss / NATS_PER_BIT,
            bits_per_byte_standard_error=loss_standard_error / NATS_PER_BIT,
            evaluation_tokens_per_second=evaluation_tokens / evaluation_seconds,
            peak_resident_bytes=peak_resident_bytes(),
            expert_activation_counts=activations,
        )
    hidden_size = match_hidden_size(arguments.model, arguments.embedding_size, koemi_parameter_count)
    torch.manual_seed(arguments.seed)
    baseline = RecurrentBaseline(arguments.model, arguments.embedding_size, hidden_size)
    parameter_count, parameter_bytes = count_parameter_bytes(baseline)
    train_loss, train_seconds, train_tokens = train_baseline(baseline, train_loader, arguments)
    loss, evaluation_seconds, evaluation_tokens, loss_standard_error = evaluate_baseline(baseline, evaluation_loader)
    return BenchmarkReport(
        model_name=arguments.model,
        task_name=arguments.task,
        parameter_count=parameter_count,
        parameter_bytes=parameter_bytes,
        state_bytes_per_sequence=baseline.state_bytes_per_sequence(),
        train_seconds=train_seconds,
        train_tokens_per_second=train_tokens / train_seconds,
        train_loss_nats=train_loss,
        evaluation_loss_nats=loss,
        evaluation_supervised_tokens=evaluation_tokens,
        evaluation_loss_standard_error_nats=loss_standard_error,
        bits_per_byte=loss / NATS_PER_BIT,
        bits_per_byte_standard_error=loss_standard_error / NATS_PER_BIT,
        evaluation_tokens_per_second=evaluation_tokens / evaluation_seconds,
        peak_resident_bytes=peak_resident_bytes(),
        expert_activation_counts=None,
    )


def run_every_model(arguments: argparse.Namespace) -> list[dict]:
    reports = []
    for model_name in MODEL_NAMES:
        command = [sys.executable, str(Path(__file__).resolve()), "--model", model_name]
        for key, value in vars(arguments).items():
            if key in {"model", "report"}:
                continue
            command.extend([f"--{key.replace('_', '-')}", str(value)])
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        if completed.returncode != 0:
            raise RuntimeError(f"benchmark for {model_name} failed: {completed.stderr.strip()}")
        reports.append(json.loads(completed.stdout))
    return reports


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Koemi-3HIP benchmark against GRU and LSTM baselines")
    parser.add_argument("--model", choices=MODEL_NAMES, default=None)
    parser.add_argument("--task", choices=TASK_NAMES, default="bytes")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--train-records", type=int, default=48)
    parser.add_argument("--evaluation-records", type=int, default=16)
    parser.add_argument("--sequence-length", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--embedding-size", type=int, default=48)
    parser.add_argument("--memory-features", type=int, default=12)
    parser.add_argument("--local-memory-size", type=int, default=12)
    parser.add_argument("--salience-memory-size", type=int, default=12)
    parser.add_argument("--salience-threshold", type=float, default=0.75)
    parser.add_argument("--expert-count", type=int, default=0)
    parser.add_argument("--expert-top-k", type=int, default=1)
    parser.add_argument("--ablation", choices=("herm", "no_refine", "no_surprise", "affine"), default="no_refine")
    parser.add_argument("--report", default=None)
    return parser


def main(argument_values: list[str] | None = None) -> int:
    arguments = create_parser().parse_args(argument_values)
    if arguments.model is not None:
        print(json.dumps(asdict(run_single_model(arguments)), indent=2, sort_keys=True))
        return 0
    payload = {
        "architecture": "Koemi-3HIP",
        "task": arguments.task,
        "seed": arguments.seed,
        "platform": sys.platform,
        "torch_version": torch.__version__,
        "reports": run_every_model(arguments),
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    if arguments.report:
        Path(arguments.report).write_text(rendered, encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
