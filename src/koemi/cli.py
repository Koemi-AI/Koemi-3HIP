from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import torch

from koemi.configuration.settings import ModelSettings, TrainingSettings
from koemi.data.adapters import SUPPORTED_DATASET_FORMATS
from koemi.data.readers import DatasetLoadReport, load_dataset_records, split_dataset_records
from koemi.data.tokenizer import ByteTokenizer
from koemi.model.cache import DiskMappingCache, WarmTokenCache
from koemi.model.network import KoemiModel
from koemi.observability.logging import configure_logging
from koemi.runtime.offload import (
    ACCELERATOR_TIER,
    DISK_TIER,
    HOST_TIER,
    OffloadEngine,
    OffloadRequest,
    prepare_offload,
)
from koemi.training.checkpoints import CheckpointStore
from koemi.training.dataset import CausalByteDataset, create_training_loader
from koemi.training.generation import generate_text
from koemi.training.trainer import Trainer


def main(arguments: Sequence[str] | None = None) -> int:
    parser = create_parser()
    parsed_arguments = parser.parse_args(arguments)
    logger = configure_logging(parsed_arguments.verbose)
    try:
        if parsed_arguments.command == "inspect-dataset":
            return inspect_dataset(parsed_arguments, logger)
        if parsed_arguments.command == "train":
            return train_model(parsed_arguments, logger)
        if parsed_arguments.command == "generate":
            return generate_completion(parsed_arguments, logger)
        parser.error(f"unsupported command: {parsed_arguments.command}")
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        logger.error("command_failed error=%s", error)
        return 2
    return 2


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="koemi", description="Koemi-2OBOV byte-level recurrent training base")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect-dataset", help="Validate and summarize JSON datasets")
    add_dataset_arguments(inspect_parser)

    train_parser = subparsers.add_parser("train", help="Train a Koemi-2OBOV checkpoint from JSON datasets")
    add_dataset_arguments(train_parser)
    train_parser.add_argument("--checkpoint", required=True, help="Output checkpoint path")
    train_parser.add_argument("--overwrite", action="store_true", help="Replace an existing checkpoint")
    train_parser.add_argument("--sequence-length", type=int, default=128)
    train_parser.add_argument("--batch-size", type=int, default=4)
    train_parser.add_argument("--epochs", type=int, default=3)
    train_parser.add_argument("--learning-rate", type=float, default=0.001)
    train_parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    train_parser.add_argument("--device", default=None, help="Training device, defaulting to CUDA when available")
    train_parser.add_argument("--execution-mode", choices=("parallel", "sequential"), default="parallel")
    train_parser.add_argument("--thinking-loss-weight", type=float, default=1.0)
    train_parser.add_argument("--weight-decay", type=float, default=0.01)
    train_parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    train_parser.add_argument("--warmup-steps", type=int, default=0)
    train_parser.add_argument("--precision", choices=("auto", "fp32", "bf16", "fp16"), default="auto")
    train_parser.add_argument("--label-smoothing", type=float, default=0.0)
    train_parser.add_argument("--validation-fraction", type=float, default=0.0)
    train_parser.add_argument("--seed", type=int, default=17)
    train_parser.add_argument("--num-workers", type=int, default=0)
    train_parser.add_argument("--prefetch-factor", type=int, default=2)
    train_parser.add_argument("--no-pin-memory", action="store_true")
    add_offload_arguments(train_parser)
    add_model_arguments(train_parser)

    generate_parser = subparsers.add_parser("generate", help="Generate text from a Koemi-2OBOV checkpoint")
    generate_parser.add_argument("--checkpoint", required=True, help="Checkpoint path")
    generate_parser.add_argument("--prompt", required=True, help="Text used to start generation")
    generate_parser.add_argument("--max-new-bytes", type=int, default=128)
    generate_parser.add_argument("--temperature", type=float, default=1.0)
    generate_parser.add_argument("--device", default="cpu")
    generate_parser.add_argument("--cache-capacity", type=int, default=None)
    generate_parser.add_argument("--mapping-cache", default=None, help="Optional SSD directory for exact inference mappings")
    generate_parser.add_argument("--mapping-cache-capacity", type=int, default=128)
    generate_parser.add_argument("--mapping-cache-max-entry-mib", type=int, default=64)
    generate_parser.add_argument("--mapping-cache-namespace", default=None)
    generate_parser.add_argument("--mapping-cache-ttl-seconds", type=float, default=3600.0)
    generate_parser.add_argument("--clear-mapping-cache", action="store_true")
    add_offload_arguments(generate_parser)
    return parser


def add_dataset_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset", action="append", required=True, help="JSON, JSONL or TXT dataset path")
    parser.add_argument("--dataset-format", choices=SUPPORTED_DATASET_FORMATS, default="auto")


def add_offload_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--offload-accelerator-mib",
        type=int,
        default=None,
        help="Parameter budget kept on the compute device, in MiB",
    )
    parser.add_argument(
        "--offload-host-mib",
        type=int,
        default=None,
        help="Parameter budget streamed from host memory, in MiB",
    )
    parser.add_argument(
        "--offload-store",
        default=None,
        help="Directory holding parameters evicted to storage",
    )


def offload_request(arguments: argparse.Namespace) -> OffloadRequest:
    return OffloadRequest(
        accelerator_bytes=mebibytes_to_bytes(arguments.offload_accelerator_mib),
        host_bytes=mebibytes_to_bytes(arguments.offload_host_mib),
        store_directory=arguments.offload_store,
    )


def mebibytes_to_bytes(value: int | None) -> int | None:
    if value is None:
        return None
    if value < 0:
        raise ValueError("an offload budget must be non-negative")
    return value * 1024 * 1024


def log_offload(engine: OffloadEngine, logger) -> None:
    statistics = engine.refresh_statistics()
    logger.info(
        "offload_plan accelerator_bytes=%s host_bytes=%s disk_bytes=%s "
        "accelerator_modules=%s host_modules=%s disk_modules=%s "
        "host_materializations=%s host_transferred_bytes=%s "
        "disk_materializations=%s disk_read_bytes=%s disk_read_seconds=%.4f",
        statistics.bytes_by_tier[ACCELERATOR_TIER],
        statistics.bytes_by_tier[HOST_TIER],
        statistics.bytes_by_tier[DISK_TIER],
        len(engine.plan.names_by_tier(ACCELERATOR_TIER)),
        len(engine.plan.names_by_tier(HOST_TIER)),
        len(engine.plan.names_by_tier(DISK_TIER)),
        statistics.host_materializations,
        statistics.host_transferred_bytes,
        statistics.disk_materializations,
        statistics.disk_read_bytes,
        statistics.disk_read_seconds,
    )


def add_model_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--embedding-size", type=int, default=64)
    parser.add_argument("--memory-features", type=int, default=16)
    parser.add_argument("--local-memory-size", type=int, default=16)
    parser.add_argument("--expert-count", type=int, default=0)
    parser.add_argument("--cache-capacity", type=int, default=256)
    parser.add_argument("--scan-chunk", type=int, default=128)
    parser.add_argument("--refine-decay-rate", type=float, default=0.0625)
    parser.add_argument("--ablation", choices=("herm", "no_refine", "no_surprise", "affine"), default="herm")


def inspect_dataset(arguments: argparse.Namespace, logger) -> int:
    report = load_and_log_dataset(arguments.dataset, arguments.dataset_format, logger)
    report_payload = {
        "adapter_counts": report.adapter_counts,
        "record_count": report.record_count,
        "source_files": [str(source_file) for source_file in report.source_files],
    }
    write_utf8(json.dumps(report_payload, indent=2, sort_keys=True))
    return 0


def train_model(arguments: argparse.Namespace, logger) -> int:
    report = load_and_log_dataset(arguments.dataset, arguments.dataset_format, logger)
    model_settings = create_model_settings(arguments)
    training_settings = TrainingSettings(
        sequence_length=arguments.sequence_length,
        batch_size=arguments.batch_size,
        epochs=arguments.epochs,
        learning_rate=arguments.learning_rate,
        gradient_clip_norm=arguments.gradient_clip_norm,
        device=arguments.device or ("cuda" if torch.cuda.is_available() else "cpu"),
        execution_mode=arguments.execution_mode,
        thinking_loss_weight=arguments.thinking_loss_weight,
        weight_decay=arguments.weight_decay,
        gradient_accumulation_steps=arguments.gradient_accumulation_steps,
        warmup_steps=arguments.warmup_steps,
        precision=arguments.precision,
        label_smoothing=arguments.label_smoothing,
        num_workers=arguments.num_workers,
        pin_memory=not arguments.no_pin_memory,
        prefetch_factor=arguments.prefetch_factor,
    )
    effective_pin_memory = training_settings.pin_memory and training_settings.device.startswith("cuda")
    training_records, validation_records = split_dataset_records(
        report.records, arguments.validation_fraction, arguments.seed
    )
    dataset = CausalByteDataset(training_records, training_settings.sequence_length)
    loader = create_training_loader(
        dataset,
        training_settings.batch_size,
        torch.Generator().manual_seed(arguments.seed),
        num_workers=training_settings.num_workers,
        pin_memory=effective_pin_memory,
        prefetch_factor=training_settings.prefetch_factor,
    )
    validation_loader = None
    if validation_records:
        validation_dataset = CausalByteDataset(validation_records, training_settings.sequence_length)
        validation_loader = create_training_loader(
            validation_dataset,
            training_settings.batch_size,
            shuffle=False,
            num_workers=training_settings.num_workers,
            pin_memory=effective_pin_memory,
            prefetch_factor=training_settings.prefetch_factor,
        )
    model = KoemiModel(model_settings)
    engine = attach_offload(model, loader, training_settings.device, arguments, logger)
    try:
        result = Trainer(logger).train(model, loader, training_settings, validation_loader)
    finally:
        if engine is not None:
            log_offload(engine, logger)
            engine.detach()
    checkpoint_path = CheckpointStore().save(arguments.checkpoint, model, overwrite=arguments.overwrite)
    logger.info(
        "training_completed checkpoint=%s mean_loss=%.6f task_loss=%.6f thinking_loss=%.6f "
        "mean_surprise=%.4f validation_loss=%s validation_perplexity=%s optimizer_steps=%s "
        "tokens_per_second=%.2f final_learning_rate=%.8f precision=%s supervised_tokens=%s tokens=%s "
        "expert_activations=%s elapsed_seconds=%.3f",
        checkpoint_path,
        result.mean_loss,
        result.mean_task_loss,
        result.mean_thinking_loss,
        result.mean_surprise,
        result.validation_loss,
        result.validation_perplexity,
        result.optimizer_steps,
        result.tokens_per_second,
        result.final_learning_rate,
        result.precision,
        result.supervised_token_count,
        result.token_count,
        result.expert_activation_counts,
        result.elapsed_seconds,
    )
    return 0


def attach_offload(
    model: KoemiModel,
    loader,
    device: str,
    arguments: argparse.Namespace,
    logger,
) -> OffloadEngine | None:
    request = offload_request(arguments)
    if not request.requested:
        return None
    sample = loader.collate_fn([loader.dataset[0]])
    input_ids = sample["input_ids"].to(device)

    def calibration_forward() -> None:
        with torch.no_grad():
            model(input_ids)

    model.to(device)
    engine = prepare_offload(model, calibration_forward, request, device)
    log_offload(engine, logger)
    return engine


def attach_inference_offload(
    model: KoemiModel,
    tokenizer: ByteTokenizer,
    prompt: str,
    arguments: argparse.Namespace,
    logger,
) -> OffloadEngine | None:
    request = offload_request(arguments)
    if not request.requested:
        return None
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    prompt_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=arguments.device)

    def calibration_forward() -> None:
        with torch.no_grad():
            model(prompt_ids)

    engine = prepare_offload(model, calibration_forward, request, arguments.device)
    log_offload(engine, logger)
    return engine


def generate_completion(arguments: argparse.Namespace, logger) -> int:
    loaded_checkpoint = CheckpointStore().load(arguments.checkpoint, arguments.device)
    cache_capacity = arguments.cache_capacity or loaded_checkpoint.model_settings.cache_capacity
    warm_cache = WarmTokenCache(cache_capacity)
    if arguments.mapping_cache is not None and not arguments.mapping_cache_namespace:
        raise ValueError("--mapping-cache-namespace is required with --mapping-cache")
    mapping_cache = (
        DiskMappingCache(
            arguments.mapping_cache,
            capacity=arguments.mapping_cache_capacity,
            namespace=f"{arguments.mapping_cache_namespace}:{checkpoint_namespace(arguments.checkpoint)}",
            max_entry_bytes=arguments.mapping_cache_max_entry_mib * 1024 * 1024,
            ttl_seconds=arguments.mapping_cache_ttl_seconds,
        )
        if arguments.mapping_cache is not None
        else None
    )
    if mapping_cache is not None and arguments.clear_mapping_cache:
        logger.info("mapping_cache_cleared entries=%s", mapping_cache.clear())
    tokenizer = ByteTokenizer()
    engine = attach_inference_offload(
        loaded_checkpoint.model, tokenizer, arguments.prompt, arguments, logger
    )
    try:
        completion = generate_text(
            loaded_checkpoint.model,
            tokenizer,
            arguments.prompt,
            arguments.max_new_bytes,
            arguments.temperature,
            arguments.device,
            warm_cache,
            mapping_cache,
        )
    finally:
        if engine is not None:
            log_offload(engine, logger)
            engine.detach()
    statistics = warm_cache.statistics()
    mapping_statistics = mapping_cache.statistics() if mapping_cache is not None else None
    logger.info(
        "generation_completed generated_bytes=%s cache_hits=%s cache_misses=%s cache_evictions=%s "
        "mapping_hits=%s mapping_misses=%s mapping_evictions=%s mapping_expirations=%s mapping_deletions=%s",
        len(completion.encode("utf-8")),
        statistics.hits,
        statistics.misses,
        statistics.evictions,
        mapping_statistics.hits if mapping_statistics else 0,
        mapping_statistics.misses if mapping_statistics else 0,
        mapping_statistics.evictions if mapping_statistics else 0,
        mapping_statistics.expirations if mapping_statistics else 0,
        mapping_statistics.deletions if mapping_statistics else 0,
    )
    write_utf8(completion)
    return 0


def load_and_log_dataset(dataset_paths: list[str], dataset_format: str, logger) -> DatasetLoadReport:
    report = load_dataset_records(dataset_paths, dataset_format)
    logger.info(
        "dataset_loaded records=%s adapters=%s source_files=%s",
        report.record_count,
        report.adapter_counts,
        len(report.source_files),
    )
    return report


def create_model_settings(arguments: argparse.Namespace) -> ModelSettings:
    return ModelSettings(
        embedding_size=arguments.embedding_size,
        memory_features=arguments.memory_features,
        local_memory_size=arguments.local_memory_size,
        expert_count=arguments.expert_count,
        cache_capacity=arguments.cache_capacity,
        scan_chunk=arguments.scan_chunk,
        refine_decay_rate=arguments.refine_decay_rate,
        ablation=arguments.ablation,
    )


def write_utf8(value: str) -> None:
    encoded_value = f"{value}\n".encode("utf-8", errors="replace")
    stdout_buffer = getattr(sys.stdout, "buffer", None)
    if stdout_buffer is None:
        sys.stdout.write(encoded_value.decode("utf-8"))
        sys.stdout.flush()
        return
    stdout_buffer.write(encoded_value)
    stdout_buffer.flush()


def checkpoint_namespace(checkpoint_path: str) -> str:
    resolved_path = Path(checkpoint_path).expanduser().resolve()
    file_stat = resolved_path.stat()
    return f"{resolved_path}:{file_stat.st_size}:{file_stat.st_mtime_ns}"
