"""Parallel fitness evaluation using ``concurrent.futures``.

On macOS with MPS, we use sequential evaluation on the main process since
MPS doesn't support multiprocessing well.  On CUDA machines with multiple
GPUs, ``ProcessPoolExecutor`` can be used for true parallelism.

For CPU-only or when ``parallel_trainers == 1``, we fall back to sequential.
"""

from __future__ import annotations

import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

import torch
from loguru import logger

from src.genome import Genome
from src.trainer import get_device, train_genome


def _train_one(
    genome_dict: Dict[str, Any],
    cfg: Dict[str, Any],
    inherited_weights: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Pickle-friendly wrapper around :func:`train_genome`.

    Genomes and weights are passed as plain dicts so that they can be
    shipped across process boundaries.
    """
    from src.genome import Genome
    from src.trainer import get_cifar10_loaders, train_genome

    genome = Genome.from_dict(genome_dict)

    dcfg = cfg.get("data", {})
    train_loader, val_loader, _ = get_cifar10_loaders(
        batch_size=cfg["training"]["batch_size"],
        val_split=dcfg.get("val_split", 0.1),
        num_workers=cfg.get("num_workers", 2),
        data_dir=cfg.get("data_dir", "data"),
        augment=dcfg.get("augmentation", {}).get("random_flip", True),
    )

    result = train_genome(
        genome,
        train_loader,
        val_loader,
        cfg,
        existing_state_dict=inherited_weights,
    )
    # Attach genome dict with updated fitness back
    result["genome"] = genome.to_dict()
    # Remove state_dict from cross-process result to save memory
    result.pop("state_dict", None)
    return result


def evaluate_population(
    population: List[Genome],
    cfg: Dict[str, Any],
    train_loader=None,
    val_loader=None,
    weight_bank: Optional[Dict[str, Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Evaluate every genome in *population* and return result dicts.

    Parameters
    ----------
    population : list[Genome]
        Architectures to evaluate (fitness will be set in-place).
    cfg : dict
        Full YAML config.
    train_loader, val_loader :
        Shared data loaders (used in sequential mode).
    weight_bank : dict, optional
        Mapping ``genome_id → state_dict`` for weight inheritance.

    Returns
    -------
    list[dict]
        One result per genome (same order).
    """
    n_parallel = cfg.get("parallel_trainers", 1)
    device = get_device(cfg.get("device", "auto"))

    # On MPS or with 1 worker → run sequentially (safer on Apple Silicon)
    use_sequential = (
        n_parallel <= 1
        or device.type == "mps"
        or not torch.cuda.is_available()
    )

    results: List[Dict[str, Any]] = []

    if use_sequential:
        results = _sequential_eval(population, cfg, train_loader, val_loader, weight_bank)
    else:
        results = _parallel_eval(population, cfg, weight_bank, n_parallel)

    return results


def _sequential_eval(
    population: List[Genome],
    cfg: Dict[str, Any],
    train_loader,
    val_loader,
    weight_bank: Optional[Dict[str, Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    """Evaluate each genome one-by-one on the current process."""
    results = []
    for i, genome in enumerate(population):
        logger.info(
            f"Evaluating genome {i + 1}/{len(population)}: {genome.summary()}"
        )
        inherited = None
        if weight_bank and genome.parent_ids:
            for pid in genome.parent_ids:
                if pid in weight_bank:
                    inherited = weight_bank[pid]
                    break

        try:
            result = train_genome(
                genome, train_loader, val_loader, cfg,
                existing_state_dict=inherited,
            )
        except Exception as exc:
            logger.error(f"Genome {genome.id} crashed: {exc}")
            traceback.print_exc()
            result = {
                "fitness": 0.0, "epochs_trained": 0, "params": 0,
                "history": [], "early_stopped": True,
            }

        # Store weights for inheritance
        if weight_bank is not None and "state_dict" in result:
            weight_bank[genome.id] = result["state_dict"]

        genome.fitness = result["fitness"]
        result["genome"] = genome.to_dict()
        results.append(result)

        logger.info(
            f"  → fitness={result['fitness']:.4f}  "
            f"params={result.get('params', '?')}  "
            f"early_stopped={result.get('early_stopped', False)}"
        )

    return results


def _parallel_eval(
    population: List[Genome],
    cfg: Dict[str, Any],
    weight_bank: Optional[Dict[str, Dict[str, Any]]],
    n_workers: int,
) -> List[Dict[str, Any]]:
    """Evaluate genomes across multiple processes (CUDA multi-GPU)."""
    futures_map = {}
    results = [None] * len(population)

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        for i, genome in enumerate(population):
            inherited = None
            if weight_bank and genome.parent_ids:
                for pid in genome.parent_ids:
                    if pid in weight_bank:
                        inherited = weight_bank[pid]
                        break
            fut = pool.submit(_train_one, genome.to_dict(), cfg, inherited)
            futures_map[fut] = i

        for fut in as_completed(futures_map):
            idx = futures_map[fut]
            try:
                result = fut.result()
                population[idx].fitness = result["fitness"]
            except Exception as exc:
                logger.error(f"Worker crashed for genome idx={idx}: {exc}")
                result = {
                    "fitness": 0.0, "epochs_trained": 0, "params": 0,
                    "history": [], "early_stopped": True,
                    "genome": population[idx].to_dict(),
                }
                population[idx].fitness = 0.0
            results[idx] = result

    return results  # type: ignore[return-value]
