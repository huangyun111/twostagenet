"""Small single-node DDP helpers for training entrypoints."""

from __future__ import annotations

import os
from typing import Any

import torch
import torch.distributed as dist


def is_dist_avail_and_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def get_world_size() -> int:
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def is_main_process() -> bool:
    return get_rank() == 0


def setup_distributed(args: Any) -> tuple[torch.device, bool]:
    """Initialize torchrun DDP from RANK/WORLD_SIZE/LOCAL_RANK when present."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1

    setattr(args, "rank", rank)
    setattr(args, "world_size", world_size)
    setattr(args, "local_rank", local_rank)
    setattr(args, "distributed", distributed)

    if distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("DDP with backend='nccl' requires CUDA, but CUDA is not available.")
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        return torch.device("cuda", local_rank), True

    device_arg = getattr(args, "device", "auto")
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu"), False
    return torch.device(device_arg), False


def cleanup_distributed() -> None:
    if is_dist_avail_and_initialized():
        dist.destroy_process_group()


def main_process_print(*args: Any, **kwargs: Any) -> None:
    if is_main_process():
        print(*args, **kwargs)


def barrier() -> None:
    if is_dist_avail_and_initialized():
        dist.barrier()
