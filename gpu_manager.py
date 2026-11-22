import gc
import os
import subprocess

import torch


class GPUManager:
    _current_stage: str | None = None
    _current_model: object | None = None

    @classmethod
    def acquire(cls, stage_name: str) -> None:
        if cls._current_stage and cls._current_stage != stage_name:
            cls.release(cls._current_stage)
        cls._current_stage = stage_name

    @classmethod
    def release(cls, stage_name: str) -> None:
        if cls._current_stage != stage_name:
            return
        cls._current_model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        cls._current_stage = None

    @classmethod
    def set_model(cls, model: object) -> None:
        cls._current_model = model

    @classmethod
    def get_vram_usage(cls) -> dict:
        if not torch.cuda.is_available():
            return {"used_mb": 0, "total_mb": 0, "free_mb": 0}
        try:
            cmd = ["nvidia-smi", "--query-gpu=memory.used,memory.total",
                   "--format=csv,noheader,nounits"]
            # Servidor preso a uma placa (CUDA_VISIBLE_DEVICES=1): mostra a dela.
            visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")[0].strip()
            if visible.isdigit():
                cmd += ["-i", visible]
            result = subprocess.run(
                cmd,
                capture_output=True, text=True, errors='replace', timeout=10,
            )
            # Pods com várias GPUs listam uma por linha: usa a primeira (cuda:0).
            parts = result.stdout.strip().splitlines()[0].split(", ")
            used = int(parts[0])
            total = int(parts[1])
            return {"used_mb": used, "total_mb": total, "free_mb": total - used}
        except Exception:
            # Sem nvidia-smi: pergunta direto pro driver (vale pra qualquer placa).
            free_b, total_b = torch.cuda.mem_get_info()
            total = total_b // (1024 * 1024)
            free = free_b // (1024 * 1024)
            return {"used_mb": total - free, "total_mb": total, "free_mb": free}

    @classmethod
    def get_vram_fraction(cls) -> float:
        info = cls.get_vram_usage()
        if info["total_mb"] == 0:
            return 0.0
        return info["used_mb"] / info["total_mb"]

    @classmethod
    def is_available(cls) -> bool:
        return torch.cuda.is_available()

    @classmethod
    def get_device(cls) -> str:
        return "cuda" if torch.cuda.is_available() else "cpu"