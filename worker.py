"""Pipeline worker — runs in an isolated subprocess per video.
GPU crashes here do NOT kill the main server."""
import logging
import os
import sys
import time

os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"

# numba JIT-compiles some of librosa's audio-feature code (a qwen-tts dependency)
# into a native .dll loaded via numba's NRT runtime. On at least one Windows
# machine an Application Control policy blocked that specific DLL mid-synthesis
# ("_nrt_python ... politica de Controle de Aplicativo bloqueou este arquivo"),
# killing the whole stage. Disabling JIT makes numba fall back to plain
# interpreted Python — slower for the handful of calls that use it, invisible
# next to GPU inference time, but it never touches that native code path at all.
os.environ["NUMBA_DISABLE_JIT"] = "1"

if "/app" not in sys.path:
    sys.path.insert(0, "/app")

if sys.platform == "win32":
    # huggingface_hub 0.24+ dropped HF_HUB_DISABLE_SYMLINKS support and relies on
    # a one-off symlink test per cache dir that can wrongly report success (Windows
    # symlink privilege is inconsistent across process/dir contexts), then a real
    # download later hits WinError 1314 and crashes the whole stage. Force "no
    # symlinks, always copy" unconditionally so model downloads never depend on it.
    try:
        import huggingface_hub.file_download as _hf_fd
        _hf_fd.are_symlinks_supported = lambda *a, **kw: False
    except ImportError:
        pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("worker")

_STAGE_NAMES = {
    "extract":    "Extrair Audio",
    "separate":   "Separar Vocais",
    "transcribe": "Transcrever",
    "translate":  "Traduzir",
    "synthesize": "Sintetizar Voz",
    "sync":       "Sincronizar",
    "merge":      "Mesclar Video",
}


# Which model_cache kinds each stage actually uses. With EVICT_IDLE_MODELS the
# rest are dropped from VRAM before the stage starts, so e.g. Whisper, Demucs
# and Opus-MT aren't sitting on a 12GB card while Qwen needs room to generate.
_STAGE_MODELS = {
    "separate": {"demucs"},
    "transcribe": {"whisper"},
    "translate": {"translation"},
    "synthesize": {"qwen3-tts", "qwen3-accent"},
}
# Qwen is the slowest to reload (~11s); keep it warm across videos.
_ALWAYS_WARM = {"qwen3-tts", "qwen3-accent"}


def run(project_id: str, session_id: str, config_overrides: dict, event_queue) -> None:
    """Execute the dubbing pipeline. Called in a subprocess by JobQueue."""

    def emit(event: str, data: dict) -> None:
        try:
            event_queue.put_nowait((session_id, event, data))
        except Exception:
            pass

    try:
        from config import Config
        from pipeline import STAGE_FUNCTIONS, PipelineState, DubbingPipeline

        config = Config()
        config.ensure_dirs()
        for key, val in config_overrides.items():
            if hasattr(config, key):
                setattr(config, key, val)

        project_dir = config.PROJECTS_DIR / project_id
        if not project_dir.exists():
            emit("error", {"message": f"Projeto nao encontrado: {project_id}"})
            return

        video_files = list((project_dir / "input").glob("*"))
        if not video_files:
            emit("error", {"message": "Nenhum video encontrado em input/"})
            return

        pipeline = DubbingPipeline(project_id=project_id, config=config)
        pipeline.setup_project(str(video_files[0]))

        state = pipeline.state
        completed = state.completed_stages
        stages_to_run = [s for s in config.STAGE_ORDER if s not in completed]
        if not config.enable_vocal_separation and "separate" in stages_to_run:
            stages_to_run.remove("separate")

        total_stages = len(stages_to_run)
        start_time = time.time()

        for stage_name in stages_to_run:
            display = _STAGE_NAMES.get(stage_name, stage_name)
            emit("stage_start", {
                "stage": stage_name,
                "display": display,
                "stages_done": len(state.completed_stages),
                "stages_total": total_stages,
            })
            state.mark_running(stage_name)

            def _progress(frac, desc="", _stage=stage_name, _display=display):
                done = len(state.completed_stages)
                overall = (done + frac) / max(total_stages, 1)
                elapsed = time.time() - start_time
                eta = (elapsed / max(overall, 0.01)) * (1.0 - overall) if overall > 0.05 else 0
                emit("progress", {
                    "frac": round(overall, 4),
                    "stage_frac": round(frac, 4),
                    "desc": f"{_display}: {desc}" if desc else _display,
                    "elapsed": round(elapsed, 1),
                    "eta": round(max(eta, 0.0), 1),
                    "stage": _stage,
                })

            if config.EVICT_IDLE_MODELS:
                from models import model_cache
                keep = _STAGE_MODELS.get(stage_name, set()) | _ALWAYS_WARM
                evicted = model_cache.keep_only(keep)
                if evicted:
                    logging.getLogger("dubbing").info(f"[GPU] Freed idle models before {stage_name}: {evicted}")

            stage_fn = STAGE_FUNCTIONS[stage_name]
            result = stage_fn(project_dir, config, progress=_progress)

            if result.success:
                # Guard: transcribe must produce at least 1 cue or everything downstream is wasted
                if stage_name == "transcribe":
                    en_srt = project_dir / "work" / "en.srt"
                    if not en_srt.exists() or en_srt.stat().st_size < 10:
                        err = "Transcricao retornou vazio — audio pode ser silencioso ou corrompido"
                        state.mark_error(stage_name, err)
                        emit("stage_error", {"stage": stage_name, "display": display, "error": err})
                        return

                state.mark_completed(stage_name, metadata=result.metadata or {})
                emit("stage_done", {
                    "stage": stage_name,
                    "display": display,
                    "warnings": (result.metadata or {}).get("warnings", []),
                    "elapsed_total": round(time.time() - start_time, 1),
                })
            else:
                state.mark_error(stage_name, result.error or "Erro desconhecido")
                emit("stage_error", {
                    "stage": stage_name,
                    "display": display,
                    "error": result.error or "Erro desconhecido",
                })
                return

        total_time = time.time() - start_time
        output_dir = config.PROJECTS_DIR / project_id / "output"
        result_files = {}
        if output_dir.exists():
            for f in output_dir.iterdir():
                if f.is_file():
                    result_files[f.name] = f"/api/file/{project_id}/output/{f.name}"

        emit("done", {
            "total_time": round(total_time, 1),
            "project_id": project_id,
            "files": result_files,
        })

    except Exception as exc:
        logger.exception("Worker pipeline error")
        emit("error", {"message": str(exc)})


# Cached models keep their weights resident in VRAM across jobs — great for speed,
# but on a long batch (10+ videos back to back) the fragmentation/growth from
# hours of continuous Demucs/Whisper/Qwen3-TTS inference activations eventually risks
# a CUDA OOM crash. Recycling the whole worker process periodically resets VRAM
# to a clean state, trading one ~20s model reload every N videos for a process
# that survives an all-night batch instead of dying partway through it.
MAX_JOBS_PER_WORKER = 10


def run_persistent(job_queue, event_queue) -> None:
    """Long-lived worker loop — processes jobs one at a time from job_queue,
    in the SAME process, so module-level model caches (models/model_cache.py)
    stay warm across videos instead of reloading Demucs/Whisper/Qwen3-TTS from
    scratch every time.

    A regular Python exception in run() for one job is caught here so the
    loop keeps serving future jobs. A hard crash (segfault, driver-level CUDA
    fault) still takes the whole process down — server.py detects that via
    is_alive() and spawns a fresh persistent worker, same as before, just
    losing the warm cache for that one restart instead of every job."""
    logger.info(f"[PERSISTENT WORKER] Started, PID {os.getpid()}")
    jobs_done = 0
    while True:
        item = job_queue.get()
        if item is None:
            logger.info("[PERSISTENT WORKER] Stop signal received, exiting")
            break
        project_id, session_id, config_overrides = item
        jobs_done += 1
        logger.info(f"{'=' * 60}")
        logger.info(f"[PERSISTENT WORKER] Job #{jobs_done} on this worker — project {project_id}")
        logger.info(f"{'=' * 60}")
        try:
            run(project_id, session_id, config_overrides, event_queue)
        except Exception:
            logger.exception(f"[PERSISTENT WORKER] Unhandled error running job for {project_id}")
            try:
                event_queue.put((session_id, "error", {"message": "Erro interno no worker."}))
            except Exception:
                pass

        try:
            import gc
            import torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

        if jobs_done >= MAX_JOBS_PER_WORKER:
            logger.info(
                f"[PERSISTENT WORKER] Reached {MAX_JOBS_PER_WORKER} jobs — recycling "
                f"worker to reset VRAM before it fragments further"
            )
            break

    # multiprocessing.Queue flushes via a background feeder thread — exiting the
    # process right after put() can drop the last event(s) before they're sent.
    # Block here until everything queued (including this job's "done"/"error"
    # event) has actually been handed off, so the server never loses track of
    # the last job a recycled/stopping worker finished.
    try:
        event_queue.close()
        event_queue.join_thread()
    except Exception:
        pass
