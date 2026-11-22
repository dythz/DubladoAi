"""Flask server with SSE for real-time pipeline progress."""

import sys as _sys

# gevent monkey-patch must be first — makes Queue.get() and socket I/O non-blocking
# so SSE streams yield to other greenlets instead of holding OS threads.
# Only useful under the gunicorn+gevent worker below. gunicorn itself requires
# fcntl (Unix-only) and is never usable on Windows, so patching sockets there
# just breaks waitress/the Flask dev server with BlockingIOError on every request.
if _sys.platform != "win32":
    try:
        from gevent import monkey as _monkey
        _monkey.patch_all()
    except ImportError:
        pass

import asyncio
import json
import logging
import multiprocessing as mp
import os
import time
import uuid
from pathlib import Path
from urllib.parse import quote
from queue import Queue, Empty, Full
from threading import Thread, Lock

from flask import Flask, request, jsonify, Response, send_file, send_from_directory
from flask_cors import CORS

os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"

if _sys.platform == "win32":
    # See worker.py for why this is needed: huggingface_hub's symlink-support
    # detection can misfire on Windows and crash a real download later with
    # WinError 1314. Force "always copy" so it never depends on that detection.
    try:
        import huggingface_hub.file_download as _hf_fd
        _hf_fd.are_symlinks_supported = lambda *a, **kw: False
    except ImportError:
        pass

# Ensure spawned worker subprocesses can import project modules from this directory.
# mp.get_context("spawn") starts a fresh Python interpreter that does NOT inherit
# sys.path from the parent — it only sees site-packages unless PYTHONPATH is set.
_APP_DIR = str(Path(__file__).parent.resolve())
if _APP_DIR not in os.environ.get("PYTHONPATH", "").split(os.pathsep):
    os.environ["PYTHONPATH"] = _APP_DIR + os.pathsep + os.environ.get("PYTHONPATH", "")

if hasattr(asyncio, "WindowsSelectorEventLoopPolicy"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from config import Config
from pipeline import DubbingPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("server")

app = Flask(__name__, static_folder="static", static_url_path="/static")
CORS(app)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024 * 1024  # 10GB upload limit

CONFIG = Config()
CONFIG.ensure_dirs()

# The persistent worker is started with mp "spawn", which re-imports this file
# as __mp_main__ inside the child process — so every module-level call below
# also runs again in each new worker (every MAX_JOBS_PER_WORKER jobs). Startup
# housekeeping must only happen in the real server process, otherwise a worker
# recycle wipes the chunks of any upload still in flight.
# (mp.parent_process() is NOT usable here — it's still None while spawn is
# re-importing the main module; the process name is already set by then.)
_IS_MAIN_PROCESS = mp.current_process().name == "MainProcess"

# Clean up any stale chunk directories left by interrupted uploads
def _cleanup_stale_chunks():
    import shutil
    try:
        for chunks_dir in CONFIG.PROJECTS_DIR.glob("*/chunks"):
            shutil.rmtree(str(chunks_dir), ignore_errors=True)
            logger.info(f"[STARTUP] Removed stale chunks: {chunks_dir}")
    except Exception:
        pass
if _IS_MAIN_PROCESS:
    _cleanup_stale_chunks()

# Events that end a job (and its SSE stream).
TERMINAL_EVENTS = ("done", "error", "stage_error")


# ── Organização por pastas ──────────────────────────────────────────────────
# Quando o site manda a pasta do curso inteira, cada vídeo vem com o caminho
# dentro dela ("NK101/1. Intro/2. Aula.mp4"). Terminada a dublagem, o vídeo
# vai pra data/dublados/<mesmo caminho>, com as legendas ao lado.
LIBRARY_DIR = Path(os.environ.get("DUB_LIBRARY_DIR", str(CONFIG.BASE_DIR / "data" / "dublados")))
_BAD_CHARS = '<>:"|?*'


def _safe_rel_path(raw: str | None) -> str:
    """'Curso/Modulo/aula.mp4' sem '..', sem drive, sem caracteres invalidos."""
    if not raw:
        return ""
    parts = []
    for part in str(raw).replace("\\", "/").split("/"):
        part = "".join(c for c in part if c not in _BAD_CHARS and ord(c) >= 32).strip().rstrip(".")
        if part and part not in (".", ".."):
            parts.append(part)
    return "/".join(parts)


def _project_meta(project_id: str) -> dict:
    try:
        with open(CONFIG.PROJECTS_DIR / project_id / "meta.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def _save_rel_path(project_id: str, raw: str | None) -> None:
    rel = _safe_rel_path(raw)
    if not rel or "/" in project_id or "\\" in project_id or ".." in project_id:
        return
    meta_path = CONFIG.PROJECTS_DIR / project_id / "meta.json"
    if meta_path.exists():
        return
    try:
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({"rel_path": rel}, f, ensure_ascii=False)
    except OSError as e:
        logger.warning(f"[LIBRARY] Couldn't save folder path for {project_id}: {e}")


def _library_name(project_id: str, video: Path) -> str:
    """Caminho do dublado dentro da biblioteca/zip: o mesmo da pasta original,
    ou só o nome do arquivo quando o vídeo veio solto."""
    rel = _project_meta(project_id).get("rel_path")
    if rel:
        return str(Path(rel).with_suffix(".mp4")).replace("\\", "/")
    return video.name.replace("_dublado_pt", "")


def _link_or_copy(src: Path, dst: Path) -> None:
    import shutil
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)  # mesmo disco: instantâneo e não ocupa espaço extra
    except OSError:
        shutil.copy2(src, dst)


def _publish_to_library(project_id: str) -> None:
    video = _dubbed_video(project_id)
    if video is None:
        return
    try:
        target = LIBRARY_DIR / _library_name(project_id, video)
        _link_or_copy(video, target)
        out = video.parent
        for lang in ("pt", "en"):
            srt = out / f"{lang}.srt"
            if srt.exists():
                _link_or_copy(srt, target.with_suffix(f".{lang}.srt"))
        logger.info(f"[LIBRARY] {target}")
    except Exception as e:
        logger.warning(f"[LIBRARY] Couldn't organize {project_id}: {e}")
    if CONFIG.CLEAN_WORK_AFTER_DONE:
        _clean_work(project_id)


def _clean_work(project_id: str) -> None:
    """Apaga os áudios intermediários (~200 MB por vídeo). Ficam as legendas,
    o state.json e dubbed_vocals.wav (pra refazer só a mixagem se precisar)."""
    import shutil
    work = CONFIG.PROJECTS_DIR / project_id / "work"
    if not work.is_dir():
        return
    freed = 0
    for name in ("audio_full.wav", "audio_whisper.wav", "vocals.wav", "background.wav",
                 "vocals_enhanced.wav", "final_audio.wav"):
        f = work / name
        if f.exists():
            freed += f.stat().st_size
            f.unlink(missing_ok=True)
    for d in ("pt_segments", "pt_synced"):
        p = work / d
        if p.is_dir():
            freed += sum(x.stat().st_size for x in p.rglob("*") if x.is_file())
            shutil.rmtree(p, ignore_errors=True)
    if freed:
        logger.info(f"[CLEAN] {project_id}: {freed / 1024 / 1024:.0f} MB de arquivos de trabalho apagados")

# Active SSE queues keyed by session_id
sse_queues: dict[str, Queue] = {}

# Per-project locks guarding chunk-upload assembly — with several videos
# uploading in parallel (each sending its own chunks in parallel batches),
# two requests can both see "all chunks present" and race to assemble +
# delete the same chunks dir at once, crashing the second one.
_upload_locks: dict[str, Lock] = {}
_upload_locks_guard = Lock()


def _get_upload_lock(project_id: str) -> Lock:
    with _upload_locks_guard:
        if project_id not in _upload_locks:
            _upload_locks[project_id] = Lock()
        return _upload_locks[project_id]


# ── Job Queue ─────────────────────────────────────────────────────────

class JobQueue:
    """FIFO queue that runs video pipelines in a single persistent worker
    process — kept alive across jobs so loaded models (Demucs/Whisper/Qwen3-TTS,
    see models/model_cache.py) stay warm in VRAM instead of reloading from
    scratch every video. If the worker process dies (hard GPU/driver crash),
    it's detected and a fresh one is spawned for the next job — that one job
    is lost, same as before, but the server itself keeps running."""

    MAX_JOB_SECONDS = 7200  # 2 hour hard limit per job

    def __init__(self):
        self._lock = Lock()
        self._queue: list[dict] = []
        self._active: dict | None = None
        self._done: list[dict] = []
        self._counter = 0
        self._mp_ctx = mp.get_context("spawn")
        self._job_input_queue = self._mp_ctx.Queue()
        self._event_queue = self._mp_ctx.Queue()
        # Dicas "o próximo da fila é X" pro worker adiantar as primeiras etapas
        # dele enquanto o atual gera a voz (config.PIPELINE_OVERLAP).
        self._prefetch_queue = self._mp_ctx.Queue()
        self._worker_process = None
        self._worker_lock = Lock()
        # Mirrors worker.MAX_JOBS_PER_WORKER — see _retire_worker_if_due below.
        self._jobs_sent_to_worker = 0

    def _ensure_worker_alive(self):
        with self._worker_lock:
            if self._worker_process is not None and self._worker_process.is_alive():
                return
            import worker as _worker_module
            self._worker_process = self._mp_ctx.Process(
                target=_worker_module.run_persistent,
                args=(self._job_input_queue, self._event_queue, self._prefetch_queue),
                daemon=False,
            )
            self._worker_process.start()
            self._jobs_sent_to_worker = 0
            logger.info(f"[WORKER] Persistent worker started, PID {self._worker_process.pid}")

    def _retire_worker_if_due(self):
        """After sending the same job count worker.py itself recycles at, block
        until that worker process has actually fully exited before returning.

        Without this, the moment a job's "done" event arrives, the queue thread
        advances to the next job and calls _ensure_worker_alive() essentially
        immediately — but a process that just returned from run_persistent()
        can still report is_alive()==True for a brief window while the OS
        tears it down. _ensure_worker_alive() would then wrongly conclude the
        recycling worker is still good to use, hand it the next job, and that
        job would sit in _job_input_queue forever — the old worker's loop has
        already exited and will never read it — while the job's own _execute()
        eventually times out waiting and reports a false "GPU crashed" error."""
        import worker as _worker_module
        with self._worker_lock:
            if self._jobs_sent_to_worker < _worker_module.MAX_JOBS_PER_WORKER:
                return
            if self._worker_process is not None:
                self._worker_process.join(timeout=30)
                if self._worker_process.is_alive():
                    logger.warning("[WORKER] Recycling worker didn't exit within 30s, forcing it")
                    self._worker_process.kill()
                    self._worker_process.join(timeout=5)
            self._jobs_sent_to_worker = 0

    def _hint_next_job(self):
        """Tell the worker which video comes next (caller holds self._lock)."""
        if self._queue:
            nxt = self._queue[0]
            try:
                self._prefetch_queue.put_nowait((nxt["project_id"], nxt["config_overrides"]))
            except Exception:
                pass

    def enqueue(self, session_id: str, project_id: str, filename: str, config_overrides: dict) -> int:
        with self._lock:
            self._counter += 1
            job = {
                "id": self._counter,
                "session_id": session_id,
                "project_id": project_id,
                "filename": filename,
                "config_overrides": config_overrides,
                "status": "queued",
                "progress": 0.0,
                "stage": "",
                "started_at": None,
            }
            if self._active is None:
                self._active = job
                job["status"] = "running"
                job["started_at"] = time.time()
                Thread(target=self._execute, args=(job,), daemon=True).start()
                return 0
            else:
                self._queue.append(job)
                if len(self._queue) == 1:
                    self._hint_next_job()
                return len(self._queue)

    def update_job(self, session_id: str, progress: float, stage: str):
        with self._lock:
            if self._active and self._active["session_id"] == session_id:
                self._active["progress"] = progress
                self._active["stage"] = stage

    def _execute(self, job: dict):
        try:
            self._ensure_worker_alive()
            self._job_input_queue.put((job["project_id"], job["session_id"], job["config_overrides"]))
            with self._worker_lock:
                self._jobs_sent_to_worker += 1
            logger.info(f"[WORKER] Job {job['id']} ({job['filename']}) sent to persistent worker")
            with self._lock:
                self._hint_next_job()
            job_start = time.time()

            while True:
                # Hard timeout — kill and replace the persistent worker, losing its warm cache
                if time.time() - job_start > self.MAX_JOB_SECONDS:
                    logger.error(f"[WORKER] Job {job['id']} timed out, killing worker")
                    if self._worker_process is not None and self._worker_process.is_alive():
                        self._worker_process.terminate()
                        self._worker_process.join(timeout=10)
                        if self._worker_process.is_alive():
                            self._worker_process.kill()
                    job["result"], job["error"] = "error", "Limite de 2 horas atingido"
                    sse_send(job["session_id"], "error", {"message": "Job atingiu limite de 2 horas e foi cancelado."})
                    break

                if self._worker_process is None or not self._worker_process.is_alive():
                    logger.error(f"[WORKER] Persistent worker died while running job {job['id']}")
                    while not self._event_queue.empty():
                        try:
                            sid, event, data = self._event_queue.get_nowait()
                            if sid == job["session_id"]:
                                self.update_job(sid, data.get("frac", 0), data.get("desc", ""))
                                sse_send(sid, event, data)
                        except Exception:
                            pass
                    job["result"], job["error"] = "error", "GPU travou"
                    sse_send(job["session_id"], "error", {
                        "message": "GPU travou. Tente reenviar o video."
                    })
                    break

                try:
                    sid, event, data = self._event_queue.get(timeout=0.5)
                except Exception:
                    continue

                # Jobs run strictly sequentially in the persistent worker, so events
                # should always belong to this job — forward stragglers just in case.
                if sid != job["session_id"]:
                    sse_send(sid, event, data)
                    continue

                self.update_job(sid, data.get("frac", 0), data.get("desc", ""))
                sse_send(sid, event, data)
                if event in TERMINAL_EVENTS:
                    if event == "done":
                        job["result"] = "ok"
                        _publish_to_library(job["project_id"])
                    else:
                        job["result"] = "error"
                        job["error"] = data.get("error") or data.get("message") or "Erro"
                    self._retire_worker_if_due()
                    break

        except Exception as e:
            logger.exception(f"[WORKER] Job {job['id']} failed: {e}")
            job["result"], job["error"] = "error", str(e)
            sse_send(job["session_id"], "error", {"message": f"Erro ao processar video: {e}"})
        finally:
            with self._lock:
                job["status"] = "done"
                job.setdefault("result", "error")
                job["finished_at"] = time.time()
                self._done.append(job)
                if len(self._done) > 20:
                    self._done = self._done[-20:]
                self._active = None
                sse_queues.pop(job["session_id"], None)
                if self._queue:
                    next_job = self._queue.pop(0)
                    self._active = next_job
                    next_job["status"] = "running"
                    next_job["started_at"] = time.time()
                    sse_send(next_job["session_id"], "queue_start", {
                        "position": 0,
                        "message": "Seu video esta sendo processado",
                    })
                    Thread(target=self._execute, args=(next_job,), daemon=True).start()

    def get_position(self, session_id: str) -> dict:
        with self._lock:
            if self._active and self._active["session_id"] == session_id:
                return {"status": "running", "position": 0}
            for i, job in enumerate(self._queue):
                if job["session_id"] == session_id:
                    return {"status": "queued", "position": i + 1}
            return {"status": "unknown", "position": -1}

    def get_all_jobs(self) -> list[dict]:
        with self._lock:
            jobs = []
            if self._active:
                a = dict(self._active)
                a.pop("run_fn", None)
                jobs.append(a)
            for j in self._queue:
                s = dict(j)
                s.pop("run_fn", None)
                jobs.append(s)
            for j in self._done[-10:]:
                s = dict(j)
                s.pop("run_fn", None)
                jobs.append(s)
            return jobs

    @property
    def queue_length(self) -> int:
        with self._lock:
            return len(self._queue)


job_queue = JobQueue()


def _read_state(project_dir: Path) -> dict:
    state_file = project_dir / "state.json"
    if not state_file.exists():
        return {}
    try:
        with open(state_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _read_completed_stages(project_dir: Path) -> list:
    return _read_state(project_dir).get("completed_stages", [])


def _video_key(video: Path) -> tuple:
    # Name + size: different courses often reuse names like "Lesson 1.mp4".
    try:
        return (video.name, video.stat().st_size)
    except OSError:
        return (video.name, -1)


def _resume_unfinished_projects():
    """Self-healing on boot: if the server died mid-batch (crash, power loss,
    manual kill), the in-memory job queue is gone but each project's progress
    is still on disk (state.json + whatever work/ files it got through). Scan
    every project and re-enqueue anything genuinely interrupted mid-pipeline.

    Deliberately conservative about what counts as "interrupted":
    - No state.json at all means this project's pipeline never actually
      started (e.g. a video was uploaded but "Iniciar Dublagem" was never
      clicked, or a stale duplicate from re-uploading the same file) — do
      NOT auto-start those; only resume work that was demonstrably underway.
    - If a project has completed_stages but none reached "merge", it was
      cut off mid-run — that's the real crash-recovery case, resume it.
    - Skip a video whose filename already has a finished output under a
      DIFFERENT project_id — with the same file uploaded multiple times
      across test runs, only the video actually needs producing once.
    """
    if not CONFIG.PROJECTS_DIR.exists():
        return
    all_projects = sorted(
        (p for p in CONFIG.PROJECTS_DIR.iterdir() if p.is_dir()),
        key=lambda p: p.stat().st_mtime,
    )

    already_done_filenames = set()
    for project_dir in all_projects:
        video_files = list((project_dir / "input").glob("*")) if (project_dir / "input").exists() else []
        if not video_files:
            continue
        if "merge" in _read_completed_stages(project_dir):
            already_done_filenames.add(_video_key(video_files[0]))

    resumed = 0
    for project_dir in all_projects:
        project_id = project_dir.name
        video_files = list((project_dir / "input").glob("*")) if (project_dir / "input").exists() else []
        if not video_files:
            continue  # upload never completed — nothing to resume

        state_file = project_dir / "state.json"
        if not state_file.exists():
            continue  # never actually started — don't assume it should run

        state = _read_state(project_dir)
        completed_stages = state.get("completed_stages", [])
        if "merge" in completed_stages:
            continue  # already finished
        if state.get("error"):
            continue  # failed with an error, not a crash: rerunning would just fail again

        filename = video_files[0].name
        if _video_key(video_files[0]) in already_done_filenames:
            continue  # a different attempt at this same video already finished

        overrides = {}
        job_file = project_dir / "job.json"
        if job_file.exists():
            try:
                with open(job_file, "r", encoding="utf-8") as f:
                    overrides = json.load(f)
            except Exception:
                pass

        session_id = f"{project_id}-resume-{int(time.time())}"
        job_queue.enqueue(session_id, project_id, filename, overrides)
        resumed += 1
        logger.info(f"[STARTUP] Resuming interrupted project {project_id} ({filename}) — had: {completed_stages}")

    if resumed:
        logger.info(f"[STARTUP] Re-queued {resumed} unfinished project(s) from a previous run")


if _IS_MAIN_PROCESS:
    _resume_unfinished_projects()


# ── SSE helpers ──────────────────────────────────────────────────────

def sse_send(session_id: str, event: str, data: dict):
    """Push an event to all listeners for a session (non-blocking — drops if client disconnected)."""
    q = sse_queues.get(session_id)
    if q is None:
        return
    try:
        q.put_nowait((event, data))
    except Full:
        if event == "progress":
            return  # a newer progress event will follow
        try:
            q.get_nowait()  # make room by dropping the oldest event
            q.put_nowait((event, data))
        except (Empty, Full):
            pass  # never block the job thread


def progress_callback_factory(session_id: str):
    """Return a progress callback that pushes SSE events."""
    def cb(frac: float, desc: str = ""):
        sse_send(session_id, "progress", {"frac": round(frac, 4), "desc": desc})
    return cb


# ── Routes ───────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/test")
def test_page():
    return send_from_directory("static", "test.html")


@app.route("/api/upload", methods=["POST"])
def upload():
    """Upload a video file and return a project_id."""
    logger.info(f"[UPLOAD] Request files: {list(request.files.keys())}")
    logger.info(f"[UPLOAD] Content-Length: {request.content_length}")
    if "video" not in request.files:
        logger.error(f"[UPLOAD] No 'video' in request. Keys: {list(request.files.keys())}")
        return jsonify({"error": "No video file provided"}), 400
    f = request.files["video"]
    if not f.filename:
        return jsonify({"error": "Empty filename"}), 400

    ext = Path(f.filename).suffix.lower()
    if ext not in (".mp4", ".mov", ".avi", ".mkv", ".webm"):
        return jsonify({"error": f"Unsupported format: {ext}"}), 400

    project_id = uuid.uuid4().hex[:12]
    project_dir = CONFIG.PROJECTS_DIR / project_id
    project_dir.mkdir(parents=True, exist_ok=True)
    (project_dir / "input").mkdir(exist_ok=True)
    (project_dir / "output").mkdir(exist_ok=True)

    dest = project_dir / "input" / f.filename
    f.save(str(dest))
    _save_rel_path(project_id, request.form.get("rel_path"))

    return jsonify({"project_id": project_id, "filename": f.filename})


@app.route("/api/upload/chunk", methods=["POST"])
def upload_chunk():
    """Receive one 48 MB chunk of a large file upload and reassemble when complete."""
    import shutil
    project_id = request.form.get("project_id", "")
    chunk_index = int(request.form.get("chunk_index", 0))
    total_chunks = int(request.form.get("total_chunks", 1))
    filename = request.form.get("filename", "video.mp4")
    chunk = request.files.get("chunk")

    if not chunk:
        return jsonify({"error": "No chunk data"}), 400

    ext = Path(filename).suffix.lower()
    if ext not in (".mp4", ".mov", ".avi", ".mkv", ".webm"):
        return jsonify({"error": f"Unsupported format: {ext}"}), 400

    if not project_id:
        project_id = uuid.uuid4().hex[:12]

    project_dir = CONFIG.PROJECTS_DIR / project_id
    chunks_dir = project_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    (project_dir / "input").mkdir(exist_ok=True)
    (project_dir / "output").mkdir(exist_ok=True)

    _save_rel_path(project_id, request.form.get("rel_path"))
    chunk_path = chunks_dir / f"chunk_{chunk_index:05d}"
    chunk.save(str(chunk_path))
    logger.info(f"[UPLOAD] Chunk {chunk_index+1}/{total_chunks} for {filename} (project {project_id})")

    received = len(list(chunks_dir.glob("chunk_*")))
    all_present = received >= total_chunks and all(
        (chunks_dir / f"chunk_{i:05d}").exists() for i in range(total_chunks)
    )
    if all_present:
        dest = project_dir / "input" / filename
        # Multiple chunk-upload requests for this project can arrive concurrently
        # (parallel batches) — only one should assemble the file and clean up.
        with _get_upload_lock(project_id):
            if not dest.exists():
                with open(str(dest), "wb") as out:
                    for i in range(total_chunks):
                        cp = chunks_dir / f"chunk_{i:05d}"
                        with open(str(cp), "rb") as cf:
                            out.write(cf.read())
                shutil.rmtree(str(chunks_dir), ignore_errors=True)
                logger.info(f"[UPLOAD] Assembled {filename} ({dest.stat().st_size // 1024 // 1024} MB)")
        with _upload_locks_guard:
            _upload_locks.pop(project_id, None)
        return jsonify({"project_id": project_id, "filename": filename, "complete": True})

    return jsonify({"project_id": project_id, "chunk_index": chunk_index, "received": received, "complete": False})


@app.route("/api/jobs")
def list_jobs():
    """List all queue jobs (active, queued, recent done)."""
    return jsonify({"jobs": job_queue.get_all_jobs()})


@app.route("/api/gpu")
def gpu_status():
    """Live GPU name / VRAM / temperature for the header widget."""
    import subprocess
    try:
        cmd = ["nvidia-smi", "--query-gpu=name,memory.used,memory.total,temperature.gpu,utilization.gpu",
               "--format=csv,noheader,nounits"]
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")[0].strip()
        if visible.isdigit():  # pod com várias placas: mostra a deste servidor
            cmd += ["-i", visible]
        out = subprocess.run(
            cmd,
            capture_output=True, text=True, errors="replace", timeout=5,
        ).stdout.strip().splitlines()[0]
        name, used, total, temp, util = [p.strip() for p in out.split(",")]
        return jsonify({"ok": True, "name": name.replace("NVIDIA GeForce ", ""), "used_mb": int(used),
                        "total_mb": int(total), "temp_c": int(temp), "util": int(util)})
    except Exception:
        return jsonify({"ok": False})


@app.route("/api/downloads")
def list_downloads():
    """List every finished dubbed video across all projects, newest first."""
    videos = []
    if CONFIG.PROJECTS_DIR.exists():
        for project_dir in CONFIG.PROJECTS_DIR.iterdir():
            if not project_dir.is_dir():
                continue
            output_dir = project_dir / "output"
            if not output_dir.exists():
                continue
            for f in output_dir.iterdir():
                if f.is_file() and f.suffix.lower() == ".mp4" and "_dublado_pt" in f.name:
                    stat = f.stat()
                    url = f"/api/file/{project_dir.name}/output/{quote(f.name)}"
                    videos.append({
                        "project_id": project_dir.name,
                        "filename": f.name,
                        "rel_path": _project_meta(project_dir.name).get("rel_path", ""),
                        "size_mb": round(stat.st_size / 1024 / 1024, 1),
                        "modified": stat.st_mtime,
                        "url": url,
                        "download_url": url + "?download=1",
                    })
    videos.sort(key=lambda v: v["modified"], reverse=True)
    return jsonify({"videos": videos})


def _dubbed_video(project_id: str) -> Path | None:
    output_dir = CONFIG.PROJECTS_DIR / project_id / "output"
    if "/" in project_id or ".." in project_id or not output_dir.is_dir():
        return None
    for f in output_dir.iterdir():
        if f.is_file() and f.suffix.lower() == ".mp4" and "_dublado_pt" in f.name:
            return f
    return None


class _ZipStream:
    """Write-only sink for zipfile that hands written bytes to a generator.
    Not seekable, so zipfile streams entries with data descriptors."""

    def __init__(self):
        self._chunks = []

    def write(self, b):
        self._chunks.append(bytes(b))
        return len(b)

    def flush(self):
        pass

    def pop(self) -> bytes:
        data, self._chunks = b"".join(self._chunks), []
        return data


@app.route("/api/downloads/zip")
def download_zip():
    import zipfile
    ids = [i for i in (request.args.get("ids") or "").split(",") if i]
    if not ids:
        ids = [d.name for d in CONFIG.PROJECTS_DIR.iterdir() if d.is_dir()] if CONFIG.PROJECTS_DIR.exists() else []
    files = [(i, f) for i, f in ((i, _dubbed_video(i)) for i in ids) if f]
    if not files:
        return jsonify({"error": "Nenhum video pronto para baixar"}), 404

    def generate():
        sink = _ZipStream()
        used = set()
        # Videos are already compressed: store them as-is (fast, no CPU).
        with zipfile.ZipFile(sink, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as zf:
            # Mesma estrutura de pastas do curso (quando veio de uma pasta),
            # em ordem: NK101/1. Intro/2. Aula.mp4 ...
            import re as _re
            def _natural(pf):
                return [int(t) if t.isdigit() else t for t in _re.split(r"(\d+)", _library_name(*pf).lower())]
            files.sort(key=_natural)
            for pid, f in files:
                name = _library_name(pid, f)
                base, n = name[:-4], 2
                while name in used:  # same lesson name twice
                    name = f"{base} ({n}).mp4"; n += 1
                used.add(name)
                with zf.open(name, "w", force_zip64=True) as dest, open(f, "rb") as src:
                    while True:
                        chunk = src.read(1024 * 1024)
                        if not chunk:
                            break
                        dest.write(chunk)
                        yield sink.pop()
                yield sink.pop()
        yield sink.pop()

    stamp = time.strftime("%Y-%m-%d")
    return Response(generate(), mimetype="application/zip", headers={
        "Content-Disposition": f"attachment; filename=\"videos_dublados_{stamp}.zip\"",
    })


@app.route("/api/downloads/<project_id>", methods=["DELETE"])
def delete_download(project_id):
    """Remove a project's dubbed output (frees disk — keeps input/work untouched)."""
    output_dir = CONFIG.PROJECTS_DIR / project_id / "output"
    if not output_dir.exists():
        return jsonify({"error": "Not found"}), 404
    import shutil
    shutil.rmtree(str(output_dir), ignore_errors=True)
    return jsonify({"status": "deleted", "project_id": project_id})


@app.route("/api/start", methods=["POST"])
def start_pipeline():
    """Start the dubbing pipeline for a project (queued)."""
    data = request.json or {}
    project_id = data.get("project_id")
    if not project_id:
        return jsonify({"error": "project_id required"}), 400

    input_dir = CONFIG.PROJECTS_DIR / project_id / "input"
    if not input_dir.exists() or not any(input_dir.iterdir()):
        return jsonify({"error": "Upload incompleto: video nao encontrado no servidor. Reenvie o arquivo."}), 400

    session_id = data.get("session_id", project_id)
    filename = data.get("filename", "video")
    _save_rel_path(project_id, data.get("rel_path"))

    tts_engine = data.get("tts_engine", "qwen")
    # Qwen speaks whole phrases far more naturally than 7-word fragments, so its
    # cues default longer; Edge keeps the short subtitle-sized defaults.
    long_cues = tts_engine == "qwen"
    words_per_cue = int(data.get("words_per_cue", 12 if long_cues else 7))
    config_overrides = {
        "enable_vocal_separation": data.get("vocal_separation", True),
        "tts_engine": tts_engine,
        "TRANSLATION_ENGINE": data.get("translation_engine", "opus-mt"),
        "OLLAMA_MODEL": data.get("ollama_model", "llama3.2"),
        "EDGE_TTS_VOICE": data.get("edge_voice", "pt-BR-ThalitaMultilingualNeural"),
        "ENHANCE_REFERENCE_AUDIO": data.get("enhance_reference_audio", True),
        "BACKGROUND_VOLUME": float(data.get("bg_volume", 0.7)),
        "DUB_VOLUME": float(data.get("dub_volume", 1.0)),
        "MAX_SPEED_RATIO": float(data.get("max_speed", 1.5)),
        "STRETCH_METHOD": data.get("stretch_method", "atempo"),
        "WORDS_PER_CUE": words_per_cue,
        "MAX_WORDS_PER_CUE": max(int(data.get("max_words_per_cue", 18 if long_cues else 10)), words_per_cue),
        "MAX_CHARS_PER_CUE": int(data.get("max_chars_per_cue", 100 if long_cues else 45)),
        "MIN_CUE_DURATION": float(data.get("min_cue_duration", 1.0)),
        "MIN_CUE_GAP": float(data.get("min_cue_gap", 0.08)),
        "MAX_CUE_DURATION": float(data.get("max_cue_duration", 10.0)),
    }
    if data.get("reference_audio"):
        config_overrides["reference_audio_path"] = data["reference_audio"]
    voice_id = data.get("voice_id")
    if voice_id and tts_engine == "qwen":
        meta = _voice_meta(VOICES_DIR / voice_id)
        if meta is None:
            return jsonify({"error": "Voz escolhida nao encontrada — escolha outra."}), 400
        config_overrides["reference_audio_path"] = str(VOICES_DIR / voice_id / "voice.wav")
        config_overrides["reference_text"] = meta.get("text", "")
        config_overrides["reference_language"] = meta.get("language", "")
    try:
        with open(CONFIG.PROJECTS_DIR / project_id / "job.json", "w", encoding="utf-8") as f:
            json.dump(config_overrides, f, ensure_ascii=False, indent=2)
    except OSError as e:
        logger.warning(f"[START] Couldn't save job settings for {project_id}: {e}")

    # Set up SSE queue
    q = Queue(maxsize=500)
    sse_queues[session_id] = q

    position = job_queue.enqueue(session_id, project_id, filename, config_overrides)

    result = {"status": "started", "project_id": project_id}
    if position > 0:
        result["queue_position"] = position
        sse_send(session_id, "queue", {"position": position, "message": f"Aguardando na fila - posicao {position}"})

    return jsonify(result)


@app.route("/api/queue/<session_id>")
def queue_status(session_id):
    return jsonify(job_queue.get_position(session_id))


@app.route("/api/events/<session_id>")
def sse_stream(session_id):
    """SSE endpoint for real-time progress."""
    # Don't overwrite the queue if start_pipeline already created one
    if session_id not in sse_queues:
        if job_queue.get_position(session_id)["status"] == "unknown":
            # Job already finished (or never existed): tell the page to stop
            # listening instead of holding a stream open on pings forever.
            return Response("event: gone\ndata: {}\n\n", mimetype="text/event-stream")
        sse_queues[session_id] = Queue(maxsize=500)

    q = sse_queues[session_id]

    def generate():
        while True:
            try:
                event, data = q.get(timeout=25)
            except Empty:
                if session_id not in sse_queues:
                    break  # job already finished and nothing left to send
                yield f"event: ping\ndata: {{}}\n\n"
                continue
            # "error" is reserved by EventSource for connection failures, so job
            # failures go out as "job_error".
            name = "job_error" if event == "error" else event
            yield f"event: {name}\ndata: {json.dumps(data)}\n\n"
            if event in TERMINAL_EVENTS:
                break

    return Response(generate(), mimetype="text/event-stream", headers={
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
    })


@app.route("/api/result/<project_id>")
def get_result(project_id):
    """Get pipeline results and file paths."""
    project_dir = CONFIG.PROJECTS_DIR / project_id
    if not project_dir.exists():
        return jsonify({"error": "Project not found"}), 404

    work_dir = project_dir / "work"
    output_dir = project_dir / "output"

    result = {"project_id": project_id}

    # Check which files exist — return URLs instead of absolute paths
    # Find the output video (name varies: *_dublado_pt.mp4)
    video_file = None
    if output_dir.exists():
        for f in output_dir.iterdir():
            if f.is_file() and f.suffix.lower() == ".mp4" and "_dublado_pt" in f.name:
                video_file = f
                break
        if not video_file:
            # Fallback to old naming
            fallback = output_dir / "dubbed_video.mp4"
            if fallback.exists():
                video_file = fallback

    file_map = {
        "vocals": work_dir / "vocals.wav",
        "background": work_dir / "background.wav",
        "dubbed_vocals": work_dir / "dubbed_vocals.wav",
        "final_audio": work_dir / "final_audio.wav",
        "en_srt": work_dir / "en.srt",
        "pt_srt": work_dir / "pt.srt",
        "video": video_file,
        "pt_vtt": output_dir / "pt.vtt",
        "en_vtt": output_dir / "en.vtt",
    }

    result["files"] = {}
    for name, path in file_map.items():
        if path and path.exists():
            # Use relative URL instead of absolute filesystem path
            # Determine subdir (work/ or output/)
            if path.parent.name == "output":
                result["files"][name] = f"/api/file/{project_id}/output/{quote(path.name)}"
            elif path.parent.name == "work":
                result["files"][name] = f"/api/file/{project_id}/work/{quote(path.name)}"
            else:
                result["files"][name] = None
        else:
            result["files"][name] = None

    # Read SRT content
    for srt_name in ("en_srt", "pt_srt"):
        path = file_map[srt_name]
        if path and path.exists():
            with open(path, "r", encoding="utf-8") as f:
                result[srt_name + "_content"] = f.read()
        else:
            result[srt_name + "_content"] = ""

    # Pipeline state
    state_file = project_dir / "state.json"
    if state_file.exists():
        with open(state_file, "r", encoding="utf-8") as f:
            result["state"] = json.load(f)
    else:
        result["state"] = None

    return jsonify(result)


@app.route("/api/file/<project_id>/<path:filepath>")
def serve_file(project_id, filepath):
    """Serve a file from a project directory (output/ or work/)."""
    if ".." in filepath or filepath.startswith("/"):
        return jsonify({"error": "Invalid path"}), 400

    project_dir = CONFIG.PROJECTS_DIR / project_id
    full_path = project_dir / filepath
    if not full_path.exists() or not full_path.is_file():
        return jsonify({"error": "File not found"}), 404

    # Set correct MIME type for VTT subtitles
    mime_map = {
        ".vtt": "text/vtt",
        ".srt": "text/plain",
        ".mp4": "video/mp4",
        ".wav": "audio/wav",
        ".mkv": "video/x-matroska",
        ".avi": "video/x-msvideo",
        ".mov": "video/quicktime",
        ".webm": "video/webm",
    }
    ext = Path(filepath).suffix.lower()
    mimetype = mime_map.get(ext)
    if request.args.get("download"):
        # Some browsers ignore <a download> and just open the video in a tab;
        # Content-Disposition makes every browser save it, with its real name.
        return send_file(str(full_path), mimetype=mimetype, as_attachment=True, download_name=full_path.name)
    return send_file(str(full_path), mimetype=mimetype)


# ── Saved reference voices ───────────────────────────────────────────
# Each voice lives in data/voices/<id>/ as voice.wav (24kHz mono, <=20s) plus
# meta.json with its transcript — Qwen clones much better knowing what was said.

VOICES_DIR = CONFIG.BASE_DIR / "data" / "voices"
VOICE_MAX_SEC = 20
_voice_whisper = None
_voice_whisper_lock = Lock()


def _voice_meta(voice_dir: Path) -> dict | None:
    try:
        with open(voice_dir / "meta.json", "r", encoding="utf-8") as f:
            meta = json.load(f)
        meta["id"] = voice_dir.name
        meta["audio_url"] = f"/api/voices/{voice_dir.name}/audio"
        return meta
    except Exception:
        return None


def _transcribe_voice(wav_path: Path) -> tuple[str, str]:
    """Transcribe a short clip on CPU so it never competes with a running job for VRAM."""
    global _voice_whisper
    with _voice_whisper_lock:
        if _voice_whisper is None:
            from faster_whisper import WhisperModel
            _voice_whisper = WhisperModel(
                CONFIG.WHISPER_MODEL, device="cpu", compute_type="int8",
                download_root=str(CONFIG.MODEL_CACHE_DIR / "whisper"),
            )
        segments, info = _voice_whisper.transcribe(str(wav_path), beam_size=5, vad_filter=True)
        text = " ".join(seg.text.strip() for seg in segments).strip()
    return text, info.language


@app.route("/api/voices")
def list_voices():
    voices = []
    if VOICES_DIR.exists():
        for d in VOICES_DIR.iterdir():
            meta = _voice_meta(d) if d.is_dir() else None
            if meta:
                voices.append(meta)
    voices.sort(key=lambda v: v.get("created", 0), reverse=True)
    return jsonify({"voices": voices})


@app.route("/api/voices", methods=["POST"])
def add_voice():
    import shutil
    import subprocess
    f = request.files.get("audio")
    if not f:
        return jsonify({"error": "Nenhum audio enviado"}), 400
    name = (request.form.get("name") or "").strip()[:40] or "Minha voz"
    voice_id = uuid.uuid4().hex[:10]
    voice_dir = VOICES_DIR / voice_id
    voice_dir.mkdir(parents=True, exist_ok=True)
    raw = voice_dir / ("upload" + (Path(f.filename or "").suffix or ".webm"))
    f.save(str(raw))
    wav = voice_dir / "voice.wav"
    try:
        # Trim leading silence, cap the length, even out the volume.
        subprocess.run([
            "ffmpeg", "-v", "error", "-y", "-i", str(raw),
            "-af", "silenceremove=start_periods=1:start_threshold=-45dB,loudnorm=I=-18",
            "-t", str(VOICE_MAX_SEC), "-ar", "24000", "-ac", "1", str(wav),
        ], check=True, capture_output=True, timeout=120)
        raw.unlink(missing_ok=True)
        import soundfile as _sf
        duration = _sf.info(str(wav)).duration
        if duration < 3:
            raise ValueError("Audio muito curto — grave pelo menos 5 segundos de fala")
        text, lang = _transcribe_voice(wav)
        if not text:
            raise ValueError("Nao deu para entender a fala no audio")
    except Exception as e:
        shutil.rmtree(str(voice_dir), ignore_errors=True)
        msg = str(e) if isinstance(e, ValueError) else f"Falha ao processar o audio: {e}"
        return jsonify({"error": msg}), 400

    meta = {"name": name, "text": text, "language": lang,
            "duration": round(duration, 1), "created": time.time()}
    with open(voice_dir / "meta.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)
    logger.info(f"[VOICES] Saved voice '{name}' ({duration:.1f}s, {lang}): {text[:60]}")
    return jsonify(_voice_meta(voice_dir))


@app.route("/api/voices/<voice_id>/audio")
def voice_audio(voice_id):
    wav = VOICES_DIR / voice_id / "voice.wav"
    if "/" in voice_id or ".." in voice_id or not wav.exists():
        return jsonify({"error": "Not found"}), 404
    return send_file(str(wav), mimetype="audio/wav")


@app.route("/api/voices/<voice_id>", methods=["DELETE"])
def delete_voice(voice_id):
    import shutil
    voice_dir = VOICES_DIR / voice_id
    if "/" in voice_id or ".." in voice_id or not voice_dir.is_dir():
        return jsonify({"error": "Not found"}), 404
    shutil.rmtree(str(voice_dir), ignore_errors=True)
    return jsonify({"status": "deleted"})


@app.route("/api/config", methods=["GET"])
def get_config():
    """Return current configuration defaults."""
    return jsonify({
        "words_per_cue": CONFIG.WORDS_PER_CUE,
        "max_words_per_cue": CONFIG.MAX_WORDS_PER_CUE,
        "max_chars_per_cue": CONFIG.MAX_CHARS_PER_CUE,
        "min_cue_duration": CONFIG.MIN_CUE_DURATION,
        "max_cue_duration": CONFIG.MAX_CUE_DURATION,
        "max_speed_ratio": CONFIG.MAX_SPEED_RATIO,
        "min_speed_ratio": CONFIG.MIN_SPEED_RATIO,
        "stretch_method": CONFIG.STRETCH_METHOD,
        "background_volume": CONFIG.BACKGROUND_VOLUME,
        "dub_volume": CONFIG.DUB_VOLUME,
        "tts_engine": CONFIG.tts_engine,
        "edge_voice": CONFIG.EDGE_TTS_VOICE,
        "vocal_separation": CONFIG.enable_vocal_separation,
    })


if __name__ == "__main__":
    # DUB_PORT: uma porta por placa quando o pod tem várias (runpod/iniciar.sh).
    PORT = int(os.environ.get("DUB_PORT", "5000"))
    print("\n  +======================================+")
    print("  |     LuminaDub AI - Server v2.1       |")
    print(f"  |     http://localhost:{PORT:<16}|")
    print("  +======================================+\n")
    try:
        if _sys.platform == "win32":
            raise ImportError("gunicorn requires fcntl, unavailable on Windows")
        import gunicorn.app.base

        class _StandaloneApp(gunicorn.app.base.BaseApplication):
            def __init__(self, application, options=None):
                self.options = options or {}
                self.application = application
                super().__init__()
            def load_config(self):
                for key, val in self.options.items():
                    self.cfg.set(key.lower(), val)
            def load(self):
                return self.application

        print("  Using gunicorn + gevent (async SSE — no thread exhaustion)")
        _StandaloneApp(app, {
            "bind": f"0.0.0.0:{PORT}",
            "worker_class": "gevent",
            "workers": 1,
            "worker_connections": 1000,
            "timeout": 120,
            "keepalive": 5,
        }).run()
    except ImportError:
        try:
            from waitress import serve as waitress_serve
            print("  Using waitress (install gunicorn+gevent for better SSE scalability)")
            # Each open /api/events/<sid> SSE stream blocks a waitress thread for as
            # long as that video sits in queue + processes — with a batch of 50
            # videos that's up to 50 threads tied up on top of normal upload/poll
            # traffic. 16 was nowhere near enough and stalled everything once the
            # pool filled up; these threads are I/O-bound (blocked on a queue.get),
            # not CPU-bound, so a much larger pool is cheap.
            waitress_serve(app, host="0.0.0.0", port=PORT, threads=128, max_request_body_size=10*1024*1024*1024)
        except ImportError:
            print("  Using Flask dev server (install waitress or gunicorn+gevent)")
            app.run(host="0.0.0.0", port=PORT, debug=False, threaded=True)