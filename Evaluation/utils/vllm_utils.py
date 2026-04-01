from __future__ import annotations

import base64
import json
import logging
import os
import shlex
import signal
import socket
import subprocess
import tempfile
import threading
import time
from urllib.parse import urlparse
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Union

import requests
from openai import OpenAI

from .models_utils import (
    DEFAULT_MODEL_SELECTION,
    MODEL_CHOICES,
    resolve_model_repo,
    served_name_for,
)

VLLM_HOST = os.environ.get("VLLM_HOST", "127.0.0.1")
VLLM_PORT = int(os.environ.get("VLLM_PORT", "0"))
DEFAULT_VLLM_BASE_URL = f"http://{VLLM_HOST}:{VLLM_PORT}/v1"

VLLM_STARTUP_TIMEOUT = int(os.environ.get("VLLM_STARTUP_TIMEOUT", "900"))
VLLM_EXTRA_ARGS = shlex.split(os.environ.get("VLLM_EXTRA_ARGS", ""))
VLLM_TENSOR_PARALLEL_SIZE = int(os.environ.get("VLLM_TENSOR_PARALLEL_SIZE", "0"))
VLLM_PIPELINE_PARALLEL_SIZE = int(os.environ.get("VLLM_PIPELINE_PARALLEL_SIZE", "0"))

# KV cache + memory safety defaults (kept for compatibility; vLLM args may use env elsewhere)
VLLM_MAX_MODEL_LEN = int(os.environ.get("VLLM_MAX_MODEL_LEN", "16000"))
VLLM_GPU_MEMORY_UTILIZATION = float(os.environ.get("VLLM_GPU_MEMORY_UTILIZATION", "0.97"))

DEFAULT_TIMEOUT = float(os.environ.get("VLLM_TIMEOUT", "3600"))
DEFAULT_MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "4096"))

VIDEO_USE_DATA_URL = os.environ.get("VIDEO_USE_DATA_URL", "1").strip() not in ("0", "false", "False")
VIDEO_URL_PREFIX = os.environ.get("VIDEO_URL_PREFIX")
VIDEO_URL_ROOT = os.environ.get("VIDEO_URL_ROOT")
VIDEO_TARGET_BITRATE_MBPS = float(os.environ.get("VIDEO_TARGET_BITRATE_MBPS", "40"))
VIDEO_REENCODE_CACHE_DIR = os.environ.get(
    "VIDEO_REENCODE_CACHE_DIR",
    str(Path(tempfile.gettempdir()) / "vlm_reencoded"),
)


JSON_MODE_RESPONSE_FORMAT: Dict[str, object] = {"type": "json_object"}

_VLLM_SERVER_MANAGER: Optional["VLLMServerManager"] = None


def _count_available_gpus(env: Dict[str, str]) -> int:
    cvd = env.get("CUDA_VISIBLE_DEVICES")
    if cvd is not None:
        raw = cvd.strip()
        if raw in ("", "-1", "none", "None"):
            return 0
        return len([x for x in raw.split(",") if x.strip()])

    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            text=True,
            timeout=5,
        )
        return len([line for line in output.splitlines() if line.strip()])
    except Exception:
        return 0


def _choose_vllm_port(host: str) -> int:
    if VLLM_PORT > 0:
        return VLLM_PORT
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def _normalize_vllm_base_url(url: str) -> str:
    normalized = url.rstrip("/")
    if normalized.endswith("/v1"):
        return normalized
    return f"{normalized}/v1"


def _parse_host_port(url: str) -> tuple[str, int]:
    parsed = urlparse(url)
    if not parsed.hostname:
        raise ValueError(f"Invalid vLLM base URL: {url}")
    port = parsed.port
    if port is None:
        if parsed.scheme == "https":
            port = 443
        else:
            port = 80
    return parsed.hostname, port


def wait_ready(url: str, timeout_s: int = 120) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            response = requests.get(url, timeout=1.0)
            if response.status_code == 200:
                return
        except Exception:
            pass
        time.sleep(0.5)
    raise RuntimeError("vLLM server did not become ready in time")


class VLLMServerManager:
    def __init__(
        self,
        model_key: str,
        host: str,
        port: int,
        model_repo: str,
        served_model_name: str,
        timeout_s: int,
        env: Dict[str, str],
        extra_args: List[str],
        base_url: Optional[str] = None,
    ) -> None:
        self.model_key = model_key
        self.host = host
        self.port = port
        self.model_repo = model_repo
        self.served_model_name = served_model_name
        self.timeout_s = timeout_s
        self.env = env
        self.extra_args = extra_args
        self.base_url = base_url.rstrip("/") if base_url else None
        resolved_base = self.base_url or f"http://{host}:{port}/v1"
        self.health_url = f"{resolved_base}/models"
        self._proc: Optional[subprocess.Popen] = None

    def start(self) -> None:
        if self.base_url:
            wait_ready(self.health_url, timeout_s=self.timeout_s)
            return
        if self._proc and self._proc.poll() is None:
            return

        num_gpus = _count_available_gpus(self.env)
        if num_gpus < 1:
            raise RuntimeError("No GPUs available for vLLM (CUDA_VISIBLE_DEVICES may be empty or invalid).")

        tensor_parallel_size = VLLM_TENSOR_PARALLEL_SIZE if VLLM_TENSOR_PARALLEL_SIZE > 0 else num_gpus
        if tensor_parallel_size < 1:
            raise RuntimeError("VLLM_TENSOR_PARALLEL_SIZE must be >= 1 when set.")
        if tensor_parallel_size > num_gpus:
            raise RuntimeError(
                f"Requested tensor-parallel size {tensor_parallel_size} exceeds visible GPUs {num_gpus}. "
                "Expose more GPUs via CUDA_VISIBLE_DEVICES or lower VLLM_TENSOR_PARALLEL_SIZE."
            )

        cmd = [
            "vllm",
            "serve",
            self.model_repo,
            "--served-model-name",
            self.served_model_name,
            "--host",
            self.host,
            "--port",
            str(self.port),
            "--max-model-len", 
            str(VLLM_MAX_MODEL_LEN),
            "--gpu-memory-utilization", 
            str(VLLM_GPU_MEMORY_UTILIZATION),
            "--tensor-parallel-size",
            str(tensor_parallel_size),
        ]
        if VLLM_PIPELINE_PARALLEL_SIZE > 0:
            cmd.extend(["--pipeline-parallel-size", str(VLLM_PIPELINE_PARALLEL_SIZE)])
        if self.extra_args:
            cmd.extend(self.extra_args)

        logging.info(
            "[vLLM] Starting server for %s as '%s' using repo '%s' on %s:%s (visible_gpus=%d, tensor_parallel=%d)",
            self.model_key,
            self.served_model_name,
            self.model_repo,
            self.host,
            self.port,
            num_gpus,
            tensor_parallel_size,
        )

        # Start in a new session (new process group) so we can signal the whole tree.
        proc = subprocess.Popen(
            cmd,
            env=self.env,
            start_new_session=True,
        )

        try:
            wait_ready(self.health_url, timeout_s=self.timeout_s)
        except Exception:
            self._terminate_proc(proc)
            raise

        self._proc = proc
        logging.info("[vLLM] Server ready at %s", self.health_url)

    def stop(self) -> None:
        if not self._proc:
            return
        self._terminate_proc(self._proc)
        self._proc = None

    def _terminate_proc(self, proc: subprocess.Popen) -> None:
        if proc.poll() is not None:
            return

        # Prefer signalling the whole process group (vLLM spawns workers).
        try:
            pgid = os.getpgid(proc.pid)
        except Exception:
            pgid = None

        def signal_tree(sig: int) -> None:
            if pgid is not None:
                try:
                    os.killpg(pgid, sig)
                    return
                except ProcessLookupError:
                    return
            try:
                proc.send_signal(sig)
            except ProcessLookupError:
                return

        # 1) Graceful: SIGINT (lets vLLM run shutdown handlers)
        signal_tree(signal.SIGINT)
        try:
            proc.wait(timeout=120)
            return
        except subprocess.TimeoutExpired:
            pass

        # 2) Less graceful: SIGTERM
        signal_tree(signal.SIGTERM)
        try:
            proc.wait(timeout=30)
            return
        except subprocess.TimeoutExpired:
            pass

        # 3) Last resort: SIGKILL
        signal_tree(signal.SIGKILL)
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            # If this happens, something is seriously wrong at OS level; give up.
            pass

def _build_vllm_env(cuda_visible_devices: Optional[str] = None) -> Dict[str, str]:
    env = dict(os.environ)
    if cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    tok = env.get("HF_TOKEN") or env.get("HUGGING_FACE_HUB_TOKEN")
    if tok:
        env["HF_TOKEN"] = tok
        env["HUGGING_FACE_HUB_TOKEN"] = tok
    return env


def ensure_vllm_server(model_key: str, cuda_visible_devices: Optional[str] = None) -> VLLMServerManager:
    global _VLLM_SERVER_MANAGER

    requested_cuda_visible_devices = (
        cuda_visible_devices if cuda_visible_devices is not None else os.environ.get("CUDA_VISIBLE_DEVICES")
    )
    current_cuda_visible_devices = (
        _VLLM_SERVER_MANAGER.env.get("CUDA_VISIBLE_DEVICES") if _VLLM_SERVER_MANAGER else None
    )

    if _VLLM_SERVER_MANAGER and (
        _VLLM_SERVER_MANAGER.model_key != model_key
        or current_cuda_visible_devices != requested_cuda_visible_devices
    ):
        _VLLM_SERVER_MANAGER.stop()
        _VLLM_SERVER_MANAGER = None

    if _VLLM_SERVER_MANAGER is None:
        env = _build_vllm_env(cuda_visible_devices=cuda_visible_devices)
        port = _choose_vllm_port(VLLM_HOST)
        model_repo = resolve_model_repo(model_key)
        served_model_name = served_name_for(model_key)
        if model_repo.startswith(("http://", "https://")):
            base_url = _normalize_vllm_base_url(model_repo)
            host, port = _parse_host_port(base_url)
            manager = VLLMServerManager(
                model_key=model_key,
                host=host,
                port=port,
                model_repo=model_repo,
                served_model_name=served_model_name,
                timeout_s=VLLM_STARTUP_TIMEOUT,
                env=env,
                extra_args=VLLM_EXTRA_ARGS,
                base_url=base_url,
            )
            manager.start()
            _VLLM_SERVER_MANAGER = manager
            return _VLLM_SERVER_MANAGER
        manager = VLLMServerManager(
            model_key=model_key,
            host=VLLM_HOST,
            port=port,
            model_repo=model_repo,
            served_model_name=served_model_name,
            timeout_s=VLLM_STARTUP_TIMEOUT,
            env=env,
            extra_args=VLLM_EXTRA_ARGS,
        )
        manager.start()
        _VLLM_SERVER_MANAGER = manager
    else:
        _VLLM_SERVER_MANAGER.start()

    return _VLLM_SERVER_MANAGER


def shutdown_vllm_server() -> None:
    global _VLLM_SERVER_MANAGER
    if _VLLM_SERVER_MANAGER is None:
        return
    try:
        _VLLM_SERVER_MANAGER.stop()
    finally:
        _VLLM_SERVER_MANAGER = None


def _file_to_data_url(video_path: Path) -> str:
    suffix = video_path.suffix.lower().lstrip(".") or "mp4"
    mime = f"video/{suffix}"
    data = video_path.read_bytes()
    b64 = base64.b64encode(data).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def _file_to_http_url(video_path: Path) -> str:
    if not VIDEO_URL_PREFIX or not VIDEO_URL_ROOT:
        raise RuntimeError(
            "VIDEO_USE_DATA_URL=0 requires VIDEO_URL_PREFIX and VIDEO_URL_ROOT to be set."
        )
    root = Path(VIDEO_URL_ROOT).resolve()
    try:
        rel = video_path.resolve().relative_to(root)
    except Exception as exc:
        raise RuntimeError(
            f"Video path is outside VIDEO_URL_ROOT: {video_path} (root={root})"
        ) from exc
    rel_url = "/".join(rel.parts)
    return f"{VIDEO_URL_PREFIX.rstrip('/')}/{rel_url}"


def _probe_format_bitrate(video_path: Path) -> Optional[int]:
    try:
        out = subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=bit_rate",
                "-of",
                "default=nk=1:nw=1",
                str(video_path),
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None
    try:
        return int(out)
    except Exception:
        return None


def _maybe_reencode_video(video_path: Path) -> Path:
    if VIDEO_TARGET_BITRATE_MBPS <= 0:
        return video_path

    bitrate = _probe_format_bitrate(video_path)
    if bitrate is None:
        return video_path

    target_bps = int(VIDEO_TARGET_BITRATE_MBPS * 1_000_000)
    if bitrate <= target_bps:
        return video_path

    cache_dir = Path(VIDEO_REENCODE_CACHE_DIR)
    cache_dir.mkdir(parents=True, exist_ok=True)

    stat = video_path.stat()
    cache_name = f"{video_path.stem}_br{VIDEO_TARGET_BITRATE_MBPS:.0f}M_{stat.st_size}_{int(stat.st_mtime)}.mp4"
    out_path = cache_dir / cache_name
    if out_path.exists():
        return out_path

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-c:v",
        "libx264",
        "-b:v",
        f"{VIDEO_TARGET_BITRATE_MBPS:.0f}M",
        "-maxrate",
        f"{VIDEO_TARGET_BITRATE_MBPS:.0f}M",
        "-bufsize",
        f"{int(VIDEO_TARGET_BITRATE_MBPS * 2):.0f}M",
        "-an",
        "-movflags",
        "+faststart",
        str(out_path),
    ]
    try:
        subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return out_path
    except Exception as exc:
        logging.warning("Re-encode failed for %s: %s", video_path, exc)
        return video_path


def _extract_first_json_object(text: str) -> Optional[str]:
    if not text:
        return None
    s = text.strip()
    if s.startswith("{") and s.endswith("}"):
        return s
    start = s.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(s)):
        ch = s[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return s[start : i + 1]
    return None


def _is_valid_json_object(text: str) -> bool:
    try:
        obj = json.loads(text)
        return isinstance(obj, dict)
    except Exception:
        return False


class InferenceResult(NamedTuple):
    response_text: str
    elapsed_s: float
    usage: Optional[Dict[str, int]]


class VLLMClient:
    """
    VLM client via OpenAI-compatible endpoint. Server lifecycle is managed externally.
    Video-only: uses `video_url` with a data URL (base64).
    """

    def __init__(
        self,
        model_key: str,
        model_name: str,
        base_url: str,
        timeout: float = DEFAULT_TIMEOUT,
        api_key: Optional[str] = None,
    ) -> None:
        self.model_key = model_key
        self.model_name = model_name
        resolved_base_url = base_url.rstrip("/")
        resolved_api_key = api_key or os.environ.get("OPENAI_API_KEY", "EMPTY")
        self.client = OpenAI(api_key=resolved_api_key, base_url=resolved_base_url, timeout=timeout)
        logging.info("[Client] Using vLLM at %s with model='%s' (key=%s)", resolved_base_url, self.model_name, model_key)

    def _request(
        self,
        video_path: Path,
        prompt: str,
        *,
        force_json_mode: bool,
        extra_system: Optional[str] = None,
    ) -> Optional[InferenceResult]:
        # safe_path = _maybe_reencode_video(video_path)
        safe_path=video_path
        if VIDEO_USE_DATA_URL:
            video_ref = _file_to_data_url(safe_path)
        else:
            video_ref = _file_to_http_url(safe_path)

        start = time.time()

        messages: List[Dict[str, object]] = []
        if extra_system:
            messages.append({"role": "system", "content": extra_system})
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "video_url", "video_url": {"url": video_ref}},
                ],
            }
        )

        request_kwargs: Dict[str, object] = {
            "model": self.model_name,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": DEFAULT_MAX_NEW_TOKENS,
        }

        if force_json_mode:
            request_kwargs["response_format"] = JSON_MODE_RESPONSE_FORMAT

        try:
            response = self.client.chat.completions.create(**request_kwargs)
        except Exception as exc:
            logging.warning("[%s] Request failed: %s", self.model_key, exc)
            return None

        choices = getattr(response, "choices", None)
        if not choices:
            return None

        msg = choices[0].message
        content = getattr(msg, "content", None)
        elapsed_s = time.time() - start

        usage = getattr(response, "usage", None)
        usage_dict = None
        if usage is not None:
            usage_dict = {
                "prompt_token_count": getattr(usage, "prompt_tokens", None),
                "candidates_token_count": getattr(usage, "completion_tokens", None),
                "total_token_count": getattr(usage, "total_tokens", None),
            }

        response_text = content.strip() if isinstance(content, str) else ""
        return InferenceResult(response_text, elapsed_s, usage_dict)

    def run_video_inference_json(
        self,
        video_path: Path,
        prompt: str,
    ) -> Optional[InferenceResult]:
        """
        Force JSON mode; extract first JSON object if needed; retry once with a stricter system message.
        """
        result = self._request(video_path, prompt, force_json_mode=True)
        if result is None:
            return None

        text = result.response_text
        extracted = _extract_first_json_object(text) or text
        if _is_valid_json_object(extracted):
            if extracted != text:
                return InferenceResult(extracted, result.elapsed_s, result.usage)
            return result

        retry = self._request(
            video_path,
            prompt,
            force_json_mode=True,
            extra_system="Return only a single valid JSON object. No markdown, no explanations, no surrounding text.",
        )
        if retry is None:
            return result

        retry_text = retry.response_text
        retry_extracted = _extract_first_json_object(retry_text) or retry_text
        if _is_valid_json_object(retry_extracted):
            if retry_extracted != retry_text:
                return InferenceResult(retry_extracted, retry.elapsed_s, retry.usage)
            return retry

        return retry

    def close(self) -> None:
        pass


class VLLMClientFactory:
    """
    Lightweight per-thread HTTP client factory.
    Assumes the vLLM server is already running.
    """

    def __init__(
        self,
        model_key: str,
        model_name: str,
        base_url: str,
        timeout: float = DEFAULT_TIMEOUT,
        api_key: Optional[str] = None,
    ) -> None:
        self.model_key = model_key
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "EMPTY")
        self._local = threading.local()

    @classmethod
    def from_server(
        cls,
        manager: VLLMServerManager,
        model_key: str,
        base_url: Optional[str] = None,
        timeout: float = DEFAULT_TIMEOUT,
        api_key: Optional[str] = None,
    ) -> "VLLMClientFactory":
        resolved_base_url = (
            base_url.rstrip("/")
            if base_url
            else (manager.base_url or f"http://{manager.host}:{manager.port}/v1")
        )
        return cls(
            model_key=model_key,
            model_name=manager.served_model_name,
            base_url=resolved_base_url,
            timeout=timeout,
            api_key=api_key,
        )

    def get_client(self) -> VLLMClient:
        client = getattr(self._local, "client", None)
        if client is None:
            client = VLLMClient(
                model_key=self.model_key,
                model_name=self.model_name,
                base_url=self.base_url,
                timeout=self.timeout,
                api_key=self.api_key,
            )
            self._local.client = client
        return client

    def close(self) -> None:
        client = getattr(self._local, "client", None)
        if client is not None:
            client.close()
            self._local.client = None


VLMClientType = Union[VLLMClient]


def shutdown_client(client: Optional[VLMClientType]) -> None:
    if client is None:
        return
    close_fn = getattr(client, "close", None)
    if callable(close_fn):
        close_fn()


def ensure_clients(
    model_keys: List[str],
    cuda_visible_devices: Optional[str] = None,
) -> OrderedDict[str, Optional[VLMClientType]]:
    clients: OrderedDict[str, Optional[VLMClientType]] = OrderedDict()
    for model_key in model_keys:
        if model_key in clients:
            continue
        try:
            manager = ensure_vllm_server(model_key, cuda_visible_devices=cuda_visible_devices)
            base_url = manager.base_url or f"http://{manager.host}:{manager.port}/v1"
            clients[model_key] = VLLMClient(
                model_key=model_key,
                model_name=manager.served_model_name,
                base_url=base_url,
            )
        except Exception as exc:
            logging.error("Unable to initialize %s: %s", model_key, exc)
            clients[model_key] = None
    return clients
