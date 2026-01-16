from __future__ import annotations

import base64
import json
import logging
import os
import shlex
import signal
import subprocess
import time
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Union

import requests
from openai import OpenAI

VLLM_HOST = os.environ.get("VLLM_HOST", "127.0.0.1")
VLLM_PORT = int(os.environ.get("VLLM_PORT", "11301"))
DEFAULT_VLLM_BASE_URL = f"http://{VLLM_HOST}:{VLLM_PORT}/v1"

VLLM_STARTUP_TIMEOUT = int(os.environ.get("VLLM_STARTUP_TIMEOUT", "300"))
VLLM_EXTRA_ARGS = shlex.split(os.environ.get("VLLM_EXTRA_ARGS", ""))

# KV cache + memory safety defaults (kept for compatibility; vLLM args may use env elsewhere)
VLLM_MAX_MODEL_LEN = int(os.environ.get("VLLM_MAX_MODEL_LEN", "70000"))
VLLM_GPU_MEMORY_UTILIZATION = float(os.environ.get("VLLM_GPU_MEMORY_UTILIZATION", "0.90"))

DEFAULT_TIMEOUT = float(os.environ.get("VLLM_TIMEOUT", "60"))
DEFAULT_MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "4096"))

VIDEO_USE_DATA_URL = os.environ.get("VIDEO_USE_DATA_URL", "1").strip() not in ("0", "false", "False")

# ----------------------------
# Model identifiers (HF repos only)
# ----------------------------

QWEN_32B_REPO = os.environ.get("QWEN_32B_REPO", "Qwen/Qwen3-VL-30B-A3B-Thinking-FP8")
QWEN_8B_REPO = os.environ.get("QWEN_8B_REPO", "Qwen/Qwen3-VL-8B-Thinking")
QWEN_2B_REPO = os.environ.get("QWEN_2B_REPO", "Qwen/Qwen3-VL-2B-Thinking")

COSMOS_REASON1_REPO = os.environ.get("COSMOS_REASON1_REPO", "nvidia/Cosmos-Reason1-7B")
COSMOS_REASON2_2B_REPO = os.environ.get("COSMOS_REASON2_2B_REPO", "nvidia/Cosmos-Reason2-2B")
COSMOS_REASON2_8B_REPO = os.environ.get("COSMOS_REASON2_8B_REPO", "nvidia/Cosmos-Reason2-8B")

MODEL_CHOICES = ("qwen-2B", "qwen-8B", "qwen-32B", "cosmos1", "cosmos2-2B", "cosmos2-8B", "all")
DEFAULT_MODEL_SELECTION = os.environ.get("DEFAULT_MODEL", "cosmos2-2B")

JSON_MODE_RESPONSE_FORMAT: Dict[str, object] = {"type": "json_object"}

_VLLM_SERVER_MANAGER: Optional["VLLMServerManager"] = None


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
    ) -> None:
        self.model_key = model_key
        self.host = host
        self.port = port
        self.model_repo = model_repo
        self.served_model_name = served_model_name
        self.timeout_s = timeout_s
        self.env = env
        self.extra_args = extra_args
        self.health_url = f"http://{host}:{port}/v1/models"
        self._proc: Optional[subprocess.Popen] = None

    def start(self) -> None:
        if self._proc and self._proc.poll() is None:
            return

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
        ]
        # if self.extra_args:
        #     cmd.extend(self.extra_args)

        logging.info(
            "[vLLM] Starting server for %s as '%s' using repo '%s' on %s:%s",
            self.model_key,
            self.served_model_name,
            self.model_repo,
            self.host,
            self.port,
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

def _build_vllm_env() -> Dict[str, str]:
    env = dict(os.environ)
    env["HF_HOME"] = os.environ.get("HF_HOME", "/mnt/ssd1/hf")
    env["HF_HUB_CACHE"] = os.environ.get("HF_HUB_CACHE", "/mnt/ssd1/hf/hub")
    env["TRANSFORMERS_CACHE"] = os.environ.get("TRANSFORMERS_CACHE", "/mnt/ssd1/hf/transformers")
    return env


def _served_name_for(model_key: str) -> str:
    if model_key == "qwen-32B":
        return os.environ.get("QWEN_VLLM_MODEL_NAME_32B", "Qwen3-VL-32B-Thinking")
    if model_key == "qwen-8B":
        return os.environ.get("QWEN_VLLM_MODEL_NAME_8B", "Qwen3-VL-8B-Thinking")
    if model_key == "qwen-2B":
        return os.environ.get("QWEN_VLLM_MODEL_NAME_2B", "Qwen3-VL-2B-Thinking")
    if model_key == "cosmos1":
        return "Cosmos-Reason1"
    if model_key == "cosmos2-2B":
        return "Cosmos-Reason2-2B"
    if model_key == "cosmos2-8B":
        return "Cosmos-Reason2-8B"
    raise ValueError(f"Unknown model_key: {model_key}")


def _resolve_model_repo(model_key: str) -> str:
    if model_key == "qwen-32B":
        return QWEN_32B_REPO
    if model_key == "qwen-8B":
        return QWEN_8B_REPO
    if model_key == "qwen-2B":
        return QWEN_2B_REPO
    if model_key == "cosmos1":
        return COSMOS_REASON1_REPO
    if model_key == "cosmos2-2B":
        return COSMOS_REASON2_2B_REPO
    if model_key == "cosmos2-8B":
        return COSMOS_REASON2_8B_REPO
    raise ValueError(f"Unknown model_key: {model_key}")


def _ensure_vllm_server(model_key: str) -> VLLMServerManager:
    global _VLLM_SERVER_MANAGER

    if _VLLM_SERVER_MANAGER and _VLLM_SERVER_MANAGER.model_key != model_key:
        _VLLM_SERVER_MANAGER.stop()
        _VLLM_SERVER_MANAGER = None

    if _VLLM_SERVER_MANAGER is None:
        env = _build_vllm_env()
        model_repo = _resolve_model_repo(model_key)
        served_model_name = _served_name_for(model_key)
        manager = VLLMServerManager(
            model_key=model_key,
            host=VLLM_HOST,
            port=VLLM_PORT,
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
    VLM client via OpenAI-compatible endpoint. The vLLM server is managed locally.
    Video-only: uses `video_url` with a data URL (base64).
    """

    def __init__(self, model_key: str, base_url: str = DEFAULT_VLLM_BASE_URL, timeout: float = DEFAULT_TIMEOUT) -> None:
        self.model_key = model_key
        manager = _ensure_vllm_server(model_key)
        self.model_name = manager.served_model_name
        api_key = os.environ.get("OPENAI_API_KEY", "EMPTY")
        self.client = OpenAI(api_key=api_key, base_url=base_url.rstrip("/"), timeout=timeout)
        logging.info("[Client] Using vLLM at %s with model='%s' (key=%s)", base_url.rstrip("/"), self.model_name, model_key)

    def _request(
        self,
        video_path: Path,
        prompt: str,
        *,
        force_json_mode: bool,
        extra_system: Optional[str] = None,
    ) -> Optional[InferenceResult]:
        if VIDEO_USE_DATA_URL:
            video_ref = _file_to_data_url(video_path)
        else:
            raise RuntimeError("VIDEO_USE_DATA_URL=0 requires providing an HTTP URL mapping for the video.")

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


VLMClientType = Union[VLLMClient]


def shutdown_client(client: Optional[VLMClientType]) -> None:
    if client is None:
        return
    close_fn = getattr(client, "close", None)
    if callable(close_fn):
        close_fn()


def ensure_clients(model_keys: List[str]) -> OrderedDict[str, Optional[VLMClientType]]:
    clients: OrderedDict[str, Optional[VLMClientType]] = OrderedDict()
    for model_key in model_keys:
        if model_key in clients:
            continue
        try:
            clients[model_key] = VLLMClient(model_key=model_key)
        except Exception as exc:
            logging.error("Unable to initialize %s: %s", model_key, exc)
            clients[model_key] = None
    return clients
