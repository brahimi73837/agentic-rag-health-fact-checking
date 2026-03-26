import json
import re
import time
import logging
from typing import Any, Dict, Optional
import requests

from src.config import OLLAMA_HOST, LLM_MODEL

log = logging.getLogger(__name__)

_JSON_BLOCK = re.compile(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", re.DOTALL)

class OllamaClient:
    def __init__(self, model: str = LLM_MODEL, host: str = OLLAMA_HOST, timeout: int = 180):
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = timeout

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
        format_json: bool = False,
    ) -> str:
        body = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "num_ctx": 3072,
            },
        }
        if system:
            body["system"] = system
        if format_json:
            body["format"] = "json"
        r = requests.post(f"{self.host}/api/generate", json=body, timeout=self.timeout)
        r.raise_for_status()
        return r.json().get("response", "").strip()

    def json_call(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
        max_retries: int = 2,
    ) -> Optional[Dict[str, Any]]:
        last_err = None
        for attempt in range(max_retries + 1):
            try:
                raw = self.generate(prompt, system, temperature, max_tokens, format_json=True)
                return self._parse_json(raw)
            except Exception as e:
                last_err = e
                log.warning("json_call attempt %d failed: %s", attempt, e)
                time.sleep(0.5)
        log.error("json_call exhausted retries: %s", last_err)
        return None

    @staticmethod
    def _parse_json(text: str) -> Optional[Dict[str, Any]]:
        if not text:
            return None
        text = re.sub(r"```(?:json)?\s*", "", text).strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        m = _JSON_BLOCK.search(text)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
        try:
            fixed = re.sub(r'(\{|,)\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', text)
            return json.loads(fixed)
        except json.JSONDecodeError:
            pass
        return None

class MiniMaxClient:

    BASE_URL = "https://api.minimax.io/v1"
    DEFAULT_MODEL = "MiniMax-M2.7"

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: int = 180,
    ):
        import os
        self.model = model or os.environ.get("MINIMAX_MODEL", self.DEFAULT_MODEL)
        self.api_key = api_key or os.environ.get("MINIMAX_API_KEY", "")
        self.timeout = timeout

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
        format_json: bool = False,
    ) -> str:
        import os

        url = f"{self.BASE_URL}/text/chatcompletion_v2"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        body = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        resp = requests.post(url, headers=headers, json=body, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()

        base = data.get("base_resp", {})
        if base.get("status_code", 0) != 0:
            raise RuntimeError(f"MiniMax API error {base.get('status_code')}: {base.get('status_msg', 'unknown')}")

        choices = data.get("choices", [])
        if not choices:
            raise ValueError(f"No choices in MiniMax response: {data}")

        raw = choices[0].get("message", {}).get("content", "")
        if raw.startswith("<think>"):
            try:
                end = raw.index("</think>")
                raw = raw[end + len("</think>"):].strip()
            except ValueError:
                pass
        return raw.strip()

    def json_call(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
        max_retries: int = 2,
    ) -> Optional[Dict[str, Any]]:
        last_err = None
        for attempt in range(max_retries + 1):
            try:
                raw = self.generate(prompt, system, temperature, max_tokens, format_json=False)
                return self._parse_json(raw)
            except Exception as e:
                last_err = e
                log.warning("MiniMax json_call attempt %d failed: %s", attempt, e)
                time.sleep(1.0)
        log.error("MiniMax json_call exhausted retries: %s", last_err)
        return None

    @staticmethod
    def _parse_json(text: str) -> Optional[Dict[str, Any]]:
        if not text:
            return None
        text = re.sub(r"```(?:json)?\s*", "", text).strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        m = _JSON_BLOCK.search(text)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
        try:
            fixed = re.sub(r'(\{|,)\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', text)
            return json.loads(fixed)
        except json.JSONDecodeError:
            pass
        return None

def get_llm_client(backend: str = "ollama") -> "OllamaClient | MiniMaxClient":
    if backend == "minimax":
        return MiniMaxClient()
    return OllamaClient()
