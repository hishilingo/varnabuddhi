# -*- coding: utf-8 -*-
"""
LLM Translation Engine
========================
Multi-provider LLM translation with academic-precision prompts.

Supported providers:

* **OpenAI** (``openai`` package)
* **Anthropic** (``anthropic`` package)
* **Google Gemini** (``google-genai`` package)
* **OpenRouter** (plain HTTP — no extra package needed)
* **Ollama** (local server, plain HTTP — no extra package needed)

Only ONE model is active at a time.  The engine supports graceful fallback
through a configurable provider order.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Generator, List, Optional

logger = logging.getLogger("varnabuddhi.llm")

# ---------------------------------------------------------------------------
# Academic system prompt
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT_SAN_TO_ENG = """\
You are an expert Sanskritist and philologist.  Translate the following \
Sanskrit text into English with academic precision.

Guidelines:
- Preserve grammatical nuance and philosophical terminology.
- Use standard scholarly transliteration (IAST) for untranslatable terms.
- Provide literal, philologically rigorous translations — avoid colloquialisms.
- Where a term has established English equivalents in Indological scholarship, \
  prefer those.
- If the text is a verse (śloka), preserve the verse structure in the \
  translation.
- Be concise but accurate.
"""

_SYSTEM_PROMPT_ENG_TO_SAN = """\
You are an expert Sanskritist and grammarian.  Translate the following \
English text into Classical Sanskrit (IAST transliteration).

Guidelines:
- Produce grammatically correct Sanskrit following Pāṇinian conventions.
- Use standard Classical Sanskrit vocabulary and constructions.
- Output IAST-transliterated Sanskrit only (not Devanagari).
- Separate individual words with spaces (pre-sandhi padapāṭha form) so \
  sandhi can be applied downstream.
- Be concise and precise.
"""

_CONTEXT_TEMPLATE = """\
The following known word-level translations from a scholarly lexicon may \
assist you.  Use them where appropriate, but you are free to improve or \
override them if the context demands:

{word_hints}
"""

_GRAMMAR_TEMPLATE = """\
The following morphological analysis was obtained from Pāṇinian \
declension/conjugation tables and should be treated as reliable \
grammatical ground truth.  Use it to resolve ambiguities:

{grammar_tags}
"""


# ===================================================================
# Provider implementations
# ===================================================================


# ---------------------------------------------------------------------------
# Generation parameter defaults (tuned for academic translation accuracy)
# ---------------------------------------------------------------------------
# temperature 0.1  — near-deterministic; avoids creative paraphrasing
# top_p 0.9        — focused nucleus sampling
# top_k 40         — limits token candidates (Gemini/Ollama)
# max_output_tokens 4096 — sufficient for longer passages & verse blocks
# frequency_penalty 0.0  — Sanskrit legitimately repeats (mantras, compounds)
# presence_penalty 0.0   — no need to force topic diversity in translation
_DEFAULT_GENERATION: Dict[str, Any] = {
    "temperature": 0.1,
    "max_output_tokens": 4096,
    "top_p": 0.9,
    "top_k": 40,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0,
}


def _gen(config: Dict[str, Any], key: str) -> Any:
    """Resolve a generation parameter: provider override → global → default."""
    gen = config.get("_generation", {})
    return gen.get(key, _DEFAULT_GENERATION.get(key))


# ===================================================================
# Provider implementations
# ===================================================================


def _call_openai(
    config: Dict[str, Any], messages: List[Dict[str, str]]
) -> str:
    """Call the OpenAI Chat Completions API."""
    try:
        import openai
    except ImportError:
        raise RuntimeError("OpenAI provider requires: pip install openai")

    client = openai.OpenAI(
        api_key=config["api_key"],
        base_url=config.get("base_url", "https://api.openai.com/v1"),
    )
    resp = client.chat.completions.create(
        model=config["model"],
        messages=messages,
        temperature=_gen(config, "temperature"),
        top_p=_gen(config, "top_p"),
        max_tokens=_gen(config, "max_output_tokens"),
        frequency_penalty=_gen(config, "frequency_penalty"),
        presence_penalty=_gen(config, "presence_penalty"),
    )
    return resp.choices[0].message.content.strip()


def _call_anthropic(
    config: Dict[str, Any], messages: List[Dict[str, str]]
) -> str:
    """Call the Anthropic Messages API."""
    try:
        import anthropic
    except ImportError:
        raise RuntimeError("Anthropic provider requires: pip install anthropic")

    system_msg = ""
    user_messages = []
    for m in messages:
        if m["role"] == "system":
            system_msg = m["content"]
        else:
            user_messages.append(m)

    client = anthropic.Anthropic(api_key=config["api_key"])
    resp = client.messages.create(
        model=config["model"],
        system=system_msg,
        messages=user_messages,
        max_tokens=_gen(config, "max_output_tokens"),
        temperature=_gen(config, "temperature"),
        top_p=_gen(config, "top_p"),
    )
    return resp.content[0].text.strip()


def _call_gemini(
    config: Dict[str, Any], messages: List[Dict[str, str]]
) -> str:
    """Call the Google Gemini API via the google-genai SDK."""
    try:
        from google import genai
        from google.genai import types as genai_types
    except ImportError:
        raise RuntimeError(
            "Gemini provider requires: pip install google-genai"
        )

    client = genai.Client(api_key=config["api_key"])

    system_msg = next(
        (m["content"] for m in messages if m["role"] == "system"), None
    )
    user_parts = [m["content"] for m in messages if m["role"] != "system"]
    prompt = "\n\n".join(user_parts)

    resp = client.models.generate_content(
        model=config["model"],
        contents=prompt,
        config=genai_types.GenerateContentConfig(
            system_instruction=system_msg,
            temperature=_gen(config, "temperature"),
            top_p=_gen(config, "top_p"),
            top_k=_gen(config, "top_k"),
            max_output_tokens=_gen(config, "max_output_tokens"),
        ),
    )
    return resp.text.strip()


def _call_openrouter(
    config: Dict[str, Any], messages: List[Dict[str, str]]
) -> str:
    """Call the OpenRouter Chat Completions API (no extra package needed)."""
    import urllib.request
    import urllib.error

    api_key = config["api_key"]
    model = config["model"]
    base_url = config.get("base_url", "https://openrouter.ai/api/v1")
    url = f"{base_url}/chat/completions"

    body: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": _gen(config, "temperature"),
        "top_p": _gen(config, "top_p"),
        "max_tokens": _gen(config, "max_output_tokens"),
        "frequency_penalty": _gen(config, "frequency_penalty"),
        "presence_penalty": _gen(config, "presence_penalty"),
    }
    if config.get("reasoning_enabled", False):
        body["reasoning"] = {"enabled": True}

    payload = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data["choices"][0]["message"]["content"].strip()
    except (urllib.error.URLError, OSError, ConnectionError) as exc:
        raise RuntimeError(f"OpenRouter API error: {exc}")
    except (KeyError, IndexError) as exc:
        raise RuntimeError(f"OpenRouter unexpected response format: {exc}")


def _call_ollama(
    config: Dict[str, Any], messages: List[Dict[str, str]]
) -> str:
    """Call a local Ollama server via HTTP."""
    import urllib.request
    import urllib.error

    host = config.get("host", "localhost")
    port = config.get("port", 11434)
    model = config.get("model", "llama3")

    url = f"http://{host}:{port}/api/chat"
    payload = json.dumps({
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": _gen(config, "temperature"),
            "top_p": _gen(config, "top_p"),
            "top_k": _gen(config, "top_k"),
            "num_predict": _gen(config, "max_output_tokens"),
            "repeat_penalty": 1.0 + _gen(config, "frequency_penalty"),
        },
    }).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            return body.get("message", {}).get("content", "").strip()
    except (urllib.error.URLError, OSError, ConnectionError) as exc:
        raise RuntimeError(
            f"Ollama server not reachable at {host}:{port} — "
            f"is it running?  ({exc})"
        )


# ===================================================================
# Streaming provider implementations
# ===================================================================


def _stream_openai(
    config: Dict[str, Any], messages: List[Dict[str, str]]
) -> Generator[str, None, None]:
    """Stream from the OpenAI Chat Completions API."""
    try:
        import openai
    except ImportError:
        raise RuntimeError("OpenAI provider requires: pip install openai")

    client = openai.OpenAI(
        api_key=config["api_key"],
        base_url=config.get("base_url", "https://api.openai.com/v1"),
    )
    stream = client.chat.completions.create(
        model=config["model"],
        messages=messages,
        temperature=_gen(config, "temperature"),
        top_p=_gen(config, "top_p"),
        max_tokens=_gen(config, "max_output_tokens"),
        frequency_penalty=_gen(config, "frequency_penalty"),
        presence_penalty=_gen(config, "presence_penalty"),
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta if chunk.choices else None
        if delta and delta.content:
            yield delta.content


def _stream_anthropic(
    config: Dict[str, Any], messages: List[Dict[str, str]]
) -> Generator[str, None, None]:
    """Stream from the Anthropic Messages API."""
    try:
        import anthropic
    except ImportError:
        raise RuntimeError("Anthropic provider requires: pip install anthropic")

    system_msg = ""
    user_messages = []
    for m in messages:
        if m["role"] == "system":
            system_msg = m["content"]
        else:
            user_messages.append(m)

    client = anthropic.Anthropic(api_key=config["api_key"])
    with client.messages.stream(
        model=config["model"],
        system=system_msg,
        messages=user_messages,
        max_tokens=_gen(config, "max_output_tokens"),
        temperature=_gen(config, "temperature"),
        top_p=_gen(config, "top_p"),
    ) as stream:
        for text in stream.text_stream:
            yield text


def _stream_gemini(
    config: Dict[str, Any], messages: List[Dict[str, str]]
) -> Generator[str, None, None]:
    """Stream from the Google Gemini API via google-genai SDK."""
    try:
        from google import genai
        from google.genai import types as genai_types
    except ImportError:
        raise RuntimeError("Gemini provider requires: pip install google-genai")

    client = genai.Client(api_key=config["api_key"])

    system_msg = next(
        (m["content"] for m in messages if m["role"] == "system"), None
    )
    user_parts = [m["content"] for m in messages if m["role"] != "system"]
    prompt = "\n\n".join(user_parts)

    for chunk in client.models.generate_content_stream(
        model=config["model"],
        contents=prompt,
        config=genai_types.GenerateContentConfig(
            system_instruction=system_msg,
            temperature=_gen(config, "temperature"),
            top_p=_gen(config, "top_p"),
            top_k=_gen(config, "top_k"),
            max_output_tokens=_gen(config, "max_output_tokens"),
        ),
    ):
        if chunk.text:
            yield chunk.text


def _stream_openrouter(
    config: Dict[str, Any], messages: List[Dict[str, str]]
) -> Generator[str, None, None]:
    """Stream from the OpenRouter Chat Completions API (SSE, no extra package)."""
    import urllib.request
    import urllib.error

    api_key = config["api_key"]
    model = config["model"]
    base_url = config.get("base_url", "https://openrouter.ai/api/v1")
    url = f"{base_url}/chat/completions"

    body: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": True,
        "temperature": _gen(config, "temperature"),
        "top_p": _gen(config, "top_p"),
        "max_tokens": _gen(config, "max_output_tokens"),
        "frequency_penalty": _gen(config, "frequency_penalty"),
        "presence_penalty": _gen(config, "presence_penalty"),
    }
    if config.get("reasoning_enabled", False):
        body["reasoning"] = {"enabled": True}

    payload = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            for line in resp:
                line = line.decode("utf-8").strip()
                if not line or not line.startswith("data: "):
                    continue
                data_str = line[len("data: "):]
                if data_str == "[DONE]":
                    break
                try:
                    obj = json.loads(data_str)
                except json.JSONDecodeError:
                    continue
                delta = obj.get("choices", [{}])[0].get("delta", {})
                content = delta.get("content", "")
                if content:
                    yield content
    except (urllib.error.URLError, OSError, ConnectionError) as exc:
        raise RuntimeError(f"OpenRouter streaming error: {exc}")


def _stream_ollama(
    config: Dict[str, Any], messages: List[Dict[str, str]]
) -> Generator[str, None, None]:
    """Stream from a local Ollama server via HTTP (NDJSON)."""
    import urllib.request
    import urllib.error

    host = config.get("host", "localhost")
    port = config.get("port", 11434)
    model = config.get("model", "llama3")

    url = f"http://{host}:{port}/api/chat"
    payload = json.dumps({
        "model": model,
        "messages": messages,
        "stream": True,
        "options": {
            "temperature": _gen(config, "temperature"),
            "top_p": _gen(config, "top_p"),
            "top_k": _gen(config, "top_k"),
            "num_predict": _gen(config, "max_output_tokens"),
            "repeat_penalty": 1.0 + _gen(config, "frequency_penalty"),
        },
    }).encode("utf-8")

    req = urllib.request.Request(
        url, data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            for line in resp:
                line = line.decode("utf-8").strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                content = obj.get("message", {}).get("content", "")
                if content:
                    yield content
                if obj.get("done"):
                    break
    except (urllib.error.URLError, OSError, ConnectionError) as exc:
        raise RuntimeError(
            f"Ollama server not reachable at {host}:{port} — "
            f"is it running?  ({exc})"
        )


_STREAM_DISPATCH = {
    "openai": _stream_openai,
    "anthropic": _stream_anthropic,
    "gemini": _stream_gemini,
    "openrouter": _stream_openrouter,
    "ollama": _stream_ollama,
}


def ollama_is_reachable(host: str = "localhost", port: int = 11434) -> bool:
    """Quick check whether an Ollama server responds (non-blocking)."""
    import urllib.request
    import urllib.error

    try:
        req = urllib.request.Request(
            f"http://{host}:{port}/api/tags", method="GET"
        )
        with urllib.request.urlopen(req, timeout=3):
            return True
    except Exception:
        return False


_PROVIDER_DISPATCH = {
    "openai": _call_openai,
    "anthropic": _call_anthropic,
    "gemini": _call_gemini,
    "openrouter": _call_openrouter,
    "ollama": _call_ollama,
}

# Sensible default models when none is specified in config
_PROVIDER_DEFAULT_MODELS: Dict[str, str] = {
    "openai": "gpt-4o",
    "anthropic": "claude-sonnet-4-20250514",
    "gemini": "gemini-2.0-flash",
    "openrouter": "google/gemini-3-flash-preview",
    "ollama": "llama3",
}


# ===================================================================
# Public LLM Engine class
# ===================================================================


class LLMEngine:
    """Multi-provider LLM translation engine with configurable prompt profiles."""

    def __init__(self, llm_config: Dict[str, Any]) -> None:
        self.config = llm_config
        self._fallback_order: List[str] = llm_config.get(
            "fallback_order", ["ollama", "gemini", "openai", "anthropic"]
        )

        # --- Active profile ---
        self._active_profile: str = llm_config.get("active_profile", "default")
        profiles = llm_config.get("profiles", {})
        if self._active_profile not in profiles and profiles:
            self._active_profile = next(iter(profiles))
            logger.info(
                "Configured profile '%s' not found; using '%s'.",
                llm_config.get("active_profile", ""), self._active_profile,
            )

        # Resolve active provider: use configured value if its provider
        # entry looks usable, otherwise auto-detect the first usable one.
        requested_provider = llm_config.get("active_provider", "")
        requested_model = llm_config.get("active_model", "")

        if requested_provider and self._is_provider_configured(requested_provider):
            self._active_provider = requested_provider
        else:
            auto = self._auto_detect_provider()
            if auto:
                if requested_provider:
                    logger.info(
                        "Configured provider '%s' is not usable; "
                        "auto-selected '%s'.",
                        requested_provider, auto,
                    )
                self._active_provider = auto
            else:
                # Nothing usable — keep the requested one so error messages
                # make sense; translate() will report the failure.
                self._active_provider = requested_provider or "ollama"
                logger.warning("No usable LLM provider found.")

        self._active_model: str = requested_model

    # ------------------------------------------------------------------
    # Provider helpers
    # ------------------------------------------------------------------
    def _is_provider_configured(self, provider: str) -> bool:
        """Return True if *provider* has enough config to attempt a call.

        * Cloud providers (openai/anthropic/gemini) need a non-empty API key.
        * Ollama only needs its entry to exist — the server may or may not
          be running, but that is checked lazily at call time.
        """
        if provider not in _PROVIDER_DISPATCH:
            return False
        prov_cfg = self.config.get("providers", {}).get(provider, {})
        if not prov_cfg:
            return False
        if provider == "ollama":
            return True  # Ollama is always "configured" if the entry exists
        return bool(prov_cfg.get("api_key"))

    def _auto_detect_provider(self) -> str:
        """Pick the first usable provider from the fallback order.

        Prefers cloud providers with an API key over Ollama, since Ollama
        may not be running.
        """
        # First pass: cloud providers with keys
        for p in self._fallback_order:
            if p == "ollama":
                continue
            if self._is_provider_configured(p):
                return p
        # Second pass: Ollama (may or may not be reachable)
        for p in self._fallback_order:
            if p == "ollama" and self._is_provider_configured(p):
                return p
        return ""

    def _resolve_model(self, provider: str) -> str:
        """Return the model to use for *provider*.

        Priority: explicit ``_active_model`` (if this is the active
        provider) → provider-level ``model`` key → provider-specific
        sensible default.
        """
        if provider == self._active_provider and self._active_model:
            return self._active_model
        prov_cfg = self.config.get("providers", {}).get(provider, {})
        model = prov_cfg.get("model", "")
        if model:
            return model
        # Sensible defaults when no model is specified at all
        return _PROVIDER_DEFAULT_MODELS.get(provider, "")

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------
    @property
    def active_provider(self) -> str:
        return self._active_provider

    @property
    def active_model(self) -> str:
        return self._resolve_model(self._active_provider)

    @property
    def active_profile(self) -> str:
        return self._active_profile

    @property
    def active_profile_description(self) -> str:
        """Human-readable description of the active profile."""
        profiles = self.config.get("profiles", {})
        profile = profiles.get(self._active_profile, {})
        return profile.get("description", self._active_profile)

    @property
    def available_profiles(self) -> List[str]:
        """Return the list of configured profile names."""
        return list(self.config.get("profiles", {}).keys())

    @property
    def _provider_config(self) -> Dict[str, Any]:
        return self.config.get("providers", {}).get(self._active_provider, {})

    def set_provider(self, provider: str, model: str = "") -> None:
        """Switch the active provider and optionally the model."""
        if provider not in _PROVIDER_DISPATCH:
            raise ValueError(
                f"Unknown provider '{provider}'.  "
                f"Available: {', '.join(_PROVIDER_DISPATCH)}"
            )
        self._active_provider = provider
        if model:
            self._active_model = model
        else:
            self._active_model = ""  # will fall through to _resolve_model
        logger.info(
            "Switched to provider=%s model=%s",
            provider, self.active_model,
        )

    def set_profile(self, profile_name: str) -> None:
        """Switch the active prompt profile."""
        profiles = self.config.get("profiles", {})
        if profile_name not in profiles:
            available = ", ".join(profiles.keys()) or "(none defined)"
            raise ValueError(
                f"Unknown profile '{profile_name}'.  "
                f"Available: {available}"
            )
        self._active_profile = profile_name
        logger.info("Switched to profile '%s'.", profile_name)

    # ------------------------------------------------------------------
    # Translation
    # ------------------------------------------------------------------
    def translate(
        self,
        text: str,
        direction: str = "san_to_eng",
        word_hints: Optional[Dict[str, str]] = None,
        grammar_tags: Optional[Dict[str, str]] = None,
    ) -> str:
        """Translate *text* using the active LLM provider.

        Parameters
        ----------
        text : str
            The source text.
        direction : str
            ``"san_to_eng"`` or ``"eng_to_san"``.
        word_hints : dict, optional
            Known word-level translations from the lexicon engine.
        grammar_tags : dict, optional
            Morphological analysis for individual words (vibhakti, lakāra, etc.)
            obtained from the verification engine.

        Returns
        -------
        str
            The translated text.
        """
        # Resolve system prompt from the active profile (fall back to
        # hardcoded defaults if the profile lacks a prompt for this direction)
        profiles = self.config.get("profiles", {})
        profile = profiles.get(self._active_profile, {})
        prompt_key = "san_to_eng" if direction == "san_to_eng" else "eng_to_san"
        system_prompt = profile.get(
            prompt_key,
            _SYSTEM_PROMPT_SAN_TO_ENG
            if direction == "san_to_eng"
            else _SYSTEM_PROMPT_ENG_TO_SAN,
        )

        user_content = text
        if word_hints:
            hint_str = "\n".join(f"  {k} → {v}" for k, v in word_hints.items())
            user_content = (
                _CONTEXT_TEMPLATE.format(word_hints=hint_str) + "\n\n" + text
            )
        if grammar_tags:
            tag_str = "\n".join(f"  {k}: {v}" for k, v in grammar_tags.items())
            user_content = (
                _GRAMMAR_TEMPLATE.format(grammar_tags=tag_str)
                + "\n\n" + user_content
            )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        # Build ordered list: active provider first, then remaining fallbacks
        providers_to_try = [self._active_provider] + [
            p for p in self._fallback_order if p != self._active_provider
        ]

        last_error: Optional[Exception] = None
        skipped_all = True
        for provider in providers_to_try:
            dispatch_fn = _PROVIDER_DISPATCH.get(provider)
            if dispatch_fn is None:
                continue
            if not self._is_provider_configured(provider):
                continue

            prov_config = dict(
                self.config.get("providers", {}).get(provider, {})
            )

            # Resolve model — skip the provider if we still have no model
            model = self._resolve_model(provider)
            if not model:
                logger.debug(
                    "Skipping provider '%s': no model configured.", provider
                )
                continue
            prov_config["model"] = model

            # Inject generation parameters:
            # defaults → global config → profile overrides → provider overrides
            gen_params = dict(_DEFAULT_GENERATION)
            gen_params.update(self.config.get("generation", {}))
            gen_params.update(profile.get("generation", {}))
            gen_params.update(prov_config.get("generation", {}))
            prov_config["_generation"] = gen_params

            skipped_all = False
            try:
                result = dispatch_fn(prov_config, messages)
                if provider != self._active_provider:
                    logger.info("Fell back to provider '%s'.", provider)
                return result
            except Exception as exc:
                last_error = exc
                logger.warning("Provider '%s' failed: %s", provider, exc)
                continue

        if skipped_all:
            raise RuntimeError(
                "No usable LLM provider is configured.  "
                "Set at least one provider with an API key (or model) in "
                "config.json → llm → providers."
            )
        raise RuntimeError(
            f"All LLM providers failed.  Last error: {last_error}"
        )

    def translate_stream(
        self,
        text: str,
        direction: str = "san_to_eng",
        word_hints: Optional[Dict[str, str]] = None,
        grammar_tags: Optional[Dict[str, str]] = None,
    ) -> Generator[str, None, None]:
        """Streaming variant of :meth:`translate`.

        Yields text chunks as they arrive from the LLM.  Falls back to
        non-streaming :meth:`translate` if the streaming call fails.
        """
        # Build messages (same logic as translate)
        profiles = self.config.get("profiles", {})
        profile = profiles.get(self._active_profile, {})
        prompt_key = "san_to_eng" if direction == "san_to_eng" else "eng_to_san"
        system_prompt = profile.get(
            prompt_key,
            _SYSTEM_PROMPT_SAN_TO_ENG
            if direction == "san_to_eng"
            else _SYSTEM_PROMPT_ENG_TO_SAN,
        )

        user_content = text
        if word_hints:
            hint_str = "\n".join(f"  {k} → {v}" for k, v in word_hints.items())
            user_content = (
                _CONTEXT_TEMPLATE.format(word_hints=hint_str) + "\n\n" + text
            )
        if grammar_tags:
            tag_str = "\n".join(f"  {k}: {v}" for k, v in grammar_tags.items())
            user_content = (
                _GRAMMAR_TEMPLATE.format(grammar_tags=tag_str)
                + "\n\n" + user_content
            )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        providers_to_try = [self._active_provider] + [
            p for p in self._fallback_order if p != self._active_provider
        ]

        last_error: Optional[Exception] = None
        for provider in providers_to_try:
            stream_fn = _STREAM_DISPATCH.get(provider)
            if stream_fn is None:
                continue
            if not self._is_provider_configured(provider):
                continue

            prov_config = dict(
                self.config.get("providers", {}).get(provider, {})
            )
            model = self._resolve_model(provider)
            if not model:
                continue
            prov_config["model"] = model

            gen_params = dict(_DEFAULT_GENERATION)
            gen_params.update(self.config.get("generation", {}))
            gen_params.update(profile.get("generation", {}))
            gen_params.update(prov_config.get("generation", {}))
            prov_config["_generation"] = gen_params

            try:
                yield from stream_fn(prov_config, messages)
                return  # Success — done
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "Streaming provider '%s' failed: %s", provider, exc
                )
                continue

        # All providers failed
        raise RuntimeError(
            f"All LLM providers failed (streaming).  Last error: {last_error}"
        )

    @property
    def available_providers(self) -> List[str]:
        """Return the list of configured provider names."""
        return list(self.config.get("providers", {}).keys())

    @property
    def usable_providers(self) -> List[str]:
        """Return only providers that have enough config to attempt a call."""
        return [
            p for p in self.available_providers
            if self._is_provider_configured(p)
        ]
