"""Strict OpenJev 0.2.2 input packing for the deployed single-choice router.

Mirrors wev.api.render/to_record and wev.model.user_tokens/encode (Apache-2.0):
https://github.com/alanhuangyoo/OpenJev . Verified against the installed native
encoder before deployment. No model weights, CUDA or service adapter needed.
"""
from __future__ import annotations

import json
from pathlib import Path
import re

from traceoffload.m2.jev.types import JevUnavailable


class OpenJevPreflight:
    SPECIAL = ["<|fim_prefix|>", "<|fim_middle|>", "<|box_start|>", "<|box_end|>", "<|fim_suffix|>"]
    _SPECIAL_RE = re.compile(r"<\|([A-Za-z0-9_]+)\|>")

    def __init__(self, tokenizer, *, max_state: int, max_branch: int, max_position: int):
        self.tokenizer = tokenizer
        self.max_state, self.max_branch, self.max_position = max_state, max_branch, max_position
        if any(type(n) is not int or n <= 0 for n in (max_state, max_branch, max_position)):
            raise ValueError("Native input limits must be positive integers")
        self.special = [tokenizer.token_to_id(t) for t in self.SPECIAL]
        if any(type(n) is not int for n in self.special):
            raise ValueError("Native delimiter tokens missing")

    @classmethod
    def from_directory(cls, directory):
        from tokenizers import Tokenizer
        root = Path(directory)
        meta = json.loads((root / "wev.json").read_text(encoding="utf-8"))
        config = json.loads((root / "config.json").read_text(encoding="utf-8"))
        return cls(Tokenizer.from_file(str(root / "tokenizer.json")),
                   max_state=meta["max_state"], max_branch=meta["max_branch"],
                   max_position=config["max_position_embeddings"])

    @classmethod
    def render(cls, value, indent=0):
        pad = "  " * indent
        if value is None:
            return ""
        if isinstance(value, (str, int, float, bool)):
            return str(value)
        if isinstance(value, list):
            return "\n".join(f"{pad}- {cls.render(x, indent + 1).lstrip()}" for x in value)
        return "\n".join(
            f"{pad}{k}:\n{cls.render(v, indent + 1)}" if isinstance(v, (dict, list))
            else f"{pad}{k}: {cls.render(v)}" for k, v in value.items())

    def _tokens(self, text):
        escaped = self._SPECIAL_RE.sub(r"<¦\1¦>", text)
        return self.tokenizer.encode(escaped, add_special_tokens=False).ids

    def encode_ids(self, body):
        questions = body.get("questions", {})
        if list(questions) != ["route"]:
            raise JevUnavailable("invalid_request")
        question = questions["route"]
        criteria = question.get("criteria", {})
        if question.get("type") != "choice" or list(criteria) != ["LLM", "SLM"]:
            raise JevUnavailable("invalid_request")
        s, q, o, c, d = self.special
        prefix = [s] + self._tokens(self.render(body["state"]))
        branch = [q] + self._tokens(self.render(question["instructions"]))
        for name, desc in criteria.items():
            text = name if desc is None or desc == "" else f"{name}: {self.render(desc)}"
            branch += [o] + self._tokens(text) + [c]
        branch += [d]
        if (len(prefix) > self.max_state or len(branch) > self.max_branch
                or len(prefix) + len(branch) > self.max_position):
            raise JevUnavailable("input_too_long")
        return prefix + branch

    def __call__(self, body):
        return len(self.encode_ids(body))
