"""The deployment preflight must count all native delimiters and reject truncation."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from traceoffload.m2.jev.types import JevUnavailable
from deploy.jev.openjev_preflight import OpenJevPreflight


class CharacterTokenizer:
    def encode(self, text, *, add_special_tokens=False):
        assert not add_special_tokens
        return SimpleNamespace(ids=[ord(c) for c in text])

    def token_to_id(self, text):
        return 200000 + OpenJevPreflight.SPECIAL.index(text)


def request():
    return {"model": "wev-1.7b", "state": {"task": "中文"}, "questions": {"route": {
        "type": "choice", "instructions": "Route", "criteria": {"LLM": "large", "SLM": "small"}}}}


def checker(**limits):
    return OpenJevPreflight(CharacterTokenizer(), max_state=limits.get("max_state", 4096),
                            max_branch=limits.get("max_branch", 8192), max_position=32768)


def test_complete_count_and_native_candidate_order():
    body = request()
    before = deepcopy(body)
    ids = checker().encode_ids(body)
    expected = 1 + len("task: 中文") + 1 + len("Route") + 2 * 2 + len("LLM: large") + len("SLM: small") + 1
    assert len(ids) == checker()(body) == expected
    assert body == before
    assert ids.count(200002) == ids.count(200003) == 2


@pytest.mark.parametrize("limits", [{"max_state": 3}, {"max_branch": 3}])
def test_over_limit_rejected_instead_of_shrunk(limits):
    with pytest.raises(JevUnavailable, match="input_too_long"):
        checker(**limits)(request())


def test_native_nested_rendering_and_special_token_escaping():
    c = checker()
    assert c.render({"x": ["a", {"b": True}], "z": None}) == "x:\n  - a\n  - b: True\nz: "
    body = request()
    body["state"] = "<|fim_middle|>"
    ids = c.encode_ids(body)
    assert ids[1:1 + len("<¦fim_middle¦>")] == [ord(x) for x in "<¦fim_middle¦>"]


def test_exact_limit_allowed():
    body = request()
    assert checker(max_state=1 + len("task: 中文"))(body) > 0


def test_rejects_wrong_question_shape():
    body = request()
    body["questions"]["route"]["criteria"] = {"SLM": "small", "LLM": "large"}
    with pytest.raises(JevUnavailable, match="invalid_request"):
        checker()(body)
