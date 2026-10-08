import pytest

from jobagent.llm.base import LLMClient, LLMError, LLMResult
from jobagent.models import Profile


class Fake(LLMClient):
    provider, model = "fake", "fake"

    def __init__(self, outputs):
        self.outputs = list(outputs)

    def _generate(self, system, user, schema, temperature):
        return LLMResult(text=self.outputs.pop(0), model="fake", provider="fake")


def test_retry_then_success():
    llm = Fake(["not json", '{"name": "A", "skills": ["Python"]}'])
    assert llm.generate_structured("s", "u", Profile).skills == ["Python"]


def test_gives_up():
    with pytest.raises(LLMError):
        Fake(["x", "x", "x"]).generate_structured("s", "u", Profile)
