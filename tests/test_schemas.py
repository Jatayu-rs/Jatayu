import pytest
from pydantic import ValidationError

from jatayu.schemas import TaskName, ToolRequest, ToolResult


def test_confidence_must_be_a_probability():
    with pytest.raises(ValidationError):
        ToolResult(answer="x", confidence=1.5, confidence_method="stub",
                   tool_name=TaskName.VQA, model_id="stub")


def test_confidence_method_is_mandatory():
    with pytest.raises(ValidationError):
        ToolResult(answer="x", confidence=0.9, tool_name=TaskName.VQA, model_id="stub")


def test_request_needs_an_image():
    with pytest.raises(ValidationError):
        ToolRequest(query="what is this", images=[])
