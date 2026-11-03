import pytest

from code_interpreter_agent import PythonTool, ToolRegistry, ToolResult


class Boom:
    name = "boom"
    description = "always fails"
    input_schema = {"type": "object"}

    def run(self, session_id, arguments):
        raise RuntimeError("bad")


def test_registry_rejects_duplicates():
    registry = ToolRegistry([Boom()])
    with pytest.raises(ValueError):
        registry.register(Boom())


def test_registry_converts_exceptions_to_error_results():
    result = ToolRegistry([Boom()]).run("s", "boom", {})
    assert result.is_error and "RuntimeError: bad" in result.content


def test_registry_specs():
    specs = ToolRegistry([Boom()]).specs()
    assert [s.name for s in specs] == ["boom"]


def test_python_tool_formats_output(executor):
    tool = PythonTool(executor)
    result = tool.run("s", {"code": "print('hi')\nopen('o.txt', 'w').write('x')"})
    assert isinstance(result, ToolResult)
    assert "stdout:\nhi" in result.content and "o.txt" in result.content
    assert result.files == ("o.txt",)


def test_python_tool_reports_no_output(executor):
    assert PythonTool(executor).run("s", {"code": "x = 1"}).content == "(no output)"


def test_python_tool_validates_arguments(executor):
    tool = PythonTool(executor)
    assert tool.run("s", {}).is_error
    assert tool.run("s", {"code": "   "}).is_error
    assert tool.run("s", {"code": 5}).is_error
