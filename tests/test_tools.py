"""Tests for the tool system: registry, safe_eval, built-in tools, create_tool"""
import pytest

from app.agent.tools import (
    Tool,
    ToolRegistry,
    CalculatorTool,
    TimeTool,
    SearchTool,
    create_tool,
    safe_eval,
)


class TestSafeEval:
    def test_basic_arithmetic(self):
        assert safe_eval("2 + 3 * 4") == 14
        assert safe_eval("(2 + 3) * 4") == 20
        assert safe_eval("10 / 4") == 2.5

    def test_unary_minus(self):
        assert safe_eval("-5 + 3") == -2

    def test_power_small(self):
        assert safe_eval("2 ** 10") == 1024

    def test_power_giant_exponent_rejected(self):
        # Previously this would hang the process via eval()
        with pytest.raises(ValueError):
            safe_eval("9 ** 999999")

    def test_division_by_zero(self):
        with pytest.raises(ZeroDivisionError):
            safe_eval("1 / 0")

    def test_rejects_names_and_calls(self):
        with pytest.raises(ValueError):
            safe_eval("__import__('os').system('ls')")
        with pytest.raises(ValueError):
            safe_eval("open('/etc/passwd')")

    def test_rejects_syntax_error(self):
        with pytest.raises(SyntaxError):
            safe_eval("2 +* 3")


class TestToolRegistry:
    def test_register_and_get(self):
        reg = ToolRegistry()
        tool = Tool(name="t", description="d", parameters={}, execute=lambda: None)
        reg.register(tool)
        assert reg.get("t") is tool
        assert reg.count == 1

    def test_unregister(self):
        reg = ToolRegistry()
        tool = Tool(name="t", description="d", parameters={}, execute=lambda: None)
        reg.register(tool)
        assert reg.unregister("t") is True
        assert reg.unregister("t") is False
        assert reg.count == 0

    def test_to_openai_schema(self):
        reg = ToolRegistry()
        reg.register(CalculatorTool())
        schema = reg.to_openai_schema()
        assert schema[0]["type"] == "function"
        assert schema[0]["function"]["name"] == "calculator"


class TestBuiltinTools:
    @pytest.mark.asyncio
    async def test_calculator(self):
        result = await CalculatorTool().execute(expression="2 + 2")
        assert result == "4"

    @pytest.mark.asyncio
    async def test_calculator_rejects_injection(self):
        result = await CalculatorTool().execute(expression="__import__('os').system('dir')")
        assert result.startswith("Error")

    @pytest.mark.asyncio
    async def test_time_tool(self):
        result = await TimeTool().execute()
        # Format: YYYY-MM-DD HH:MM:SS
        assert len(result) == 19 and result[4] == "-" and ":" in result


class TestCreateTool:
    @pytest.mark.asyncio
    async def test_creates_tool_from_function(self):
        async def greet(name: str, times: int = 1):
            return name * times

        tool = create_tool("greet", "Greets someone", greet)
        assert tool.name == "greet"
        props = tool.parameters["properties"]
        assert props["name"]["type"] == "string"
        assert props["times"]["type"] == "integer"
        assert tool.parameters["required"] == ["name"]
        assert await tool.execute(name="hi", times=2) == "hihi"
