"""
Tool System
===========
Allows the agent to use external tools and functions.

Example tools:
- Web search
- Code execution
- Data lookup
- Calculations
"""

import ast
import operator as op
from typing import Any, Optional, Callable, List, Dict
from pydantic import BaseModel, Field, PrivateAttr

# Binary operators allowed in the safe evaluator, mapped to their functions
_ALLOWED_BINOPS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.FloorDiv: op.floordiv,
    ast.Mod: op.mod,
    ast.Pow: op.pow,
}
_ALLOWED_UNARYOPS = {
    ast.UAdd: op.pos,
    ast.USub: op.neg,
}


def safe_eval(expression: str) -> float:
    """
    Safely evaluate a math expression using Python's AST — no eval().

    Only numbers and basic arithmetic are allowed. Guards against
    giant-exponent blowups like 9**9**9**9 and division by zero.
    """
    def _eval(node: ast.AST) -> float:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) \
                and not isinstance(node.value, bool):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
            left, right = _eval(node.left), _eval(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 100:
                raise ValueError("exponent too large")
            if isinstance(node.op, (ast.Div, ast.FloorDiv, ast.Mod)) and right == 0:
                raise ZeroDivisionError("division by zero")
            return _ALLOWED_BINOPS[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARYOPS:
            return _ALLOWED_UNARYOPS[type(node.op)](_eval(node.operand))
        raise ValueError(f"unsupported expression element: {ast.dump(node)}")

    tree = ast.parse(expression, mode="eval")
    return _eval(tree.body)


class Tool(BaseModel):
    """
    Represents an available tool for the agent.
    
    A tool has:
    - name: Unique identifier
    - description: What the tool does
    - parameters: Schema for input parameters
    - execute: Async function to run the tool
    """
    
    name: str
    description: str
    parameters: dict = Field(default_factory=dict)
    execute: Callable[..., Any]
    
    def to_dict(self) -> dict:
        """Convert to OpenAI function format"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters
            }
        }


class ToolRegistry:
    """
    Registry for managing available tools.
    
    Usage:
        registry = ToolRegistry()
        registry.register(Tool(...))
        schema = registry.to_openai_schema()
    """
    
    def __init__(self):
        self.tools: dict[str, Tool] = {}
        
    def register(self, tool: Tool) -> None:
        """Register a new tool"""
        self.tools[tool.name] = tool
        
    def unregister(self, name: str) -> bool:
        """Remove a tool by name"""
        if name in self.tools:
            del self.tools[name]
            return True
        return False
        
    def get(self, name: str) -> Optional[Tool]:
        """Get a tool by name"""
        return self.tools.get(name)
        
    def list(self) -> list[Tool]:
        """List all registered tools"""
        return list(self.tools.values())
        
    def to_openai_schema(self) -> List[Dict]:
        """Get tools in OpenAI function calling format"""
        return [tool.to_dict() for tool in self.tools.values()]
        
    def clear(self) -> None:
        """Remove all tools"""
        self.tools.clear()
        
    @property
    def count(self) -> int:
        """Get number of registered tools"""
        return len(self.tools)


# ── Built-in Example Tools ──────────────────────
class CalculatorTool(Tool):
    """Simple calculator tool for math operations"""
    
    def __init__(self):
        super().__init__(
            name="calculator",
            description="Performs mathematical calculations",
            parameters={
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Mathematical expression to evaluate (e.g., '2 + 3 * 4')"
                    }
                },
                "required": ["expression"]
            },
            execute=self._execute
        )
    
    async def _execute(self, expression: str) -> str:
        """Execute the calculation safely (AST-based, no eval)"""
        try:
            return str(safe_eval(expression))
        except SyntaxError:
            return "Error: invalid mathematical expression"
        except Exception as e:
            return f"Error: {e}"


class TimeTool(Tool):
    """Get current time and date"""
    
    def __init__(self):
        super().__init__(
            name="get_current_time",
            description="Get the current date and time",
            parameters={
                "type": "object",
                "properties": {}
            },
            execute=self._execute
        )
    
    async def _execute(self) -> str:
        """Return current timestamp"""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class SearchTool(Tool):
    """
    Generic search tool - wire it to any search backend.
    
    Usage:
        tool = SearchTool(search_function=my_search_fn)
    """
    
    _search_fn: Optional[Callable] = PrivateAttr(default=None)
    
    def __init__(self, search_function: Optional[Callable] = None, **data):
        super().__init__(
            name="web_search",
            description="Search the web for information",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query"
                    }
                },
                "required": ["query"]
            },
            execute=self._execute,
            **data
        )
        self._search_fn = search_function
    
    async def _execute(self, query: str) -> str:
        """Execute web search"""
        if self._search_fn:
            return await self._search_fn(query)
        return "Search functionality not configured"


# ── Convenience Functions ──────────────────────
def create_tool(name: str, description: str, fn: Callable[..., Any]) -> Tool:
    """
    Create a tool from a simple async function.
    
    The function's parameters become the tool's parameters.
    """
    import inspect
    
    sig = inspect.signature(fn)
    params = {}
    required = []
    
    for param_name, param in sig.parameters.items():
        if param_name == "self":
            continue
            
        param_type = "string"
        if param.annotation != inspect.Parameter.empty:
            if param.annotation == int:
                param_type = "integer"
            elif param.annotation == float:
                param_type = "number"
            elif param.annotation == bool:
                param_type = "boolean"
        
        params[param_name] = {
            "type": param_type,
            "description": f"Parameter {param_name}"
        }
        
        if param.default == inspect.Parameter.empty:
            required.append(param_name)
    
    schema = {"type": "object", "properties": params}
    if required:
        schema["required"] = required
    
    return Tool(
        name=name,
        description=description,
        parameters=schema,
        execute=fn
    )
