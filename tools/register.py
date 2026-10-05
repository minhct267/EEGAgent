from typing import Callable, Dict, Any, Optional, List, get_type_hints
import inspect
import json

# Type mapping from Python types to JSON Schema types
TYPE_MAP = {
    int: "integer",
    float: "number",
    str: "string",
    bool: "boolean",
    list: "array",
    dict: "object",
    type(None): "null"
}

class FunctionRegistry:
    """Store callables and the JSON schemas the planner is allowed to call."""
    def __init__(self):
        self.functions: Dict[str, Callable] = {}
        self.metadata: Dict[str, dict] = {}

    def register(self,
                 name: Optional[str] = None,
                 description: str = "",
                 parameters: Optional[List[Dict]] = None,
                 constraints: str = "",
                 returns: Optional[Dict] = None):
        """Decorator that records a function and the schema shown to the planner."""
        def decorator(func: Callable):
            nonlocal name
            if name is None:
                name = func.__name__

            properties = {}
            required = []
            for p in parameters or []:
                param_name = p["name"]
                properties[param_name] = {
                    "type": p["type"],
                    "description": p.get("description"),
                }
                if p.get("required", True):  # Omitted "required" means the planner must supply it.
                    required.append(param_name)

            schema = {
                "type": "object",
                "properties": properties,
                "required": required
            }

            if returns:
                return_info = returns
            else:
                # Fall back to the function's return annotation when the decorator omits one.
                return_type = TYPE_MAP.get(get_type_hints(func).get('return'), 'object')
                return_info = {
                    "type": return_type
                }

            self.functions[name] = func
            self.metadata[name] = {
                "name": name,
                "description": description,
                "parameters": schema,
                "constraints": constraints,
                "returns": return_info
            }
            return func
        return decorator

    def get_function(self, name: str) -> Callable:
        """Return the callable registered under this name, or None."""
        return self.functions.get(name)

    def get_metadata(self, name: str) -> dict:
        """Return the schema metadata for a registered name, or None."""
        return self.metadata.get(name)

    def list_functions(self) -> list:
        """Return every registered function name."""
        return list(self.functions.keys())

    def export_tool_schemas(self) -> list:
        """Export every registered tool as an LLM function-call schema."""
        return [
            {
                "type": "function",
                "function": {
                    "name": meta["name"],
                    "description": meta["description"],
                    "constraints": meta["constraints"],
                    "parameters": meta["parameters"],
                    "returns": meta["returns"]
                }
            }
            for meta in self.metadata.values()
        ]

# Shared registry. Tool modules fill it when the tools package is imported.
function_register = FunctionRegistry()
