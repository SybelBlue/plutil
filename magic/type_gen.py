import json
from pathlib import Path


def _preference_type(preference: object) -> str:
    if not isinstance(preference, dict):
        return "Any"
    values = preference.get("enum")
    if isinstance(values, list) and values:
        return f"Literal[{', '.join(map(repr, values))}]"
    type_ = preference.get("type")
    assert isinstance(type_, str)
    return {
        "string": "str",
        "number": "float",
        "boolean": "bool",
    }.get(type_, "Any")


def parse_preferences_type(info_json_path: Path) -> str:
    """Emit Python source for an info.json preferences JSON schema."""
    info = json.loads(info_json_path.read_bytes())
    schema = info.get("preferences", {})
    if not isinstance(schema, dict):
        schema = {}
    fields = "\n".join(
        f"    {name}: {_preference_type(preference)}"
        for name, preference in schema.items()
    )
    imports = "from typing import Any, Literal, TypedDict"
    return f"{imports}\n\n\nclass Preferences(TypedDict):\n{fields or '    pass'}\n"
