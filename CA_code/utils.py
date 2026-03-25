# ============================================================
# utils.py — Reusable helper functions
# ============================================================

def string2any(value, target_type):
    """
    Convert a string value to a target Python type.
    Used by BaseModel.build() to parse configuration values.

    Args:
        value: The string value to convert.
        target_type: The Python type to convert to (e.g., int, float, str).

    Returns:
        The converted value.
    """
    try:
        return target_type(value)
    except (ValueError, TypeError) as e:
        raise ValueError(
            f"Cannot convert '{value}' to {target_type.__name__}: {e}"
        )
