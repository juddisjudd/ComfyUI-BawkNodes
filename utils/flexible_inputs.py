"""
Flexible Input Utilities for Dynamic UI (based on rgthree)
File: utils/flexible_inputs.py
"""

from typing import Union


class AnyType(str):
    """A special class that is always equal in not equal comparisons. Credit to pythongosssss"""

    def __ne__(self, __value: object) -> bool:
        return False


class FlexibleOptionalInputType(dict):
    """A special class to make flexible nodes that pass data to our python handlers.

    Enables both flexible/dynamic input types (like for Any Switch) or a dynamic number of inputs
    (like for Any Switch, Context Switch, Context Merge, Power Lora Loader, etc).

    Initially, ComfyUI only needed to return True for `__contains__` below, which told ComfyUI that
    our node will handle the input, regardless of what it is.

    However, after https://github.com/comfyanonymous/ComfyUI/pull/2666 ComfyUI's execution changed
    also checking the data for the key; specifically, the type which is the first tuple entry. This
    type is supplied to our FlexibleOptionalInputType and returned for any non-data key. This can be a
    real type, or use the AnyType for additional flexibility.
    """

    def __init__(self, type_func, data: Union[dict, None] = None):
        """Initializes the FlexibleOptionalInputType.

        Args:
            type_func: The flexible type to use when ComfyUI retrieves an unknown key (via `__getitem__`).
            data: An optional dict to use as the basis. This is stored both in a `data` attribute, so we
                can look it up without hitting our overrides, as well as iterated over and adding its key
                and values to our `self` keys. This way, when looked at, we will appear to represent this
                data. When used in an "optional" INPUT_TYPES, these are the starting optional node types.
        """
        super().__init__()
        self.data = data or {}
        self.type = type_func
        for key, value in self.data.items():
            self[key] = value

    def __contains__(self, key):
        """Return True for any key, allowing ComfyUI to pass any input to our node."""
        return True

    def __getitem__(self, key):
        """Return the type for any unknown key, or the stored value for known keys."""
        if key in self.data:
            return self.data[key]
        return self.type


def any_type(s=None):
    """Helper function for any type inputs"""
    return AnyType("*")