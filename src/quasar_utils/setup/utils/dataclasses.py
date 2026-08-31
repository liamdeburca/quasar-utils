from collections.abc import Callable
from dataclasses import _MISSING_TYPE, MISSING, Field, fields
from dataclasses import field as dataclass_field
from typing import Any, TypeVar

from astropy.units import Quantity

from .parsing import PARSER_KEYS, Parser

D = TypeVar("D", bound=type)


def field(
    *,
    default: Any | _MISSING_TYPE = MISSING,
    default_factory: Callable[[], Any] | _MISSING_TYPE = MISSING,
    dtype: str = "<No dtype>",
    parse_as: str = "<No parse_as>",
    update_to: str | None = None,
    desc: str = "<No description>",
    comment: str = "<No comment>",
    has_unit: bool = False,
    init: bool = True,
    repr: bool = False,
    hash: bool | None = None,
    compare: bool = True,
    kw_only: bool = True,
) -> Field:
    """
    Simple wrapper for 'dataclasses.field' that requires additional fields for 
    the metadata dictionary.

    Parameters
    ----------
    default : Any, optional
        Default value for the field. If not provided, 'default_factory' must be 
        specified.
    dtype: str, optional
        Data type used in config files. 
    parse_as: str | None, optional
        How to parse the field from config files. If None, no parsing is applied.
    update_to: str | None, optional
        How to update the field from the config file. Is required if a field's 
        name starts with an underscore.
    desc: str, optional
        Description of the field.
    comment: str, optional
        Additional comments for the field.
    default_factory: Callable[[], Any] | Literal[_MISSING_TYPE.MISSING], optional
        A callable that returns the default value for the field. If not 
        provided, 'default' must be specified.
    init: bool, optional
        Whether the field should be included in the generated __init__ method.
    repr: bool, optional
        Whether the field should be included in the generated __repr__ method.
    hash: bool | None, optional
        Whether the field should be included in the generated __hash__ method.
    compare: bool, optional
        Whether the field should be included in comparison methods.
    kw_only: bool, optional
        Whether the field should be keyword-only in the generated __init__ method.
    """
    assert default is not MISSING or default_factory is not MISSING

    if (parse_as != "<No parse_as>") and (parse_as not in PARSER_KEYS):
        raise ValueError(f"Invalid parse_as value '{parse_as}'.")
        
    return dataclass_field(
        default=default,
        default_factory=default_factory,
        init=init,
        repr=repr,
        hash=hash,
        compare=compare,
        metadata={
            "desc": desc,
            "comment": comment,
            "dtype": dtype,
            "parse_as": parse_as,
            "update_to": (update_to or "<No update_to>").removeprefix("to_"),
            "has_unit": has_unit,
        },
        kw_only=kw_only,
    )

def field_to_dict(dc: object, field: Field, jsonify: bool = False) -> dict[str, dict[str, Any]]:
    """
    Creates a dictionary representation of a dataclass field.

    This function is designed for writing JSON/YAML config files.

    Parameters
    ----------
    field : Field
        The dataclass field to convert.

    Returns
    -------
    dict[str, Any]
        A dictionary representation of the dataclass field.
    """
    metadata = field.metadata

    name = field.name
    value = getattr(dc, name)
    default = field.default_factory() if field.default is MISSING else field.default
    desc = metadata.get("desc", "<No description>")
    comment = metadata.get("comment", "<No comment>")
    dtype = metadata.get("dtype", "<No dtype>")
    parse_as = metadata.get("parse_as", "<No parse_as>")
    update_to = metadata.get("update_to", "<No update_to>")
    has_unit = metadata.get("has_unit", False)

    if isinstance(value, Quantity) and isinstance(default, Quantity):
        default = default.to(value.unit)

    if jsonify and parse_as != "<No parse_as>":
        undoer = Parser._undo(f"undo_as_{parse_as}")
        value_field = undoer(value)
        default_field = undoer(default)

        out = {
            "value": value_field["value"],
            "unit": value_field["unit"],
            "default": default_field["value"],
            "default_unit": default_field["unit"],
        }
        if out["unit"] is None and not has_unit:
            out.pop("unit")
        if out["default_unit"] is None and not has_unit:
            out.pop("default_unit")
    else:
        if has_unit:
            if isinstance(value, Quantity):
                unit = str(value.unit)
                value = value.value
            else:
                unit = None

            if isinstance(default, Quantity):
                default_unit = str(default.unit)
                default = default.value
            else:
                default_unit = None

            out = {
                "value": value, 
                "unit": unit, 
                "default": default, 
                "default_unit": default_unit,
            }
        else:
            out = {
                "value": value, 
                "default": default,
            }

    if desc and desc != "<No description>":
        out["desc"] = desc
    if comment and comment != "<No comment>":
        out["comment"] = comment
    if dtype and dtype != "<No dtype>":
        out["dtype"] = dtype
    if parse_as and parse_as != "<No parse_as>":
        out["parse_as"] = parse_as
    if update_to and update_to != "<No update_to>":
        out["update_to"] = update_to

    return {name.removeprefix("_"): out}

def get_field_metadata(cls: type) -> dict[str, dict[str, Any]]:
    return {f.name: f.metadata for f in fields(cls)}

###

def finalise_dataclass[D](
    cls: D | None = None,
    *,
    additional_keys: list[str] | None = None,
) -> D:
    """
    Finalises a dataclass instance by:

    1.  Collecting all field names (and any additional names) into the _keys 
        class attribute.
    2.  Updating the '_values_to_update' class attribute with fields that start 
        with an underscore. 
    3.  Initialising the '_cache' class attribute as an empty dictionary.
    """
    def func[D](_cls: D, _additional_keys=additional_keys or []) -> D:
        _keys: list[str] = []
        _keys.extend(_additional_keys)

        all_fields = fields(_cls)
        for field in all_fields:
            name = field.name
            _keys.append(name)

        _values_to_update: dict[str, str] = {}
        for field in filter(lambda f: f.name.startswith("_"), all_fields):
            name: str = field.name
            update_to: str = field.metadata.get("update_to", "<No update_to>")

            if name.removeprefix("_") not in _keys:
                raise ValueError(
                    f"Field '{name}' starts with an underscore but has no "
                    f"corresponding field '{name.removeprefix('_')}'."
                )
            
            if update_to == "<No update_to>":
                raise ValueError(
                    f"Field '{name}' needs to be updated but no 'update_to' method "
                    "is specified."
                )
            _values_to_update[name.removeprefix('_')] = update_to

        _cls._keys = frozenset(_keys)
        _cls._cache = {}
        _cls._values_to_update = _values_to_update

        return _cls

    return func if cls is None else func(cls)
