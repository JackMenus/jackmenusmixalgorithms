"""契约对象的通用序列化：dataclass <-> dict/JSON。

所有契约都继承 ``Contract``，自动获得 ``to_dict`` / ``from_dict`` / ``to_json`` / ``from_json``。
目标是让每一条消息都能无损地跨进程、跨语言传递（JSON），并带上 ``schema_version``
方便将来演进。
"""
from __future__ import annotations

import dataclasses
import enum
import json
import typing
from typing import Any, ClassVar, Type, TypeVar, Union, get_args, get_origin

T = TypeVar("T", bound="Contract")

SCHEMA_VERSION = "1.0"


def _to_jsonable(obj: Any) -> Any:
    """递归把 dataclass / enum / numpy 标量 / 容器转换成纯 JSON 类型。"""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        out = {}
        for f in dataclasses.fields(obj):
            out[f.name] = _to_jsonable(getattr(obj, f.name))
        return out
    if isinstance(obj, enum.Enum):
        return obj.value
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_to_jsonable(v) for v in obj]
    # numpy 标量 / pandas Timestamp 等：尽量退化为 python 原生类型
    item = getattr(obj, "item", None)
    if callable(item) and not isinstance(obj, (str, bytes)):
        try:
            return item()
        except Exception:  # pragma: no cover - 防御性
            pass
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    return obj


def _is_optional(tp: Any) -> bool:
    return get_origin(tp) is Union and type(None) in get_args(tp)


def _strip_optional(tp: Any) -> Any:
    if _is_optional(tp):
        args = [a for a in get_args(tp) if a is not type(None)]
        return args[0] if len(args) == 1 else Union[tuple(args)]
    return tp


def _from_value(tp: Any, value: Any) -> Any:
    """按类型注解把 JSON 值还原成对象（支持嵌套 dataclass / list / dict / enum）。"""
    if value is None:
        return None
    tp = _strip_optional(tp)
    origin = get_origin(tp)
    if origin in (list, typing.List, tuple, typing.Tuple, set):
        (inner,) = get_args(tp)[:1] or (Any,)
        seq = [_from_value(inner, v) for v in value]
        return tuple(seq) if origin in (tuple, typing.Tuple) else seq
    if origin in (dict, typing.Dict):
        args = get_args(tp)
        vt = args[1] if len(args) == 2 else Any
        return {k: _from_value(vt, v) for k, v in value.items()}
    if isinstance(tp, type):
        if dataclasses.is_dataclass(tp) and isinstance(value, dict):
            return _from_dict(tp, value)
        if issubclass(tp, enum.Enum):
            return tp(value)
    return value


def _from_dict(cls: Type[T], data: dict) -> T:
    hints = typing.get_type_hints(cls)
    kwargs = {}
    for f in dataclasses.fields(cls):
        if f.name not in data:
            continue
        kwargs[f.name] = _from_value(hints.get(f.name, Any), data[f.name])
    return cls(**kwargs)  # type: ignore[arg-type]


@dataclasses.dataclass
class Contract:
    """所有跨库契约的基类。"""

    schema_version: ClassVar[str] = SCHEMA_VERSION

    def to_dict(self) -> dict:
        d = _to_jsonable(self)
        d["schema_version"] = self.schema_version
        return d

    @classmethod
    def from_dict(cls: Type[T], data: dict) -> T:
        data = dict(data)
        data.pop("schema_version", None)
        return _from_dict(cls, data)

    def to_json(self, **kwargs: Any) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, **kwargs)

    @classmethod
    def from_json(cls: Type[T], text: str) -> T:
        return cls.from_dict(json.loads(text))

    def replace(self: T, **changes: Any) -> T:
        return dataclasses.replace(self, **changes)
