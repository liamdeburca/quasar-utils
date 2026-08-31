__all__ = ["Info"]

from collections.abc import Iterable
from dataclasses import field
from json import dump as dump_json
from json import load as load_json
from logging import getLogger
from typing import Any, ClassVar, Self

from pydantic.dataclasses import dataclass
from quasar_typing.pathlib import (
    AbsoluteJSONPath,
    AbsoluteYAMLPath,
    AnyAbsoluteJSONPath,
    AnyAbsoluteYAMLPath,
)
from yaml import dump as dump_yaml
from yaml import safe_load as load_yaml

from ..decorators import validate_call
from .absorption import AbsorptionInfo
from .balmer import BalmerInfo
from .continuum import ContinuumInfo
from .convolution import ConvolutionInfo
from .error import ErrorInfo
from .host import HostInfo
from .iron import IronInfo
from .lines import LinesInfo
from .loading import LoadingInfo
from .nonlinear import NonLinearInfo
from .units import UnitsInfo

logger = getLogger(__name__)


@dataclass
class Info:
    absorption: AbsorptionInfo = field(
        default_factory=AbsorptionInfo,
        kw_only=True, 
    )
    balmer: BalmerInfo = field(
        default_factory=BalmerInfo,
        kw_only=True, 
    )
    continuum: ContinuumInfo = field(
        default_factory=ContinuumInfo,
        kw_only=True, 
    )
    convolution: ConvolutionInfo = field(
        default_factory=ConvolutionInfo,
        kw_only=True, 
    )
    error: ErrorInfo = field(
        default_factory=ErrorInfo,
        kw_only=True, 
    )
    host: HostInfo = field(
        default_factory=HostInfo,
        kw_only=True, 
    )
    iron: IronInfo = field(
        kw_only=True, 
        default_factory=IronInfo,
    )
    lines: LinesInfo = field(
        default_factory=LinesInfo,
        kw_only=True, 
    )
    loading: LoadingInfo = field(
        default_factory=LoadingInfo,
        kw_only=True, 
    )
    nonlinear: NonLinearInfo = field(
        default_factory=NonLinearInfo,
        kw_only=True, 
    )
    units: UnitsInfo = field(
        kw_only=True, 
        default_factory=UnitsInfo,
    )

    _keys: ClassVar[frozenset[str]] = frozenset([
        "absorption",
        "balmer",
        "continuum",
        "convolution",
        "error",
        "host",
        "iron",
        "lines",
        "loading",
        "nonlinear",
        "units",
    ])

    def __post_init__(self) -> None:
        self.update()

    def __hash__(self) -> int:
        return hash(tuple((key, getattr(self, key)) for key in self._keys))

    def to_dict(
        self,
        jsonify: bool = False,
    ) -> dict[str, dict[str, Any]]:
        out = {}
        for key in sorted(self._keys):
            out.update(getattr(self, key).to_dict(jsonify=jsonify))
        return out

    @classmethod
    @validate_call
    def from_json(
        cls,
        json: dict[str, dict] | AbsoluteJSONPath | None = None,
        create_copy: bool = True,
    ) -> Self:

        if json is None:
            return Info()

        if not isinstance(json, dict):
            with open(json, "r") as f:
                json = load_json(f)

        def func(cls):
            return cls.from_json.__wrapped__(
                cls, 
                json=json, 
                create_copy=create_copy,
            )

        return Info(
            absorption=func(AbsorptionInfo),
            balmer=func(BalmerInfo),
            continuum=func(cls=ContinuumInfo),
            convolution=func(cls=ConvolutionInfo),
            error=func(cls=ErrorInfo),
            host=func(cls=HostInfo),
            iron=func(cls=IronInfo),
            lines=func(cls=LinesInfo),
            loading=func(cls=LoadingInfo),
            nonlinear=func(cls=NonLinearInfo),
            units=func(cls=UnitsInfo),
        )

    @classmethod
    @validate_call
    def from_yaml(
        cls,
        yaml: dict[str, dict] | AbsoluteYAMLPath | None = None,
        create_copy: bool = True,
    ) -> Self:

        if yaml is None:
            return Info()

        if not isinstance(yaml, dict):
            with open(yaml, "r") as f:
                yaml = load_yaml(f, )

        def func(cls):
            return cls.from_json.__wrapped__(
                cls, 
                json=yaml, 
                create_copy=create_copy,
            )

        return Info(
            absorption=func(AbsorptionInfo),
            balmer=func(BalmerInfo),
            continuum=func(cls=ContinuumInfo),
            convolution=func(cls=ConvolutionInfo),
            error=func(cls=ErrorInfo),
            host=func(cls=HostInfo),
            iron=func(cls=IronInfo),
            lines=func(cls=LinesInfo),
            loading=func(cls=LoadingInfo),
            nonlinear=func(cls=NonLinearInfo),
            units=func(cls=UnitsInfo),
        )

    @validate_call
    def to_json(self, path: AnyAbsoluteJSONPath) -> None:
        with open(path, "w") as f:
            dump_json(self.to_dict(jsonify=True), f, indent=4)

    @validate_call
    def to_yaml(self, path: AnyAbsoluteYAMLPath) -> None:
        with open(path, "w") as f:
            dump_yaml(self.to_dict(jsonify=True), f, indent=4)

    def __getstate__(self) -> dict:
        state: dict = {"_keys": self._keys}
        state.update({key: getattr(self, key) for key in self._keys})
        return state

    def __setstate__(self, state: dict) -> None:
        self._keys: frozenset[str] = state.pop("_keys")
        for key, value in state.items():
            setattr(self, key, value)

    def update(self) -> None:
        if self.is_updated:
            logger.debug("'Info' class is already updated!")
        else:
            self.force_update()

    def force_update(self) -> None:
        logger.debug("Updating 'Info' class.")
        self.loading.update(self)
        for key in self._keys:
            getattr(self, key).update(self)

    @property
    def is_updated(self) -> bool:
        return all(getattr(self, key).is_updated for key in self._keys)

    def __bool__(self) -> bool:
        return self.is_updated

    def __getitem__(
        self,
        key: str,
        subjects: Iterable[str] = [
            "absorption",
            "balmer",
            "continuum",
            "convolution",
            "error",
            "host",
            "iron",
            "lines",
            "loading",
            "nonlinear",
            "units",
            # 'plotting',
        ],
    ) -> Any:
        result = None
        for subinfo in (getattr(self, subject) for subject in subjects):
            if key in subinfo._keys:
                result = getattr(subinfo, key)
                break

        return result