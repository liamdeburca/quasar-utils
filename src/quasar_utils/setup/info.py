__all__ = ["Info"]

import hashlib
import json as json_pkg
from collections.abc import Iterable
from dataclasses import field
from logging import getLogger
from typing import Any, ClassVar, Self

import pydantic
from quasar_typing.pathlib import (
    AnyJSONPath,
    AnyYAMLPath,
    JSONPath,
    YAMLPath,
)

from .absorption import AbsorptionInfo
from .balmer import BalmerInfo
from .continuum import ContinuumInfo
from .convolution import ConvolutionInfo
from .cosmology import CosmologyInfo
from .error import ErrorInfo
from .host import HostInfo
from .iron import IronInfo
from .lines import LinesInfo
from .loading import LoadingInfo
from .mcmc import MCMCInfo
from .nonlinear import NonLinearInfo
from .random import RandomInfo
from .units import UnitsInfo

logger = getLogger(__name__)


@pydantic.dataclasses.dataclass
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
    cosmology: CosmologyInfo = field(
        default_factory=CosmologyInfo,
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
    mcmc: MCMCInfo = field(
        default_factory=MCMCInfo,
        kw_only=True,
    )
    nonlinear: NonLinearInfo = field(
        default_factory=NonLinearInfo,
        kw_only=True, 
    )
    random: RandomInfo = field(
        default_factory=RandomInfo,
        kw_only=True,
    )
    units: UnitsInfo = field(
        default_factory=UnitsInfo,
        kw_only=True, 
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
        "mcmc", 
        "nonlinear",
        "random",
        "units",
    ])

    def __post_init__(self) -> None:
        self.update()

    def _hash_helper(self) -> str:
        d = {
            k2: {k1: v1["value"] for k1, v1 in v2.items()}
            for k2, v2 in self.to_dict(jsonify=True).items()
        }
        return json_pkg.dumps(
            d, 
            sort_keys=True, 
            separators=(",", ":"),
            indent=None,
        )

    def create_hash(self, algo: str = "sha256") -> str:
        return hashlib.new(
            algo, 
            self._hash_helper().encode("utf-8"),
        ).hexdigest()

    def __hash__(self) -> int:
        return hash(self._hash_helper())

    def to_dict(
        self,
        jsonify: bool = False,
    ) -> dict[str, dict[str, Any]]:
        out = {}
        for key in sorted(self._keys):
            out.update(getattr(self, key).to_dict(jsonify=jsonify))
        return out

    @classmethod
    @pydantic.validate_call
    def from_json(
        cls,
        json: dict[str, dict] | JSONPath | None = None,
        create_copy: bool = True,
    ) -> Self:
    
        if json is None:
            return Info()

        if not isinstance(json, dict):
            with open(json, "r") as f:
                json = json_pkg.load(f)

        def func(cls):
            return cls.from_json(
                json=json, 
                create_copy=create_copy,
            )

        return Info(
            absorption=func(AbsorptionInfo),
            balmer=func(BalmerInfo),
            continuum=func(cls=ContinuumInfo),
            convolution=func(cls=ConvolutionInfo),
            cosmology=func(cls=CosmologyInfo),
            error=func(cls=ErrorInfo),
            host=func(cls=HostInfo),
            iron=func(cls=IronInfo),
            lines=func(cls=LinesInfo),
            loading=func(cls=LoadingInfo),
            mcmc=func(cls=MCMCInfo),
            nonlinear=func(cls=NonLinearInfo),
            random=func(cls=RandomInfo),
            units=func(cls=UnitsInfo),
        )

    @classmethod
    @pydantic.validate_call
    def from_yaml(
        cls,
        yaml: dict[str, dict] | YAMLPath | None = None,
        create_copy: bool = True,
    ) -> Self:
        import yaml as yaml_pkg

        if yaml is None:
            return Info()

        if not isinstance(yaml, dict):
            with open(yaml, "r") as f:
                yaml = yaml_pkg.load(f, yaml_pkg.Loader)

        def func(cls):
            return cls.from_json(
                json=yaml, 
                create_copy=create_copy,
            )

        return Info(
            absorption=func(AbsorptionInfo),
            balmer=func(BalmerInfo),
            continuum=func(cls=ContinuumInfo),
            convolution=func(cls=ConvolutionInfo),
            cosmology=func(cls=CosmologyInfo),
            error=func(cls=ErrorInfo),
            host=func(cls=HostInfo),
            iron=func(cls=IronInfo),
            lines=func(cls=LinesInfo),
            loading=func(cls=LoadingInfo),
            mcmc=func(cls=MCMCInfo),
            nonlinear=func(cls=NonLinearInfo),
            random=func(cls=RandomInfo),
            units=func(cls=UnitsInfo),
        )

    @pydantic.validate_call
    def to_json(self, path: AnyJSONPath) -> None:
        with open(path, "w") as f:
            json_pkg.dump(self.to_dict(jsonify=True), f, indent=4)

    @pydantic.validate_call
    def to_yaml(self, path: AnyYAMLPath) -> None:
        import yaml as yaml_pkg
        with open(path, "w") as f:
            yaml_pkg.dump(self.to_dict(jsonify=True), f, indent=4)

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
            "cosmology",
            "error",
            "host",
            "iron",
            "lines",
            "loading",
            "mcmc",
            "nonlinear",
            "random",
            "units",
        ],
    ) -> Any:
        result = None
        for subinfo in (getattr(self, subject) for subject in subjects):
            if key in subinfo._keys:
                result = getattr(subinfo, key)
                break

        return result