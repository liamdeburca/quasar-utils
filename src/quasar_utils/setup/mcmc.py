from logging import getLogger
from typing import Any, Literal, Self, TypedDict

from pydantic.dataclasses import dataclass
from quasar_typing.misc import MCMCSamplingMethod
from quasar_typing.pathlib import AbsoluteFilePath

from ..decorators import validate_call
from .utils import _Info, field, finalise_dataclass

logger = getLogger(__name__)


class MCMCKwargs(TypedDict):
    n_walkers: int
    n_walkers_per_dim: int
    max_nfev: int
    batch_size: int
    ftol: float | None
    xtol: float | None
    max_attempts: int


@finalise_dataclass(additional_keys=["mcmc_kwargs"])
@dataclass
class MCMCInfo(_Info):
    max_nfev: int = field(
        desc="Maximum number of function evaluations for MCMC (per walker)",
        default=1000,
        dtype="int",
        parse_as="int",
    )
    batch_size: int = field(
        desc="Batch size for MCMC sampling",
        default=100,
        dtype="int",
        parse_as="int",
    )
    xtol: float | None = field(
        desc="Parameter tolerance for early termination of MCMC sampling",
        default=1e-4,
        dtype="float | None",
        parse_as="optional_float",
    )
    ftol: float | None = field(
        desc="Loss function tolerance for early termination of MCMC sampling",
        default=1e-4,
        dtype="float | None",
        parse_as="optional_float",
    )
    n_walkers: int = field(
        desc="Default number of walkers for MCMC sampling",
        default=100,
        dtype="int",
        parse_as="int",
    )
    n_walkers_per_dim: int = field(
        desc="Number of walkers per dimension for MCMC sampling",
        default=10,
        dtype="int",
        parse_as="int",
    )
    sampling_method: MCMCSamplingMethod = field(
        desc="Method used to initialise MCMC walkers",
        default="fisher",
        dtype="Literal['fisher', 'uniform']",
        parse_as="str",
    )
    max_attempts: int = field(
        desc="Maximum no. of attempts for finding valid initial states",
        default=1000,
        dtype="int",
        parse_as="int",
    )
    renew_rng: bool = field(
        default=True,
        dtype="bool",
        parse_as="bool",
    )

    @property
    def mcmc_kwargs(self) -> MCMCKwargs:
        return {
            'n_walkers': self.n_walkers,
            'n_walkers_per_dim': self.n_walkers_per_dim,
            'max_nfev': self.max_nfev,
            'batch_size': self.batch_size,
            'ftol': self.ftol,
            'xtol': self.xtol,
            'sampling_method': self.sampling_method,
            'max_attempts': self.max_attempts,
        }

    def __hash__(self) -> int:
        return super().__hash__()

    def __getitem__(self, key: str) -> Any:
        if key == "mcmc_kwargs":
            return self.mcmc_kwargs
        return super().__getitem__(key)

    def __getstate__(self) -> dict:
        state = super().__getstate__()
        state.pop("mcmc_kwargs", None)
        return state
    
    def update(self, info) -> None:
        super().update(info, logger)

    def to_dict(
        self,
        jsonify: bool = False,
    ) -> dict[Literal["mcmc"], dict[str, Any]]:
        return super().to_dict("mcmc", jsonify=jsonify)

    @classmethod
    @validate_call
    def from_json(
        cls,
        json: dict[str, dict] | AbsoluteFilePath | None = None,
        create_copy: bool = True,
    ) -> Self:
        return cls._from_json(json, create_copy, "mcmc", logger)
