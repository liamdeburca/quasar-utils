from collections.abc import Callable
from dataclasses import field
from typing import Literal

from pydantic.dataclasses import dataclass
from quasar_typing.astropy import FitInfo, Model_
from quasar_typing.numpy import FittableFloatVector

from quasar_utils.fitting import DogBoxLSQFitter, LMLSQFitter, TRFLSQFitter


@dataclass
class FitterInstance:
    algo_name: Literal["trf", "dogbox", "lm"]

    loss: str = field(kw_only=True)
    maxiter: int = field(kw_only=True)
    ftol: float = field(kw_only=True)
    xtol: float = field(kw_only=True)
    gtol: float = field(kw_only=True)
    f_scale: float = field(kw_only=True)

    algo: Callable[
        [
            Model_,
            FittableFloatVector,
            FittableFloatVector,
            FittableFloatVector,
            bool,
        ],
        Model_,
    ] = field(init=False)

    kwargs: dict[str, float | int | str] = field(init=False)

    def __post_init__(self):
        match self.algo_name:
            case "trf":
                self.algo = TRFLSQFitter(calc_uncertainties=True)
            case "dogbox":
                self.algo = DogBoxLSQFitter(calc_uncertainties=True)
            case "lm":
                self.algo = LMLSQFitter(calc_uncertainties=True)

        self.kwargs = dict(
            loss=self.loss,
            maxiter=self.maxiter,
            ftol=self.ftol,
            xtol=self.xtol,
            gtol=self.gtol,
            f_scale=self.f_scale,
        )

    def __call__(
        self,
        model: Model_,
        x: FittableFloatVector,
        y: FittableFloatVector,
        dy: FittableFloatVector,
        inplace: bool = False,
    ) -> tuple[Model_, FitInfo]:
        fit = self.algo(
            model,
            x,
            y,
            weights=1 / dy,
            inplace=inplace,
            **self.kwargs,
        )
        return fit, self.algo.fit_info

    def __getstate__(self) -> dict:
        return {
            "algo_name": self.algo_name,
            "loss": self.loss,
            "maxiter": self.maxiter,
            "ftol": self.ftol,
            "xtol": self.xtol,
            "gtol": self.gtol,
            "f_scale": self.f_scale,
        }

    def __setstate__(self, state: dict) -> None:
        self.algo_name = state["algo_name"]
        self.loss = state["loss"]
        self.maxiter = state["maxiter"]
        self.ftol = state["ftol"]
        self.xtol = state["xtol"]
        self.gtol = state["gtol"]
        self.f_scale = state["f_scale"]
        self.__post_init__()

    def __hash__(self) -> int:
        return hash(tuple(self.__getstate__().values()))
