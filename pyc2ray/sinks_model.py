from collections.abc import Callable
from pathlib import Path

import numpy as np
import numpy.typing as npt

from pyc2ray.parameters import (
    ConstantClumpingParameters,
    ConstantMfpParameters,
    DensityClumpingParameters,
    RedshiftClumpingParameters,
    SinksParameters,
    StochasticClumpingParameters,
    Worseck14MfpParameters,
)
from pyc2ray.utils.other_utils import find_bins

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int32]


class SinksPhysics:
    def __init__(
        self, sinks_params: SinksParameters, meshsize: int, boxsize: float
    ) -> None:
        self.clumping_model = sinks_params.clumping_model
        self.mfp_model = sinks_params.mfp_model
        self.N = meshsize

        res = boxsize / self.N

        # MFP parameters
        if isinstance(sinks_params.mfp, ConstantMfpParameters):
            # Set R_max (LLS 3) in cell units
            self.R_mfp_cell_unit = sinks_params.mfp.R_max_cMpc / res
        elif isinstance(sinks_params.mfp, Worseck14MfpParameters):
            self.A_mfp = sinks_params.mfp.A_mfp
            self.etha_mfp = sinks_params.mfp.eta_mfp
            self.z1_mfp = sinks_params.mfp.z1_mfp
            self.eta1_mfp = sinks_params.mfp.eta1_mfp
        else:
            raise TypeError(f"MFP model not implemented: {self.mfp_model}")

        self.clumping_factor: FloatArray
        # Clumping factor parameters
        if isinstance(sinks_params.clumping, ConstantClumpingParameters):
            self.clumping_factor = np.full(
                (self.N, self.N, self.N), sinks_params.clumping.value, dtype=np.float64
            )
        else:
            clump_dir = Path(__file__).parent / "tables" / "clumping"
            self.model_res = np.loadtxt(clump_dir / "resolutions.txt")

            # use parameters from tables with similare spatial resolution
            tab_res = self.model_res[
                np.argmin(np.abs(self.model_res * 0.7 - res))
            ]  # the tables where calculated for cMpc/h with h=0.7

            # get parameter files
            self.clumping_params = np.loadtxt(
                clump_dir / f"par_{self.clumping_model}_{tab_res:.3f}Mpc.txt"
            )
            self.calculate_clumping: Callable[..., np.ndarray]
            if isinstance(sinks_params.clumping, RedshiftClumpingParameters):
                self.c2, self.c1, self.C0 = self.clumping_params[:3]
                self.calculate_clumping = self.biashomogeneous_clumping
            elif isinstance(sinks_params.clumping, DensityClumpingParameters):
                self.calculate_clumping = self.inhomogeneous_clumping
            elif isinstance(sinks_params.clumping, StochasticClumpingParameters):
                self.calculate_clumping = self.stochastic_clumping
            else:
                raise TypeError(
                    " Clumping factor model not implemented : %s" % self.clumping_model
                )

    def mfp_Worseck14(self, z: float) -> float:
        assert self.A_mfp is not None
        assert self.etha_mfp is not None
        assert self.eta1_mfp is not None
        assert self.z1_mfp is not None
        R_mfp = self.A_mfp * ((1 + z) / 5.0) ** self.etha_mfp
        R_mfp = R_mfp * (1 + ((1 + z) / (1 + self.z1_mfp)) ** self.eta1_mfp)
        return R_mfp

    def biashomogeneous_clumping(self, z: float) -> FloatArray:
        clump_fact = self.C0 * np.exp(self.c1 * z + self.c2 * z**2) + 1.0
        return np.full((self.N, self.N, self.N), clump_fact, dtype=np.float64)

    def inhomogeneous_clumping(self, z: float, ndens: FloatArray) -> FloatArray:
        redshift = self.clumping_params[:, 0]

        # find nearest redshift bin
        zlow, zhigh = find_bins(z, redshift)
        i_low, i_high = np.digitize(zlow, redshift), np.digitize(zhigh, redshift)

        # calculate weight to
        w_l, w_h = 1 - (z - zlow) / (zhigh - zlow), 1 - (zhigh - z) / (zhigh - zlow)

        # get parameters weighted
        a, b, c = (
            self.clumping_params[i_low, 1:4] * w_l
            + self.clumping_params[i_high, 1:4] * w_h
        )

        # MB (22.10.24): In the original paper, Bianco+ ('21), we used to do the fit in log-space log10(1+<delta>) VS log10(C). Later, and in the current parameters files we did the fit in the linear-space.
        x = 1 + ndens / ndens.mean()
        clump_fact = a * x**2 + b * x + c

        return np.clip(clump_fact, 1.0, clump_fact.max())

    def stochastic_clumping(self, z, ndens):
        # TODO: implement
        # MaxBin = 5
        # lognormParamsFile = pd.read_csv(
        #     "par_stochastic_2.024Mpc.csv",
        #     index_col=0,
        #     converters={
        #         "bin%d" % i: lambda string: np.array(
        #             string[1:-1].split(", "), dtype=float
        #         )
        #         for i in range(MaxBin)
        #     },
        # )

        return 0
