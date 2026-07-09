import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import pyc2ray.constants as c


class RadiationTables:
    NB1 = 1
    NB2 = 26
    NB3 = 20

    def __init__(
        self,
        ion_freq_HI: float = c.ev2hz * 13.598,
        ion_freq_HeI: float = c.ev2hz * 24.587,
        ion_freq_HeII: float = c.ev2hz * 54.416,
    ) -> None:
        ntot = self.NB1 + self.NB2 + self.NB3

        self.freq_max = np.zeros(ntot, dtype=np.float64)
        self.freq_min = np.zeros(ntot, dtype=np.float64)

        self.powerlax_index_HI = np.zeros(ntot, dtype=np.float64)
        self.powerlax_index_HeI = np.zeros(ntot, dtype=np.float64)
        self.powerlax_index_HeII = np.zeros(ntot, dtype=np.float64)

        self.cross_section_HI = np.zeros(ntot, dtype=np.float64)
        self.cross_section_HeI = np.zeros(ntot, dtype=np.float64)
        self.cross_section_HeII = np.zeros(ntot, dtype=np.float64)

        self.factors = np.zeros((ntot, 12), dtype=np.float64)

        # Define frequency bands
        band1 = slice(0, self.NB1)
        band2 = slice(self.NB1, self.NB1 + self.NB2)
        band3 = slice(self.NB1 + self.NB2, self.NB1 + self.NB2 + self.NB3)

        # --------
        # freq_min
        # --------
        self.freq_min[band1] = ion_freq_HI

        # fmt: off
        self.freq_min[band2] = [
            1.00, 1.02, 1.05, 1.07, 1.10,
            1.15, 1.20, 1.25, 1.30, 1.35,
            1.40, 1.45, 1.50, 1.55, 1.60,
            1.65, 1.70, 1.75, 1.80, 1.85,
            1.90, 1.95, 2.00, 2.05, 2.10,
            2.15,
        ]
        self.freq_min[band2] *= ion_freq_HeI

        self.freq_min[band3] = [
            1.00, 1.05, 1.10, 1.20, 1.40,
            1.70, 2.00, 2.50, 3.00, 4.00,
            5.00, 7.00, 10.00, 15.00, 20.00,
            30.00, 40.00, 50.00, 70.00, 90.00,
        ]
        self.freq_min[band3] *= ion_freq_HeII
        # fmt: on

        # --------
        # freq_max
        # --------
        self.freq_max[:-1] = self.freq_min[1:]
        self.freq_max[-1] = 100.0 * ion_freq_HeII

        # --------------
        # cross-sections
        # --------------
        self.cross_section_HI[band1] = 6.346e-18
        self.cross_section_HeI[band1] = 0.0
        self.cross_section_HeII[band1] = 0.0

        # fmt: off
        self.cross_section_HI[band2] = [
            1.239152e-18, 1.171908e-18, 1.079235e-18, 1.023159e-18, 9.455687e-19,
            8.329840e-19, 7.374876e-19, 6.559608e-19, 5.859440e-19, 5.254793e-19,
            4.729953e-19, 4.272207e-19, 3.874251e-19, 3.521112e-19, 3.209244e-19,
            2.932810e-19, 2.686933e-19, 2.467523e-19, 2.271125e-19, 2.094813e-19,
            1.936094e-19, 1.792838e-19, 1.663215e-19, 1.545649e-19, 1.438778e-19,
            1.341418e-19,
        ]
        self.cross_section_HeI[band2] = [
            7.434699e-18, 7.210641e-18, 6.887151e-18, 6.682491e-18, 6.387263e-18,
            5.931487e-18, 5.516179e-18, 5.137743e-18, 4.792724e-18, 4.477877e-18,
            4.190200e-18, 3.926951e-18, 3.687526e-18, 3.465785e-18, 3.261781e-18,
            3.073737e-18, 2.900074e-18, 2.739394e-18, 2.590455e-18, 2.452158e-18,
            2.323526e-18, 2.203694e-18, 2.091889e-18, 1.987425e-18, 1.889687e-18,
            1.798126e-18,
        ]
        self.cross_section_HeII[band2] = 0.0

        self.cross_section_HI[band3] = [
            1.230696e-19, 1.063780e-19, 9.253883e-20, 7.123014e-20, 4.464019e-20,
            2.465533e-20, 1.492667e-20, 7.446712e-21, 4.196728e-21, 1.682670e-21,
            8.223247e-22, 2.763830e-22, 8.591126e-23, 2.244684e-23, 8.593853e-24,
            2.199718e-24, 8.315674e-25, 3.898672e-25, 1.238718e-25, 5.244957e-26,
        ]
        self.cross_section_HeI[band3] = [
            1.690781e-18, 1.521636e-18, 1.373651e-18, 1.128867e-18, 7.845096e-19,
            4.825331e-19, 3.142134e-19, 1.696228e-19, 1.005051e-19, 4.278712e-20,
            2.165403e-20, 7.574790e-21, 2.429426e-21, 6.519748e-22, 2.534069e-22,
            6.599821e-23, 2.520412e-23, 1.189810e-23, 3.814490e-24, 1.624492e-24,
        ]
        self.cross_section_HeII[band3] = [
            1.587280e-18, 1.391911e-18, 1.227391e-18, 9.686899e-19, 6.338284e-19,
            3.687895e-19, 2.328072e-19, 1.226873e-19, 7.214988e-20, 3.081577e-20,
            1.576429e-20, 5.646276e-21, 1.864734e-21, 5.177347e-22, 2.059271e-22,
            5.526508e-23, 2.151467e-23, 1.029637e-23, 3.363164e-24, 1.450239e-24,
        ]
        # fmt: on

        # -----------------
        # power-law indices
        # -----------------
        self.powerlax_index_HI[band1] = 2.761

        # fmt: off
        self.powerlax_index_HI[band2] = [
            2.8277, 2.8330, 2.8382, 2.8432, 2.8509,
            2.8601, 2.8688, 2.8771, 2.8850, 2.8925,
            2.8997, 2.9066, 2.9132, 2.9196, 2.9257,
            2.9316, 2.9373, 2.9428, 2.9481, 2.9532,
            2.9582, 2.9630, 2.9677, 2.9722, 2.9766,
            2.9813,
        ]
        self.powerlax_index_HeI[band2] = [
            1.5509, 1.5785, 1.6047, 1.6290, 1.6649,
            1.7051, 1.7405, 1.7719, 1.8000, 1.8253,
            1.8486, 1.8701, 1.8904, 1.9098, 1.9287,
            1.9472, 1.9654, 1.9835, 2.0016, 2.0196,
            2.0376, 2.0557, 2.0738, 2.0919, 2.1099,
            2.1302,
        ]

        self.powerlax_index_HI[band3] = [
            2.9884, 2.9970, 3.0088, 3.0298, 3.0589,
            3.0872, 3.1166, 3.1455, 3.1773, 3.2089,
            3.2410, 3.2765, 3.3107, 3.3376, 3.3613,
            3.3816, 3.3948, 3.4078, 3.4197, 3.4379,
        ]
        self.powerlax_index_HeI[band3] = [
            2.1612, 2.2001, 2.2564, 2.3601, 2.5054,
            2.6397, 2.7642, 2.8714, 2.9700, 3.0528,
            3.1229, 3.1892, 3.2451, 3.2853, 3.3187,
            3.3464, 3.3640, 3.3811, 3.3967, 3.4203,
        ]
        self.powerlax_index_HeII[band3] = [
            2.6930, 2.7049, 2.7213, 2.7503, 2.7906,
            2.8300, 2.8711, 2.9121, 2.9577, 3.0041,
            3.0522, 3.1069, 3.1612, 3.2051, 3.2448,
            3.2796, 3.3027, 3.3258, 3.3472, 3.3805,
        ]
        # fmt: on

        # --------------------
        # heating form factors
        # --------------------

        F1ION_HI, F1ION_HEI, F1ION_HEII = 0, 1, 2
        F2ION_HI, F2ION_HEI, F2ION_HEII = 3, 4, 5
        F1HEAT_HI, F1HEAT_HEI, F1HEAT_HEII = 6, 7, 8
        F2HEAT_HI, F2HEAT_HEI, F2HEAT_HEII = 9, 10, 11

        # fmt: off
        self.factors[band2, F1ION_HI] = [
            0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
            0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
            0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
            0.0000, 1.0000, 1.0000, 1.0000, 1.0000,
            1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
            1.0000,
        ]
        self.factors[band2, F1ION_HEI] = [
            0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
            0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
            0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
            0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
            0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
            1.0000,
        ]
        self.factors[band2, F1ION_HEII] = [
            0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
            0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
            0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
            0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
            0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
            0.0000,
        ]
        self.factors[band2, F2ION_HI] = [
            0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
            0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
            0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
            0.0000, 0.9971, 0.9802, 0.9643, 0.9493,
            0.9350, 0.9215, 0.9086, 0.8964, 0.8847,
            0.8735,
        ]
        self.factors[band2, F2ION_HEI] = [
            0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
            0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
            0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
            0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
            0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
            0.9960,
        ]
        self.factors[band2, F2ION_HEII] = [
            0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
            0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
            0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
            0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
            0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
            0.0000,
        ]
        self.factors[band2, F1HEAT_HI] = [
            0.0000, 1.0000, 1.0000, 1.0000, 1.0000,
            1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
            1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
            1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
            1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
            1.0000,
        ]
        self.factors[band2, F1HEAT_HEI] = [
            0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
            0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
            0.0000, 1.0000, 1.0000, 1.0000, 1.0000,
            1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
            1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
            1.0000,
        ]
        self.factors[band2, F1HEAT_HEII] = [
            0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
            0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
            0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
            0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
            0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
            0.0000,
        ]
        self.factors[band2, F2HEAT_HI] = [
            0.0000, 0.9704, 0.9290, 0.9037, 0.8687,
            0.8171, 0.7724, 0.7332, 0.6985, 0.6675,
            0.6397, 0.6145, 0.5916, 0.5707, 0.5514,
            0.5337, 0.5173, 0.5021, 0.4879, 0.4747,
            0.4623, 0.4506, 0.4397, 0.4293, 0.4196,
            0.4103,
        ]
        self.factors[band2, F2HEAT_HEI] = [
            0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
            0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
            0.0000, 0.9959, 0.9250, 0.8653, 0.8142,
            0.7698, 0.7309, 0.6965, 0.6657, 0.6380,
            0.6130, 0.5903, 0.5694, 0.5503, 0.5327,
            0.5164,
        ]
        self.factors[band2, F2HEAT_HEII] = [
            0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
            0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
            0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
            0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
            0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
            0.0000,
        ]

        self.factors[band3, F1ION_HI] = [
            1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
            1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
            1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
            1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
        ]
        self.factors[band3, F1ION_HEI] = [
            1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
            1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
            1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
            1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
        ]
        self.factors[band3, F1ION_HEII] = [
            0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
            1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
            1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
            1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
        ]
        self.factors[band3, F2ION_HI] = [
            0.8600, 0.8381, 0.8180, 0.7824, 0.7249,
            0.6607, 0.6128, 0.5542, 0.5115, 0.4518,
            0.4110, 0.3571, 0.3083, 0.2612, 0.2325,
            0.1973, 0.1757, 0.1606, 0.1403, 0.1269,
        ]
        self.factors[band3, F2ION_HEI] = [
            0.9750, 0.9415, 0.9118, 0.8609, 0.7831,
            0.7015, 0.6436, 0.5755, 0.5273, 0.4619,
            0.4182, 0.3615, 0.3109, 0.2627, 0.2334,
            0.1979, 0.1761, 0.1609, 0.1405, 0.1270,
        ]
        self.factors[band3, F2ION_HEII] = [
            0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
            0.8841, 0.7666, 0.6518, 0.5810, 0.4940,
            0.4403, 0.3744, 0.3183, 0.2668, 0.2361,
            0.1993, 0.1771, 0.1616, 0.1409, 0.1273,
        ]
        self.factors[band3, F1HEAT_HI] = [
            1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
            1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
            1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
            1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
        ]
        self.factors[band3, F1HEAT_HEI] = [
            1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
            1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
            1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
            1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
        ]
        self.factors[band3, F1HEAT_HEII] = [
            0.0000, 0.0000, 0.0000, 0.0000, 1.0000,
            1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
            1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
            1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
        ]
        self.factors[band3, F2HEAT_HI] = [
            0.3994, 0.3817, 0.3659, 0.3385, 0.2961,
            0.2517, 0.2207, 0.1851, 0.1608, 0.1295,
            0.1097, 0.0858, 0.0663, 0.0496, 0.0405,
            0.0304, 0.0248, 0.0212, 0.0167, 0.0140,
        ]
        self.factors[band3, F2HEAT_HEI] = [
            0.4974, 0.4679, 0.4424, 0.4001, 0.3389,
            0.2796, 0.2405, 0.1977, 0.1697, 0.1346,
            0.1131, 0.0876, 0.0673, 0.0501, 0.0408,
            0.0305, 0.0249, 0.0213, 0.0168, 0.0140,
        ]
        self.factors[band3, F2HEAT_HEII] = [
            0.0000, 0.0000, 0.0000, 0.0000, 0.6202,
            0.4192, 0.3265, 0.2459, 0.2010, 0.1513,
            0.1237, 0.0932, 0.0701, 0.0515, 0.0416,
            0.0309, 0.0251, 0.0214, 0.0169, 0.0141,
        ]
        # fmt: on

    @property
    def freqs(self) -> tuple[np.ndarray, np.ndarray]:
        return self.freq_min, self.freq_max

    @property
    def cross_sections(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.cross_section_HI, self.cross_section_HeI, self.cross_section_HeII

    @property
    def powerlaw_indices(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (
            self.powerlax_index_HI,
            self.powerlax_index_HeI,
            self.powerlax_index_HeII,
        )

    @property
    def ion_freq_HI(self) -> float:
        return self.freq_min[0]

    @property
    def ion_freq_HeI(self) -> float:
        return self.freq_min[self.NB1]

    @property
    def ion_freq_HeII(self) -> float:
        return self.freq_min[self.NB1 + self.NB2]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--print",
        action="store_true",
        help="Print cross-sections, power indices and heating factors",
    )
    group.add_argument(
        "--save-txt",
        type=Path,
        metavar="DIR",
        help="Save cross-sections, power indices and photo heating factors "
        "to text files in %(metavar)s",
    )
    group.add_argument(
        "--save-numpy",
        type=Path,
        metavar="DIR",
        help="Save cross-sections, power indices and photo heating factors "
        "to numpy files in %(metavar)s",
    )
    group.add_argument(
        "--plot", action="store_true", help="Plot the cross-section and power indices"
    )
    group.add_argument(
        "--save-plot",
        type=Path,
        metavar="FILENAME",
        help="Save the plot of the cross-section and power indices to %(metavar)s",
    )

    args = parser.parse_args()
    if not any(v for v in vars(args).values()):
        print("Please specify exactly one option!\n")
        parser.print_help()
        sys.exit(1)

    rt = RadiationTables()

    if args.print:
        print("Cross sections")
        print("fmin\tfmax\tsigma_HI\tsigma_HeI\tsigma_HeII")
        for fmin, fmax, sHI, sHeI, sHeII in zip(*rt.freqs, *rt.cross_sections):
            print(f"{fmin:.6e}\t{fmax:.6e}\t{sHI:.8e}\t{sHeI:.8e}\t{sHeII:.8e}")
        print("\nPower law indices")
        print("fmin\tfmax\tpl_index_HI\tpl_index_HeI\tpl_index_HeII")
        for fmin, fmax, plHI, plHeI, plHeII in zip(*rt.freqs, *rt.powerlaw_indices):
            print(f"{fmin:.6e}\t{fmax:.6e}\t{plHI:.5f}\t{plHeI:.5f}\t{plHeII:.5f}")

        print("\nForm factors for photo heating rates")
        print(
            "fmin\tfmax\tf1ion(HI, HeI, HeII), f2ion(HI, HeI, HeII), f1heat(HI, HeI, HeII), f2heat(HI, HeI, HeII)"
        )
        for fmin, fmax, facts in zip(*rt.freqs, rt.factors):
            print(f"{fmin:.6e}\t{fmax:.6e}\t{facts.tolist()}")

    if args.save_txt is not None:
        args.save_txt.mkdir(parents=True, exist_ok=True)

        print("Saving cross-sections in", args.save_txt / "Verner1996_crossect.txt")
        np.savetxt(
            args.save_txt / "Verner1996_crossect.txt",
            np.array([*rt.freqs, *rt.cross_sections]).T,
            fmt="%.10e\t%.10e\t%.10e\t%.10e\t%.10e",
            header="fmin [Hz]\t\tfmax [Hz]\t\tsigma_HI\tsigma_HeI\tsigma_HeII",
        )

        print("Saving power indices in", args.save_txt / "Verner1996_spectidx.txt")
        np.savetxt(
            args.save_txt / "Verner1996_spectidx.txt",
            np.array([*rt.freqs, *rt.powerlaw_indices]).T,
            fmt="%.10e\t%.10e\t%.5e\t%.5e\t%.5e",
            header="fmin [Hz]\t\tfmax [Hz]\t\tpower index HI\tpower index HeI\tpower index HeII",
        )

        print(
            "Saving photo heating factors in",
            args.save_txt / "photo_heating_factors.txt",
        )
        np.savetxt(
            args.save_txt / "photo_heating_factors.txt",
            rt.factors,
            fmt="%.5e",
            header="f1ionHI\tf1ionHeI\tf1ionHeII\tf2ionHI\tf2ionHeI\tf2ionHeII\t"
            "f1heatHI\tf1heatHeI\tf1heatHeII\tf2heatHI\tf2heatHeI\tf2heatHeII",
        )

    if args.save_numpy is not None:
        args.save_numpy.mkdir(parents=True, exist_ok=True)

        print("Saving cross-sections in", args.save_numpy / "Verner1996_crossect.npz")
        np.savez(
            args.save_numpy / "Verner1996_crossect.npz",
            fmin=rt.freqs[0],
            fmax=rt.freqs[1],
            sigma_HI=rt.cross_sections[0],
            sigma_HeI=rt.cross_sections[1],
            sigma_HeII=rt.cross_sections[2],
        )

        print("Saving power indices in", args.save_numpy / "Verner1996_spectidx.npz")
        np.savez(
            args.save_numpy / "Verner1996_spectidx.npz",
            fmin=rt.freqs[0],
            fmax=rt.freqs[1],
            pl_index_HI=rt.powerlaw_indices[0],
            pl_index_HeI=rt.powerlaw_indices[1],
            pl_index_HeII=rt.powerlaw_indices[2],
        )

        print(
            "Saving photo heating factors in",
            args.save_numpy / "photo_heating_factors.npy",
        )
        np.save(args.save_numpy / "photo_heating_factors.npy", rt.factors)

    if args.plot or args.save_plot is not None:
        fig, axs = plt.subplots(figsize=(16, 7), ncols=2, nrows=1)
        colors = ["blue", "red", "lime"]
        label = ["HI", "HeI", "HeII"]

        fig.suptitle("Spectral Index Cross-Section (Verner+ 1996)")
        fig.tight_layout(pad=3.0)

        for i, pl in enumerate(rt.powerlaw_indices):
            axs[0].plot(rt.freqs[0], pl, label=label[i], color=colors[i], marker="x")

        axs[0].axvline(rt.ion_freq_HI, color="blue", ls="--")
        axs[0].axvline(rt.ion_freq_HeI, color="red", ls="--")
        axs[0].axvline(rt.ion_freq_HeII, color="lime", ls="--")
        axs[0].set_xlabel(r"$\nu$ [Hz]")
        axs[0].set_ylabel("Power law index")
        axs[0].set_xscale("log")
        axs[0].legend()

        for i, sig in enumerate(rt.cross_sections):
            axs[1].plot(rt.freqs[0], sig, label=label[i], color=colors[i], marker="x")
        axs[1].set_xlabel(r"$\nu$ [Hz]")
        axs[1].set_ylabel(r"$\sigma_\nu$ $[cm^2]$")
        axs[1].set_xscale("log")
        axs[1].set_yscale("log")
        axs[1].legend()

        if args.save_plot is not None:
            plt.savefig(args.save_plot / "tables.png", dpi=300)
        plt.show()
