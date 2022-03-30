from ..neuro_ensemble import NeuroEnsembleBase, NeuroEnsembleResult


class Double_Circle_3N(NeuroEnsembleBase):
    def __init__(
        self,
        w1,
        w2,
        w3,
        phi0_1=0,
        phi0_2=0,
        phi0_3=0,
        d1=0,
        d2=0,
        sigma=0,
        k=-500,
    ):
        self._w1 = w1
        self._w2 = w2
        self._w3 = w3

        self._phi0_1 = phi0_1
        self._phi0_2 = phi0_2
        self._phi0_3 = phi0_3

        self._d1 = d1
        self._d2 = d2

        self._make_neurons_data()
        self._make_connections_data()

        super().__init__(sigma, k)

    def _make_neurons_data(self):
        self._neurons_data = [
            {"phase_0": self._phi0_1, "natural_freq": self._w1},
            {"phase_0": self._phi0_2, "natural_freq": self._w2},
            {"phase_0": self._phi0_3, "natural_freq": self._w3},
        ]

    def _make_connections_data(self):
        self._connections_data = [
            *(
                {
                    "in": n_in,
                    "out": n_out,
                    "force": self._d1,
                }
                for n_in, n_out in [(0, 1), (1, 2), (2, 0)]
            ),
            *(
                {
                    "in": n_in,
                    "out": n_out,
                    "force": self._d2,
                }
                for n_in, n_out in [(0, 2), (2, 1), (1, 0)]
            ),
        ]

    def update_circles(self, d1=None, d2=None):
        if d1 is not None:
            self._d1 = d1
        if d2 is not None:
            self._d2 = d2

        self._make_connections_data()

        self._init_connections()
        self._init_ensemble()

    def update_neurons_starts(self, phi0_1=None, phi0_2=None, phi0_3=None):
        if phi0_1 is not None:
            self._phi0_1 = phi0_1
        if phi0_2 is not None:
            self._phi0_2 = phi0_2
        if phi0_3 is not None:
            self._phi0_3 = phi0_3

        self._make_neurons_data()

        self._init_neurons()
        self._init_ensemble()