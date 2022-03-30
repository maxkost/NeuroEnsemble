import numpy as np

from ensemble import Neuron, Connection

from ..neuro_ensemble import NeuroEnsembleBase, NeuroEnsembleResult

class CPG_2N(NeuroEnsembleBase):
    def __init__(self, w1, w2, p0_1=0, p0_2=0, d=0, sigma=0, k=-500):
        
        self._connections_data = [
            {
                "in": 0,
                "out": 1,
                "force": d,
            },
            {
                "in": 1,
                "out": 0,
                "force": d,
            }
        ]

        self._neurons_data = [
            {
                "phase_0": p0_1,
                "natural_freq": w1
            },
            {
                "phase_0": p0_2,
                "natural_freq": w2
            }
        ]

        super().__init__(sigma, k)
