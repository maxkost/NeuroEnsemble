from functools import cached_property
from itertools import product
from re import L

import numpy as np
from joblib import Memory
from matplotlib import pyplot as plt

from ensemble import Ensemble, Neuron, Connection


class NeuroEnsembleResult:
    def __init__(self, result_data):
        t, neurons_points = result_data

        self._t = np.array(t)
        self._dt = t[1] - t[0]
        self._rotation_start = int(len(t) * 0.2)
        self._neurons_points = np.array(
            [np.array(neuron_points) for neuron_points in neurons_points]
        )

    @cached_property
    def t(self):
        return self._t.copy()

    @cached_property
    def neurons(self):
        return self._neurons_points.copy()

    def __getitem__(self, index):
        def adjust(index):
            if index is None:
                return None
            return int(index / self._dt)

        def apply_to_neurons_points(index):
            return np.array([neuron_points[index] for neuron_points in self._neurons_points])

        if isinstance(index, slice):
            index = slice(adjust(index.start), adjust(index.stop), adjust(index.step))
            return NeuroEnsembleResult((self._t[index], apply_to_neurons_points(index)))
        elif isinstance(index, int):
            index = adjust(index)
            return self._t[index], apply_to_neurons_points(index)
        else:
            raise TypeError

    @staticmethod
    def _rotation_number(t, x, rotation_start):
        return (x[-1] - x[rotation_start]) / (t[-1] - t[rotation_start])

    @cached_property
    def rotations(self):
        return np.fromiter(
            map(
                lambda x: self._rotation_number(self.t, x, self._rotation_start),
                self._neurons_points,
            ),
            dtype=np.float64,
        )

    def plot(self):
        func = np.vectorize(lambda x: x % (2 * np.pi))
        for neuron_points in self._neurons_points:
            plt.plot(self._t, func(neuron_points))


class NeuroEnsembleBase:
    _neurons_data = []
    _connections_data = []

    __neurons = []
    __connections = []

    __ensemble = None

    def __init__(self, sigma, k=-500):
        self._sigma = sigma
        self._k = k

        self._init_neurons()
        self._init_connections()

        self._init_ensemble()

    @staticmethod
    def __make_neurons(neurons_data):
        return [
            Neuron(neuron_data["phase_0"], neuron_data["natural_freq"])
            for neuron_data in neurons_data
        ]

    def _init_neurons(self):
        self.__neurons = self.__make_neurons(self._neurons_data)

    def update_neurons(self, neurons_new_data):
        for neuron_index, neuron_new_data in neurons_new_data.items():
            self._neurons_data[neuron_index].update(neuron_new_data)

        self._init_neurons()
        self._init_ensemble()

    @staticmethod
    def __make_connections(connections_data):
        return [
            Connection(
                (connection_data["out"], connection_data["in"]),
                connection_data["force"],
            )
            for connection_data in connections_data
        ]

    def _init_connections(self):
        self.__connections = self.__make_connections(self._connections_data)

    def _update_connections(self, connections_new_data):
        for connection_index, connection_new_data in connections_new_data:
            self._connections_data[connection_index].update(connection_new_data)

        self._init_connections()
        self._init_ensemble()

    @staticmethod
    def __make_ensemble(neurons, connections):
        return Ensemble(neurons, connections)

    def _init_ensemble(self):
        self.__ensemble = self.__make_ensemble(self.__neurons, self.__connections)

    @staticmethod
    def __compute(ens, neurons_data, connections_data, sigma, k, t_step, t_end):
        return ens.compute(sigma, k, t_end, t_step)

    def compute(self, t_step: float, t_end: float):
        result_data = self.__compute(
            self.__ensemble,
            self._neurons_data,
            self._connections_data,
            self._sigma,
            self._k,
            t_step,
            t_end,
        )

        return NeuroEnsembleResult(result_data)

    
    def memoize(self, memory: Memory):
        self.__compute = memory.cache(self.__class__.__compute, ignore=["ens"])
