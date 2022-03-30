from itertools import product

import numpy as np
import matplotlib.pyplot as plt


def make_phase_portrait(
    ens, *phase_portraits_neurons, p0_lims=(0, 2 * np.pi), p0_count=20, dt=0.1, t=100
):
    n = 2

    p0_points = np.linspace(*p0_lims, p0_count)

    fig, axs = plt.subplots(1, len(phase_portraits_neurons))
    if len(phase_portraits_neurons) == 1:
        axs = np.array([axs])

    for ax in axs:
        ax.set_xlim(*p0_lims)
        ax.set_ylim(*p0_lims)

    for p0_x, p0_y in product(p0_points, p0_points):
        ens.update_neurons({0: dict(phase_0=p0_x), 1: dict(phase_0=p0_y)})

        ens_result = ens.compute(dt, t)

        for phase_portrait_neurons, ax in zip(phase_portraits_neurons, axs):
            x = ens_result[phase_portrait_neurons[0]]
            y = ens_result[phase_portrait_neurons[1]]

            ax.quiver(
                x[:-1:n],
                y[:-1:n],
                x[1::n] - x[:-1:n],
                y[1::n] - y[:-1:n],
                scale_units="xy",
                angles="xy",
                scale=1,
            )

    return fig, axs
