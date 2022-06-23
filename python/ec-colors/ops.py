#!/usr/bin/env python3

"""
Author:     Jacopo Di Matteo
Date:       18.06.2022

This code is provided "As Is"
"""

# Third-party
from turtle import left
import matplotlib.pyplot as plt
import numpy as np
from numba import njit
from tqdm import tqdm

# Correction term for 0 division
EPS = 1 * (10**-9)


@njit
def correct_num_points(num_points: int) -> tuple[int, int]:
    """Round number of points to closest number which can be square rooted
    Input
        num_points: value to match to a square-root-able number

    Output: closest_square (squared), closest_square
    """
    grid_size = int(np.ceil(np.sqrt(num_points)))
    new_num_points = grid_size**2
    return grid_size, new_num_points


@njit
def get_xy_coord(grid_size: int) -> np.array:
    """Get (x, y) coordinates of every pixel in grid (origin is top left)
    Input
        grid_size: length of side (expects square)

    Output: np.array with coords at each position
    """
    return np.array([[i, j] for j in range(grid_size) for i in range(grid_size)])


@njit
def basic_distance(points: np.array, grid_size: int) -> np.array:
    """Find distance between points in a square grid
    Input
        points: array of coordinates
        grid_size: length of side (expects square)

    Output: distances between each single point in an image
    """
    xx = np.empty(shape=(grid_size**2, grid_size, grid_size), dtype=np.float32)
    for n, point in enumerate(points):
        x, y = int(point[0]), int(point[1])
        for i, ii in enumerate(range(-y, grid_size - y)):
            for j, jj in enumerate(range(-x, grid_size - x)):
                # xx[n, i, j] = np.abs(ii) + np.abs(jj) + EPS
                xx[n, i, j] = np.abs(ii) + np.abs(jj) + np.abs(ii - jj) + EPS
                # xx[n, i, j] = np.abs(ii) + np.abs(jj) + np.abs(ii - jj) / 2 + EPS
    return xx


@njit
def color_distance(points: np.array, grid_size: int) -> np.array:
    """Find distance between all points values in an image
    Input
        points: pixels
        grid_size: dimensions of image

    TODO
    Output:
    """
    cc = np.empty(shape=(grid_size**2, grid_size**2), dtype=np.float32)
    for idx, point_i in enumerate(points):
        for jdx, point_j in enumerate(points):
            x1, y1, z1 = point_i
            x2, y2, z2 = point_j
            cc[idx, jdx] = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)
    return cc.reshape(grid_size**2, grid_size, grid_size)


@njit
def calc_probs(inv_ranks: list, c: float = 0.97) -> list:
    """Calculate normalized probabilities for value c for every rank
    Input:
        inv_ranks: inverted list of ranks [99 -> 0]
        c: value of which to perform the exponentiation

    Output: list of probabilities of every rank
    """
    sum_c = np.sum(np.array([c**i for i in inv_ranks]))
    return [c**i / sum_c for i in inv_ranks][::-1]


## == EXPERIMENTS == ##


def plot_synthetic_cluster(num_points: int = 300) -> None:
    """Generate 3 normal distributions and plot results
    Input
        num_points: number of points to generate
    """
    p1 = np.random.normal(loc=0.0, scale=1.0, size=[num_points, 2])
    p2 = np.random.normal(loc=2.5, scale=1.0, size=[num_points, 2])
    p3 = np.random.normal(loc=5.0, scale=1.0, size=[num_points, 2])
    p = np.vstack([p1, p2, p3])

    plt.scatter(p[:, 0], p[:, 1], c="#001FB8")
    plt.show()

    plt.scatter(p1[:, 0], p1[:, 1], c="#001FB8")
    plt.scatter(p2[:, 0], p2[:, 1], c="#EF233C")
    plt.scatter(p3[:, 0], p3[:, 1], c="#2EC4B6")
    plt.show()

    plt.scatter(p1[:, 0], p1[:, 1], c="#001FB8")
    plt.scatter(p2[:, 0], p2[:, 1], c="#EF233C", marker="x")
    plt.scatter(p3[:, 0], p3[:, 1], c="#2EC4B6", marker="D")
    plt.show()


def plot_affect_of_c(pop_size: int) -> None:
    """plots the effect of different values of c
    Input
        pop_size: number of individuals in population
    """
    # Global parameters
    ranks = [i for i in range(1, pop_size + 1)]
    inv_ranks = [i for i in range(pop_size - 1, -1, -1)]
    add_pop = int(np.ceil(pop_size * 0.2))

    def _generate_key(c_values: list, round: int = 3) -> list:
        """Generate a list of values of c in string format for ax.legend
        Input:
            c_values: list of values of c
        Output: list of values of c in string format
        """
        labels = [f"{np.round(i, round)}" for i in c_values]
        return labels

    def _probability_curves(c_values: list) -> None:
        """Plots curves with varying values of
        Note: doesn't accept 0 <= c <= 1 !

        Input
            c_values: a list of c values of which to plot the probabilities
        """
        # Remove out of bounds (0 <= c <= 1)
        c_values = [x for x in c_values if x > 0]
        c_values = [x for x in c_values if x < 1]

        # Generate probabilities
        probs = []
        for c in c_values:
            probs.append(calc_probs(inv_ranks=inv_ranks, c=c))

        # Generate plots
        keys = _generate_key(c_values)
        __, ax = plt.subplots()
        for prob in probs:
            prob = (prob - min(prob)) / (max(prob) - min(prob))
            ax.plot(ranks, prob)

        # Plot prettify
        title = f"Probability Distributions Of Ranks For Varying Values Of 'C'"
        ax.set_title(title)
        ax.legend(keys, title="Values of C")
        plt.xlabel("Ranks")
        plt.ylabel("Probability (Normalized)")
        plt.show()

    def _replace_plots(
        c_values: np.array, samples: int = 1000, replace: bool = True
    ) -> None:
        """Plots box-plots of min, max, and mean values of ranks with varying values of c
        Note: doesn't accept 0 <= c <= 1 !

        Input
            c_values: a list of c values of which to create the box plots
            samples: number of samples to use (more samples = more accuracy, lower speed)
        """
        # remove out of bounds (0 <= c <= 1)
        c_values = c_values[c_values > 0]
        c_values = c_values[c_values < 1]

        stats = [[], [], []]
        for c in tqdm(c_values):
            probs = calc_probs(inv_ranks=inv_ranks, c=c)

            # main loop
            _max, _min, _avg = [], [], []
            for __ in range(samples):
                parents = np.random.choice(
                    ranks, [2, add_pop], replace=replace, p=probs
                )
                _max.append(np.max(parents))
                _min.append(np.min(parents))
                _avg.append(np.mean(parents))
            stats[0].append(_max)
            stats[1].append(_min)
            stats[2].append(_avg)

        # plot
        keys = _generate_key(c_values, 2)
        stats_str = ["Max", "Min", "Avg"]
        for idx, stat in enumerate(stats):
            title = f"Distribution Of {stats_str[idx]} Values With Varying Values Of 'C' (Replace = {replace}, Samples={samples})"
            __, ax = plt.subplots()
            ax.set_title(title)
            ax.boxplot(stat, notch=True, showfliers=False, labels=keys)
            ax.set_xticklabels(keys, rotation=90)
            ax.yaxis.grid(True)
            plt.show()

    # Test 1
    c_values = [0, 0.2, 0.5, 0.75, 0.9, 0.95, 0.975, 0.99, 0.999, 1]
    _probability_curves(c_values)
    del c_values

    # Test 2.1
    samples = 10000
    c_values = np.linspace(start=0, stop=1, num=50)
    # _replace_plots(c_values, samples=samples, replace=True)
    del c_values

    # Test 2.2
    c_values = np.linspace(start=0, stop=1, num=50)
    # _replace_plots(c_values, samples=samples, replace=False)
    del c_values


def plot_synthetic_cluster_v2(num_points: int = 300) -> None:
    """Generate 3 normal distributions and plot results
    Input
        num_points: number of points to generate
    """
    edge = num_points / 2
    size = 25
    x = []
    y = []
    for i in range(num_points):
        for j in range(num_points):
            x.append(i)
            y.append(j * 1.5)
    colors = np.random.rand(num_points * num_points)
    # plt.scatter(x, y, c=colors, alpha=0.5, s=15)

    # plt.xlim(-5, num_points + edge)
    # plt.ylim(-5, num_points + edge)
    # plt.axis("off")
    # plt.show()

    # First pass
    x = []
    y = []
    end_point = int(num_points / 4)
    for i in range(end_point):
        for j in range(end_point):
            x.append(i)
            y.append(j * 1.5)
    # colors = [colors[0] for _ in range(end_point * end_point)]
    # plt.scatter(x, y, c=colors, alpha=0.5, s=size, label="Year 1950")
    # plt.legend()
    # ax = plt.gca()
    # leg = ax.get_legend()

    # Second pass

    end_point = int(num_points / (4**2))
    painters = [
        "William J. L'Engle",
        "James O. Chapin",
        "Thomas Hart Benton ",
        "Reginald Marsh",
        "Ben Shahn",
        "Mary Fife",
        "Daniel R. Celentano",
        "Beatrice Cuming",
    ]
    colors = ["red", "cyan", "green", "pink", "blue", "magenta", "black", "orange"]

    for k, painter in enumerate(painters):
        x = []
        y = []
        for i in range(end_point):
            for j in range(end_point):
                _x = i + (k * len(painters))
                _y = j + (k * len(painters))
                x.append(_x)
                y.append(_y)
        plt.scatter(x, y, alpha=0.5, s=size, label=painter, c=colors[k])
        plt.legend()

    # plt.xlim(-5, num_points + 10)
    # plt.ylim(-5, num_points + 10)
    plt.axis("off")
    plt.show()


plot_synthetic_cluster_v2(100)
