#!/usr/bin/env python3

"""
Author:     Jacopo Di Matteo
Date:       18.06.2022

This code is provided "As Is"
"""

# Default
import json
from copy import deepcopy
from pathlib import Path

# Third-party
import matplotlib as plt
import matplotlib.pylab as plt
import numba as nb
import numpy as np
from skimage.color import rgb2lab
from tqdm import trange

# Local
from ops import (
    EPS,
    basic_distance,
    calc_probs,
    color_distance,
    correct_num_points,
    get_xy_coord,
)

LOW = 0
HIGH = 255


class pc:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def print_clr(to_print: str) -> None:
    """
    TODO
    """
    print(pc.OKBLUE + to_print + pc.ENDC)


class EA:
    """
    TODO
    """

    def __init__(
        self,
        number_of_colors: int,
        population_size: int,
        number_of_generations: int,
        c: float = 0.96,
        lambda_factor: float = 0.3,  # lambda factor (mu + lambda)
        mutation_rate: float = 0.3,
        cie: bool = False,
    ) -> None:
        """
        TODO
        """
        # Set population size
        self.pop_size = population_size

        # Set number of generations
        self.gen_num = number_of_generations

        # Set color space
        self.cie = cie

        # Get number of colors to produce
        self.grid_size, self.num_points = correct_num_points(number_of_colors)
        print_clr(
            f"=== Starting EA: c: {c} lambda: {lambda_factor} mu: {mutation_rate}  palette: {self.num_points} == "
        )
        print_clr("Get Number of Colors to Produce")

        # Calculate difference between requested number of colors and actual
        self.diff = self.num_points - number_of_colors

        # Get coordinates of each pixel
        print_clr("Get Coordinates of Each Pixel")
        coords = get_xy_coord(grid_size=self.grid_size).astype(np.float32)

        # Retrieve correction matrix for given configuration (number of colors)
        print_clr("Retrieve Correction Matrix")
        self.corr_mat = self.get_correction_matrix(coords)

        # Generate initial population
        print_clr("Generate Initial Population")
        self.x, self.f = self.initial_population()
        self.age = [1 for __ in range(self.pop_size)]

        # Plot samples of initial population
        # idx = np.ceil(np.linspace(start=0, stop=self.pop_size - 1, num=9)).astype(
        #     np.int32
        # )
        # print_clr("Plot Initial Batch")
        # self.plot_organisms(self.x, self.f, idx)

        # Evolutionary steps
        self.c = c
        self.ranks = [i for i in range(1, self.pop_size + 1)]
        self.inv_ranks = nb.typed.List([i for i in range(self.pop_size - 1, -1, -1)])
        self.add_pop = int(np.ceil(self.pop_size * lambda_factor))
        self.mutation_rate = mutation_rate

        _f = []
        _x = []
        for idx in trange(self.gen_num):
            self.x, self.f = self.step()
            _f.append(self.f[0])
            _x.append(self.x[0])

        # Plot fitness progression
        __ = plt.figure()
        ax = plt.axes()
        x = [i for i in range(self.gen_num)]
        ax.plot(x, _f)

        Path(f"./{self.pop_size}/{self.gen_num}/line_plot/").mkdir(
            parents=True, exist_ok=True
        )
        plt.savefig(
            f"./{self.pop_size}/{self.gen_num}/line_plot/{self.num_points}",
            dpi=300,
            bbox_inches="tight",
        )

        # Plot samples of final population
        # idx = np.ceil(np.linspace(start=0, stop=self.pop_size - 1, num=9)).astype(
        #     np.int32
        # )
        # self.plot_organisms(self.x, self.f, idx)

        # Plot samples of best performers
        f = np.array(_f)
        x = np.array(_x)
        _idx = np.argsort(_f)[::-1]
        f = f[_idx]
        x = x[_idx]
        idx = _idx[-9:][::-1]
        self.plot_organisms(x, f, idx)

        # Save best performer
        palette = {}
        best = np.reshape(x[_idx[-1]], [self.grid_size, self.grid_size, 3])
        for i in range(self.grid_size):
            palette[i] = {}
            for j in range(self.grid_size):
                r, g, b = best[i, j]
                palette[i][j] = "#{0:02x}{1:02x}{2:02x}".format(r, g, b)

        # Serializing json
        Path(f"./{self.pop_size}/{self.gen_num}/palette/").mkdir(
            parents=True, exist_ok=True
        )
        with open(
            f"./{self.pop_size}/{self.gen_num}/palette/{self.num_points}.json", "w"
        ) as outfile:
            json.dump(palette, outfile)

    def step(self) -> None:
        x_parents = self.parent_selection()
        x_new = self.cross_over(x_parents)
        x_new = self.mutation(x_new)
        f_new = self.fitness_eval(x_new)
        return self.survivor_selection(x_new, f_new)

    # @jit
    def survivor_selection(self, new_x: np.array, new_f: np.array) -> np.array:
        """
        TODO
        """
        # Age based correction of survival probability
        age_corr = np.array([1 / i for i in self.age])
        probs_age_corr = self.probs * age_corr

        # Fitness based selection as used for parents
        ranks_p = np.random.choice(
            self.inv_ranks[::-1],
            [np.abs(self.pop_size - self.add_pop * 2)],
            replace=False,
            p=probs_age_corr,
        )

        # Congregation
        x = self.x[ranks_p]
        f = self.f[ranks_p]
        x = np.append(x, new_x, axis=0)
        f = np.append(f, new_f, axis=0)

        # Sort
        idx = np.argsort(f)[::-1]
        f = f[idx]
        x = x[idx]
        return x, f

    # @jit
    def fitness_eval(self, new_x: np.array) -> np.array:
        """
        TODO
        """
        f = np.zeros(new_x.shape[0])
        for idx, i in enumerate(new_x):
            i = i / 255
            if self.cie:
                i = rgb2lab(i)
            distances = color_distance(i, self.grid_size)
            distances *= self.corr_mat
            fitness = np.sum(distances)
            f[idx] = fitness / (self.num_points**2)
        return f

    # @jit
    def mutation(self, new_x: np.array) -> np.array:
        """
        TODO
        """
        n_pairs, pair_arity, n_alleles, channels = new_x.shape
        new_x = new_x.reshape([n_pairs * pair_arity, n_alleles, channels])
        # assert new_x[0].all() == _temp.all()
        for idx, child in enumerate(new_x):
            for jdx, __ in enumerate(child):
                if self.mutation_rate >= np.random.uniform(0, 1):
                    new_x[idx, jdx] = np.random.randint(LOW, HIGH, 3)
        return new_x

    # @jit
    def cross_over(self, parents: np.array) -> np.array:
        """
        TODO
        """
        children = deepcopy(parents)
        for idx, pair in enumerate(parents):
            crossover_rate = np.random.uniform(0, 1)
            for jdx in range(self.num_points):
                if crossover_rate >= np.random.uniform(0, 1):
                    children[idx][0, jdx] = pair[1, jdx]
                    children[idx][1, jdx] = pair[0, jdx]
        return children

    # @jit
    def parent_selection(self, replace: bool = True) -> np.array:
        """
        TODO
        """
        self.probs = calc_probs(inv_ranks=self.inv_ranks, c=self.c)
        ranks_p = np.random.choice(
            self.inv_ranks[::-1], [self.add_pop, 2], replace=replace, p=self.probs
        )
        return self.x[ranks_p]

    # @jit
    def plot_organisms(self, x, f, idx_list: list) -> None:
        """
        TODO
        """
        fig, axs = plt.subplots(3, 3)
        plt.subplots_adjust(wspace=0.5, hspace=0.5)
        for i, organism_idx in enumerate(idx_list):
            row = int(np.floor(i / 3))
            column = i % 3
            title = f"Position: {organism_idx}, Fitness: {np.round(f[organism_idx], 3)}"
            axs[row, column].set_title(title)
            axs[row, column].matshow(
                x[organism_idx].reshape(self.grid_size, self.grid_size, 3)
            )
        fig.set_size_inches(10, 8)
        Path(f"./{self.pop_size}/{self.gen_num}/matshow/").mkdir(
            parents=True, exist_ok=True
        )
        plt.savefig(
            f"./{self.pop_size}/{self.gen_num}/matshow/{self.num_points}",
            dpi=300,
            bbox_inches="tight",
        )

    # @jit
    def initial_population(self) -> tuple[np.array, np.array]:
        """
        TODO
        """
        # Generate values
        x = np.random.randint(
            low=LOW, high=HIGH, size=[self.pop_size, self.num_points, 3]
        )

        # Calculate individual fitness
        f = self.fitness_eval(x)

        # Sort by fitness
        idx = np.argsort(f)[::-1]
        f_sorted = f[idx]
        x_sorted = x[idx]
        return x_sorted, f_sorted

    # @jit
    def get_correction_matrix(self, coords) -> np.array:
        """
        TODO
        """
        return 1 / (basic_distance(points=coords, grid_size=self.grid_size))


def main() -> None:
    pops = [50, 100, 250, 500, 1000]
    gens = [100, 500, 1000, 5000, 10000]
    for gen in gens:
        for pop in pops:
            for i in range(2, 25):
                num_of_colors = i**2
                population_size = pop
                num_of_gen = gen
                ea = EA(
                    number_of_colors=num_of_colors,
                    population_size=population_size,
                    number_of_generations=num_of_gen,
                )
                del ea


if __name__ == "__main__":
    main()
