import os
import pickle

import cma
import numpy as np

from evorob.algorithms.base_ea import EA
from evorob.utils.filesys import get_last_checkpoint_dir


class EvoAlgAPI(EA):
    """Evolutionary algorithm API wrapper.

    This class provides an interface to wrap any EA framework that uses
    the ask-tell pattern (CMA-ES, pyribs, evosax, etc.).

    Example frameworks to use:
    - CMA-ES: https://github.com/CMA-ES/pycma
    - pyribs: https://github.com/icaros-usc/pyribs/
    - evosax: https://github.com/RobertTLange/evosax/
    - EvoJAX: https://github.com/google/evojax
    """

    def __init__(self, n_params: int, population_size: int = 100, num_generations: int = 100,
                 output_dir: str = "./results/EA", **kwargs):
        """Initialize the evolutionary algorithm.

        Args:
            n_params: Dimensionality of the search space
            population_size: Number of solutions per generation
            num_generations: Number of generations
            output_dir: Directory for saving checkpoints
            **kwargs: Additional arguments for the EA framework
        """
        # TODO: Initialize your chosen EA framework here
        self.n_params = n_params
        self.sigma = kwargs.pop("sigma", 0.3)
        self.opts = dict(kwargs.pop("opts", {}))
        self.opts.setdefault("popsize", population_size)
        self.opts.update(kwargs)
        self.ea = cma.CMAEvolutionStrategy(
            np.zeros(self.n_params), self.sigma, self.opts
        )

        self.n_gen = num_generations
        self.population_size = population_size
        
        # % bookkeeping for base EA
        self.directory_name = output_dir
        self.current_gen = 0
        self.full_x = []
        self.full_f = []
        self.x_best_so_far = None
        self.f_best_so_far = -np.inf
        self.x = None
        self.f = None
        return None
        raise NotImplementedError(
            "TODO: Initialize your chosen EA framework.\n"
            "Recommended: pip install cma, then import cma and create CMAEvolutionStrategy.\n"
            "See https://github.com/CMA-ES/pycma for documentation."
        )

    def ask(self) -> np.ndarray:
        """Sample population from the algorithm.
        Returns:
            population: Array of shape (population_size, n_params)
                       Each row is a candidate solution
        """
        # TODO: Get new population from your EA
        # Make sure the returned array has shape (population_size, n_params)
        self.x = self.ea.ask()  # sample candidate solutions
        return np.array(self.x)  # convert to numpy array if needed
        

        raise NotImplementedError(
            "TODO: Implement ask() to sample new population.\n"
            "This should return an array of shape (population_size, n_params)."
        )

    def tell(self, population: np.ndarray, fitnesses: np.ndarray, save_checkpoint: bool = False) -> None:
        """Update the algorithm with evaluated population.

        Args:
            population: Array of shape (population_size, n_params)
            fitnesses: Array of shape (population_size,) with fitness values
                      Higher is better (maximization)
            save_checkpoint: Whether to save checkpoint after update
        """
        # TODO: Update your EA with the evaluated population
        # Note: Some algorithms minimize, others maximize.
        # Adjust accordingly (negate fitnesses if needed).
        self.ea.tell(population, -fitnesses)  # update EA with fitnesses
        
        # After updating the EA, do bookkeeping for checkpointing:
        self.full_f.append(fitnesses)
        self.full_x.append(population)
        self.f = fitnesses
        self.x = population
        
        # Track best individual
        best_idx = np.argmax(fitnesses)
        if fitnesses[best_idx] > self.f_best_so_far:
            self.f_best_so_far = fitnesses[best_idx]
            self.x_best_so_far = population[best_idx].copy()
        
        if save_checkpoint:
            self.save_checkpoint()
        self.current_gen += 1

        return None
        raise NotImplementedError(
            "TODO: Implement tell() to update the EA.\n"
            "Pass the population and their fitness values to update the search distribution.\n"
            "Don't forget to add the bookkeeping code shown above for checkpointing!"
        )

    def save_checkpoint(self):
        super().save_checkpoint()
        curr_gen_path = os.path.join(self.directory_name, str(self.current_gen))
        cma_state_path = os.path.join(curr_gen_path, "cma_state.pkl")
        with open(cma_state_path, "wb") as f:
            pickle.dump(self.ea, f)

    def resume_from_checkpoint(self) -> str:
        """Resume the optimizer from the latest checkpoint in directory_name.

        Returns:
            "exact" if the serialized CMA-ES state was found and restored.
            "warm_start" if only the historical best solution was available.
        """
        checkpoint_dir = self.directory_name
        checkpoint_name = os.path.basename(os.path.normpath(checkpoint_dir))
        root_full_f_path = os.path.join(checkpoint_dir, "full_f.npy")
        root_full_x_path = os.path.join(checkpoint_dir, "full_x.npy")

        # Allow resuming either from the run root directory or from a specific
        # generation subdirectory such as ".../299".
        if checkpoint_name.isdigit():
            candidate_gen_dir = checkpoint_dir
            candidate_root_dir = os.path.dirname(checkpoint_dir)
            if all(
                os.path.isfile(os.path.join(candidate_gen_dir, file))
                for file in ("x_best.npy", "f_best.npy", "x.npy", "f.npy")
            ) and all(
                os.path.isfile(os.path.join(candidate_root_dir, file))
                for file in ("full_f.npy", "full_x.npy")
            ):
                last_gen_dir = candidate_gen_dir
                self.directory_name = candidate_root_dir
            else:
                last_gen_dir = ""
        else:
            last_gen_dir = get_last_checkpoint_dir(
                checkpoint_dir,
                required_files=("x_best.npy", "f_best.npy", "x.npy", "f.npy"),
            )

        if not last_gen_dir:
            raise FileNotFoundError(
                f"No checkpoint directories found in '{self.directory_name}'."
            )

        full_f_path = os.path.join(self.directory_name, "full_f.npy")
        full_x_path = os.path.join(self.directory_name, "full_x.npy")
        x_best_path = os.path.join(last_gen_dir, "x_best.npy")
        f_best_path = os.path.join(last_gen_dir, "f_best.npy")
        x_path = os.path.join(last_gen_dir, "x.npy")
        f_path = os.path.join(last_gen_dir, "f.npy")

        if not all(
            os.path.isfile(path)
            for path in [full_f_path, full_x_path, x_best_path, f_best_path, x_path, f_path]
        ):
            raise FileNotFoundError(
                f"Checkpoint in '{last_gen_dir}' is incomplete and cannot be resumed."
            )

        self.full_f = [np.array(gen) for gen in np.load(full_f_path)]
        self.full_x = [np.array(gen) for gen in np.load(full_x_path)]
        self.x_best_so_far = np.load(x_best_path)
        self.f_best_so_far = float(np.load(f_best_path))
        self.x = np.load(x_path)
        self.f = np.load(f_path)

        last_gen = int(os.path.basename(last_gen_dir))
        self.current_gen = last_gen + 1

        cma_state_path = os.path.join(last_gen_dir, "cma_state.pkl")
        if os.path.isfile(cma_state_path):
            with open(cma_state_path, "rb") as f:
                self.ea = pickle.load(f)
            self.population_size = getattr(self.ea, "popsize", self.population_size)
            return "exact"

        # Backward-compatible fallback for older checkpoints without serialized state:
        # restart CMA-ES around the best solution found so far.
        loaded_population_size = int(self.x.shape[0])
        if loaded_population_size != self.population_size:
            print(
                "Checkpoint has no serialized CMA-ES state; warm-start resume will "
                f"use checkpoint population_size={loaded_population_size} instead of "
                f"requested population_size={self.population_size}."
            )
        self.population_size = loaded_population_size
        warm_start_opts = dict(self.opts)
        warm_start_opts["popsize"] = loaded_population_size
        self.ea = cma.CMAEvolutionStrategy(
            np.array(self.x_best_so_far, copy=True), self.sigma, warm_start_opts
        )
        return "warm_start"
