
import time
import math
import argparse
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

# Expect multi_drone.py to be on PYTHONPATH or in the same folder
from multi_drone import MultiDrone


@dataclass
class RRTConfig:
    max_iterations: int = 100000000
    step_size: float = 1.0                 # step in meters in joint space projection
    goal_sample_rate: float = 0.05         # with this prob, sample a goal-biased target
    goal_check_every: int = 10             # call is_goal for nodes every N iters
    goal_tolerance: float = 0.5            # extra safety when snapping to goal via motion_valid
    time_limit: float = 120.0              # seconds (assignment constraint)
    seed: Optional[int] = None


class RRTPlanner:
    """Rapidly-exploring Random Tree (RRT) for centralized multi-drone planning.

    Works for K=1 (single drone) and K>1 (multi-drone) without any code changes.
    """
    def __init__(self, sim: MultiDrone, cfg: RRTConfig):
        self.sim = sim
        self.cfg = cfg
        if cfg.seed is not None:
            np.random.seed(cfg.seed)

        # Cache bounds and dimensions
        self.N = self.sim.N
        # Using internal bounds; they exist in provided MultiDrone
        self.bounds = np.array(self.sim._bounds, dtype=np.float32)  # (3, 2)
        self.low = self.bounds[:, 0]
        self.high = self.bounds[:, 1]

        # Flatten helpers
        self.dim = 3 * self.N

    # --------------- Utility functions ---------------
    def _flatten(self, q: np.ndarray) -> np.ndarray:
        """(N,3) -> (3N,)"""
        return q.reshape(-1)

    def _unflatten(self, x: np.ndarray) -> np.ndarray:
        """(3N,) -> (N,3)"""
        return x.reshape(self.N, 3)

    def _sample_random(self) -> np.ndarray:
        """Uniform sample within bounds for each drone; returns (N,3)."""
        samples = np.random.uniform(self.low, self.high, size=(self.N, 3)).astype(np.float32)
        return samples

    def _nearest(self, nodes: List[np.ndarray], x_rand: np.ndarray) -> int:
        """Return index of nearest node in Euclidean norm of flattened joint space."""
        xr = self._flatten(x_rand)
        dists = [np.linalg.norm(self._flatten(n) - xr) for n in nodes]
        return int(np.argmin(dists))

    def _steer_towards(self, x_from: np.ndarray, x_to: np.ndarray, step: float) -> np.ndarray:
        """Move from x_from towards x_to by at most 'step' in joint-space L2.
        Both are (N,3). Returns (N,3).
        """
        v = self._flatten(x_to) - self._flatten(x_from)
        dist = np.linalg.norm(v)
        if dist <= 1e-9:
            return x_from.copy()
        scale = min(1.0, step / dist)
        x_new = self._flatten(x_from) + scale * v
        return self._unflatten(x_new.astype(np.float32))

    def _path_length(self, path: List[np.ndarray]) -> float:
        """Total physical path length: sum over drones of their traveled distance along the polyline."""
        if path is None or len(path) < 2:
            return 0.0
        total = 0.0
        for i in range(1, len(path)):
            seg = path[i] - path[i-1]  # (N,3)
            # sum of per-drone Euclidean distances for this segment
            d_per_drone = np.linalg.norm(seg, axis=1)  # (N,)
            total += float(np.sum(d_per_drone))
        return total

    # --------------- Planning ---------------
    def plan(self) -> Tuple[Optional[List[np.ndarray]], dict]:
        """Plan from initial_configuration to goal areas defined in the simulator.

        Returns (path, info)
          - path: list of (N,3) configurations from start to goal (inclusive), or None if failed
          - info: dict with stats (time, iters, success, path_length)
        """
        start_time = time.time()

        start = np.array(self.sim.initial_configuration, dtype=np.float32)
        # A convenient target for goal-biased sampling is the center of each goal sphere
        goal_hint = np.array(self.sim.goal_positions, dtype=np.float32)

        # Tree storage
        nodes: List[np.ndarray] = [start]
        parents: List[int] = [-1]

        iters = 0
        success = False
        last_goal_idx = -1

        while iters < self.cfg.max_iterations and (time.time() - start_time) < self.cfg.time_limit:
            iters += 1

            # Sample target (goal-biased sometimes)
            if np.random.rand() < self.cfg.goal_sample_rate:
                x_rand = goal_hint
            else:
                x_rand = self._sample_random()

            # Nearest
            idx_near = self._nearest(nodes, x_rand)
            x_near = nodes[idx_near]

            # Steer
            x_new = self._steer_towards(x_near, x_rand, self.cfg.step_size)

            # Collision checking on the motion
            if self.sim.motion_valid(x_near, x_new):
                nodes.append(x_new)
                parents.append(idx_near)

                # Periodically check goal condition for early success
                if (iters % self.cfg.goal_check_every == 0) or (np.random.rand() < 0.02):
                    # If inside all goal spheres -> success
                    if self.sim.is_goal(x_new):
                        success = True
                        last_goal_idx = len(nodes) - 1
                        break

                    # Try snapping directly to the goal center (useful when goals are regions)
                    if self.sim.motion_valid(x_new, goal_hint):
                        nodes.append(goal_hint.copy())
                        parents.append(len(nodes) - 2)
                        success = True
                        last_goal_idx = len(nodes) - 1
                        break

        # Build path if success
        path: Optional[List[np.ndarray]] = None
        if success and last_goal_idx >= 0:
            path = []
            cur = last_goal_idx
            while cur != -1:
                path.append(nodes[cur])
                cur = parents[cur]
            path.reverse()

        info = {
            "time": time.time() - start_time,
            "iters": iters,
            "success": success,
            "path_length": (self._path_length(path) if path is not None else math.inf),
            "num_nodes": len(nodes),
        }
        return path, info


def run_once(env_yaml: str, num_drones: Optional[int] = None, seed: Optional[int] = None,
             step_size: float = 1.0, time_limit: float = 120.0, goal_sample_rate: float = 0.05):
    """Helper for single run + visualization."""
    # If num_drones not provided, infer from YAML's initial_configuration length
    if num_drones is None:
        import yaml
        with open(env_yaml, "r") as f:
            cfg = yaml.safe_load(f)
        num_drones = len(cfg["initial_configuration"])

    sim = MultiDrone(num_drones=num_drones, environment_file=env_yaml)
    rrt_cfg = RRTConfig(step_size=step_size, time_limit=time_limit, goal_sample_rate=goal_sample_rate, seed=seed)
    planner = RRTPlanner(sim, rrt_cfg)

    path, info = planner.plan()

    print(f"Planning info: {info}")
    if info["success"] and path is not None:
        # Visualize
        sim.visualize_paths(path)
    else:
        print("No solution found within limits.")


def main():
    parser = argparse.ArgumentParser(description="RRT Planner for (Multi)Drone")
    parser.add_argument("--env", type=str, required=True, help="Path to environment YAML")
    parser.add_argument("--num_drones", type=int, default=None, help="Override number of drones (optional)")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--step_size", type=float, default=1.0)
    parser.add_argument("--time_limit", type=float, default=120.0)
    parser.add_argument("--goal_sample_rate", type=float, default=0.05)
    args = parser.parse_args()

    run_once(
        env_yaml=args.env,
        num_drones=args.num_drones,
        seed=args.seed,
        step_size=args.step_size,
        time_limit=args.time_limit,
        goal_sample_rate=args.goal_sample_rate,
    )


if __name__ == "__main__":
    main()
