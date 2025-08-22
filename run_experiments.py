
import argparse
import time
import math
from typing import List, Optional, Dict, Any

import numpy as np
import yaml

from rrt_planner import RRTPlanner, RRTConfig
from multi_drone import MultiDrone


def mean_ci_95(xs: List[float]):
    arr = np.array(xs, dtype=np.float64)
    n = len(arr)
    if n == 0:
        return (float("nan"), (float("nan"), float("nan")))
    m = float(np.mean(arr))
    if n == 1:
        return (m, (m, m))
    s = float(np.std(arr, ddof=1))
    half = 1.96 * s / math.sqrt(n)
    return (m, (m - half, m + half))


def run_env(env_yaml: str, runs: int, step_size: float, time_limit: float, goal_sample_rate: float, seed: Optional[int]) -> Dict[str, Any]:
    with open(env_yaml, "r") as f:
        cfg_yaml = yaml.safe_load(f)
    num_drones = len(cfg_yaml["initial_configuration"])

    times = []
    lengths = []
    successes = 0
    iters_list = []
    nodes_list = []

    for r in range(runs):
        this_seed = None if seed is None else (seed + r)
        sim = MultiDrone(num_drones=num_drones, environment_file=env_yaml)
        planner = RRTPlanner(sim, RRTConfig(step_size=step_size, time_limit=time_limit,
                                            goal_sample_rate=goal_sample_rate, seed=this_seed))

        path, info = planner.plan()

        if info["success"]:
            successes += 1
            times.append(info["time"])
            lengths.append(info["path_length"])
        else:
            times.append(info["time"])
            lengths.append(float("nan"))
        iters_list.append(info["iters"])
        nodes_list.append(info["num_nodes"])

    success_rate = successes / runs if runs > 0 else 0.0
    time_mean, time_ci = mean_ci_95([t for t in times if not math.isnan(t)])
    succ_lengths = [l for l in lengths if not math.isnan(l)]
    length_mean, length_ci = mean_ci_95(succ_lengths) if len(succ_lengths) > 0 else (float("nan"), (float("nan"), float("nan")))
    iters_mean, iters_ci = mean_ci_95(iters_list)
    nodes_mean, nodes_ci = mean_ci_95(nodes_list)

    return {
        "env": env_yaml,
        "num_drones": num_drones,
        "runs": runs,
        "success_rate": success_rate,
        "time_mean": time_mean, "time_ci_low": time_ci[0], "time_ci_high": time_ci[1],
        "length_mean": length_mean, "length_ci_low": length_ci[0], "length_ci_high": length_ci[1],
        "iters_mean": iters_mean, "iters_ci_low": iters_ci[0], "iters_ci_high": iters_ci[1],
        "nodes_mean": nodes_mean, "nodes_ci_low": nodes_ci[0], "nodes_ci_high": nodes_ci[1],
    }


def main():
    parser = argparse.ArgumentParser(description="Run RRT experiments across environments")
    parser.add_argument("--envs", type=str, nargs="+", required=True, help="List of YAML files (vary complexity or #drones)")
    parser.add_argument("--runs", type=int, default=10, help="Repeats per environment")
    parser.add_argument("--step_size", type=float, default=1.0)
    parser.add_argument("--time_limit", type=float, default=120.0)
    parser.add_argument("--goal_sample_rate", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--csv_out", type=str, default="results.csv")
    args = parser.parse_args()

    rows = []
    for env in args.envs:
        print(f"=== {env} ===")
        res = run_env(env, args.runs, args.step_size, args.time_limit, args.goal_sample_rate, args.seed)
        print(res)
        rows.append(res)

    import csv
    fieldnames = list(rows[0].keys()) if rows else []
    with open(args.csv_out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"Wrote {args.csv_out}")


if __name__ == "__main__":
    main()
