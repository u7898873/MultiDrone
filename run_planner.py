
import argparse
from rrt_planner import run_once

def main():
    parser = argparse.ArgumentParser(description="Run RRT once and visualize if success")
    parser.add_argument("--env", required=True, type=str)
    parser.add_argument("--num_drones", type=int, default=None)
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
        goal_sample_rate=args.goal_sample_rate
    )

if __name__ == "__main__":
    main()
