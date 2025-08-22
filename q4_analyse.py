import pandas as pd
import matplotlib.pyplot as plt
import re

def extract_obstacle_number(env_name):
    match = re.search(r"obstacle(\d+)", env_name)
    return int(match.group(1)) if match else None

def main():
    df = pd.read_csv("results_q4.csv")
    print("CSV columns:", df.columns.tolist())

    df["num_obstacles"] = df["env"].apply(extract_obstacle_number)

    summary_df = df[[
        "env", "num_obstacles", "success_rate",
        "time_mean", "time_ci_low", "time_ci_high"
    ]].sort_values("num_obstacles")

    print(summary_df)

    summary_df.to_csv("summary_q4.csv", index=False)
    print("Saved summary_q4.csv")

    plt.figure(figsize=(8, 5))
    plt.plot(summary_df["num_obstacles"], summary_df["success_rate"], marker="o", label="Success Rate")
    plt.xlabel("Number of Obstacles")
    plt.ylabel("Success Rate")
    plt.title("Success Rate vs Environment Complexity")
    plt.grid(True)
    plt.savefig("success_rate_q4.png")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.errorbar(
        summary_df["num_obstacles"],
        summary_df["time_mean"],
        yerr=[summary_df["time_mean"] - summary_df["time_ci_low"], summary_df["time_ci_high"] - summary_df["time_mean"]],
        fmt="o-",
        capsize=5,
        label="Planning Time"
    )
    plt.xlabel("Number of Obstacles")
    plt.ylabel("Planning Time (s)")
    plt.title("Planning Time vs Environment Complexity")
    plt.grid(True)
    plt.savefig("planning_time_q4.png")
    plt.close()

    print("Saved plots: success_rate_q4.png, planning_time_q4.png")

if __name__ == "__main__":
    main()
