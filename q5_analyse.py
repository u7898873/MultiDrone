import pandas as pd
import matplotlib.pyplot as plt
import re

def extract_drone_number(env_name):
    match = re.search(r"drone(\d+)", env_name)
    return int(match.group(1)) if match else None

def main():
    df = pd.read_csv("results_q5.csv")
    print("CSV columns:", df.columns.tolist())

    df["num_drones"] = df["env"].apply(extract_drone_number)

    summary_df = df[[
        "env", "num_drones", "success_rate",
        "time_mean", "time_ci_low", "time_ci_high",
        "length_mean", "length_ci_low", "length_ci_high"
    ]].sort_values("num_drones")

    print(summary_df)

    summary_df.to_csv("summary_q5.csv", index=False)
    print("Saved summary_q5.csv")

    plt.figure(figsize=(8, 5))
    plt.plot(summary_df["num_drones"], summary_df["success_rate"], marker="o", label="Success Rate")
    plt.xlabel("Number of Drones")
    plt.ylabel("Success Rate")
    plt.title("Success Rate vs Number of Drones")
    plt.grid(True)
    plt.savefig("success_rate_q5.png")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.errorbar(
        summary_df["num_drones"],
        summary_df["time_mean"],
        yerr=[summary_df["time_mean"] - summary_df["time_ci_low"], summary_df["time_ci_high"] - summary_df["time_mean"]],
        fmt="o-",
        capsize=5,
        label="Planning Time"
    )
    plt.xlabel("Number of Drones")
    plt.ylabel("Planning Time (s)")
    plt.title("Planning Time vs Number of Drones")
    plt.grid(True)
    plt.savefig("planning_time_q5.png")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.errorbar(
        summary_df["num_drones"],
        summary_df["length_mean"],
        yerr=[summary_df["length_mean"] - summary_df["length_ci_low"], summary_df["length_ci_high"] - summary_df["length_mean"]],
        fmt="o-",
        capsize=5,
        label="Path Length"
    )
    plt.xlabel("Number of Drones")
    plt.ylabel("Path Length")
    plt.title("Path Length vs Number of Drones")
    plt.grid(True)
    plt.savefig("path_length_q5.png")
    plt.close()

    print("Saved plots: success_rate_q5.png, planning_time_q5.png, path_length_q5.png")

if __name__ == "__main__":
    main()
