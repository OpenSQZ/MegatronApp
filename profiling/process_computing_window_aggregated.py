import argparse
import json
import matplotlib.pyplot as plt
import os
import numpy as np

parser = argparse.ArgumentParser(description="Process trace file and generate plots.")
parser.add_argument("--trace_dir", help="Directory containing trace (required)", required=True)
args = parser.parse_args()
aggregated_average_compute_windows = []
for pp_type in ["pp", "dpp"]:
    trace_dir = args.trace_dir + "-" + pp_type
    trace_file = os.path.join(trace_dir, trace_dir.split("/")[-1] + "-aggregated.json")
    with open(trace_file) as f:
        trace = json.load(f)

    pp_ranks = 1
    for item in trace:
        if item["ph"] == "E" and item["args"]["pp_rank"] + 1 > pp_ranks:
            pp_ranks = item["args"]["pp_rank"] + 1

    forward_compute_starts_by_rank = []
    forward_compute_ends_by_rank = []
    backward_compute_starts_by_rank = []
    backward_compute_ends_by_rank = []
    avg_compute_windows = []
    for rank in range(pp_ranks):
        forward_compute_ends_by_rank.append(
            sorted(
                [item for item in trace if item["ph"] == "E" and item["args"]["pp_rank"] == rank and item["args"]["iteration"] > 0 and item["name"] == "forward"],
                key=lambda item: item["ts"]
            )
        )
        forward_compute_starts_by_rank.append(
            sorted(
                [item for item in trace if item["ph"] == "B" and item["pid"] == rank and item["args"]["iteration"] > 0 and item["name"] == "forward"],
                key=lambda item: item["ts"]
            )
        )
        backward_compute_ends_by_rank.append(
            sorted(
                [item for item in trace if item["ph"] == "E" and item["args"]["pp_rank"] == rank and item["args"]["iteration"] > 0 and item["name"] == "backward"],
                key=lambda item: item["ts"]
            )
        )
        backward_compute_starts_by_rank.append(
            sorted(
                [item for item in trace if item["ph"] == "B" and item["pid"] == rank and item["args"]["iteration"] > 0 and item["name"] == "backward"],
                key=lambda item: item["ts"]
            )
        )
        compute_times = 0
        for i in range(len(forward_compute_ends_by_rank[rank])):
            assert forward_compute_ends_by_rank[rank][i]["ts"] >= forward_compute_starts_by_rank[rank][i]["ts"]
            compute_times += (forward_compute_ends_by_rank[rank][i]["ts"] - forward_compute_starts_by_rank[rank][i]["ts"])
        for i in range(len(backward_compute_ends_by_rank[rank])):
            assert forward_compute_ends_by_rank[rank][i]["ts"] >= forward_compute_starts_by_rank[rank][i]["ts"]
            compute_times += (backward_compute_ends_by_rank[rank][i]["ts"] - backward_compute_starts_by_rank[rank][i]["ts"])
        avg_compute_windows.append(compute_times / (len(forward_compute_starts_by_rank[rank]) + len(backward_compute_starts_by_rank[rank])) / 1000)
    aggregated_average_compute_windows.append(avg_compute_windows)
ranks = ["0", "1", "2", "3"]
pp_data = aggregated_average_compute_windows[0]
dpp_data = aggregated_average_compute_windows[1]
x = np.arange(len(ranks))
width = 0.35
plt.figure(figsize=(10, 6))
plt.bar(x - width / 2, pp_data, width, label='pp')
plt.bar(x + width / 2, dpp_data, width, label='dpp')

plt.xlabel("GPU Number")
plt.ylabel("Compute Window (ms)")
plt.title(args.trace_dir.split("/")[-1] + " pp vs dpp")
plt.xticks(x, ranks)
plt.legend()
plt.tight_layout()
plt.savefig(f"plots/{args.trace_dir.split('/')[-1]}-compute-windows-aggregated.png")
