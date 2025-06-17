import argparse
import json
import matplotlib.pyplot as plt
from scipy.stats import gmean

parser = argparse.ArgumentParser(description="Process trace file and generate plots.")
parser.add_argument("--trace_file", help="File containing trace (required)", required=True)
args = parser.parse_args()
trace_file = args.trace_file
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
ranks = ["0", "1", "2", "3"]
avg_compute_ratios = []
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
    iter_starts = sorted([item for item in trace if item["name"] == "iteration begin" and item["pid"] == rank], key=lambda item: item["ts"])
    iter_ends = sorted([item for item in trace if item["name"] == "iteration end" and item["pid"] == rank], key=lambda item: item["ts"])
    compute_ratios = []
    for iter in range(1, 20):
        compute_times = 0
        for i in range(len(forward_compute_ends_by_rank[rank])):
            assert forward_compute_ends_by_rank[rank][i]["ts"] >= forward_compute_starts_by_rank[rank][i]["ts"]
            if forward_compute_ends_by_rank[rank][i]["args"]["iteration"] == iter:
                compute_times += (forward_compute_ends_by_rank[rank][i]["ts"] - forward_compute_starts_by_rank[rank][i]["ts"])
        for i in range(len(backward_compute_ends_by_rank[rank])):
            assert forward_compute_ends_by_rank[rank][i]["ts"] >= forward_compute_starts_by_rank[rank][i]["ts"]
            if forward_compute_ends_by_rank[rank][i]["args"]["iteration"] == iter:
                compute_times += (backward_compute_ends_by_rank[rank][i]["ts"] - backward_compute_starts_by_rank[rank][i]["ts"])
        compute_ratios.append(compute_times / (iter_ends[iter]["ts"] - iter_starts[iter]["ts"]))
    avg_compute_ratios.append(gmean(compute_ratios))
plt.bar(ranks, avg_compute_ratios)
plt.xlabel("GPU Number")
plt.ylabel("Compute Ratio")
plt.title(trace_file.split("/")[-1][:-5])
plt.savefig(f"plots/{trace_file.split("/")[-1][:-5]}-computing-ratios.png")
