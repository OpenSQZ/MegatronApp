import argparse
import json
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.stats import gmean

parser = argparse.ArgumentParser(description="Process trace file and generate plots.")
parser.add_argument("--trace_dir", help="Directory containing trace (required)", required=True)
args = parser.parse_args()
iter_times = []
aggregated_reduce_windows = []
aggregated_reduce_ratios = []
aggregated_iters = []
for pp_type in ["pp", "dpp"]:
    trace_dir = args.trace_dir + "-" + pp_type
    trace_file = os.path.join(trace_dir, trace_dir.split("/")[-1] + "-aggregated.json")
    with open(trace_file) as f:
        trace = json.load(f)
    
    pp_ranks = 1
    for item in trace:
        if item["ph"] == "E" and item["args"]["pp_rank"] + 1 > pp_ranks:
            pp_ranks = item["args"]["pp_rank"] + 1

    reduce_ratios = []
    reduce_windows = []
    iters = []
    for rank in range(pp_ranks):
        reduce_window = []
        reduce_ratio = []
        iter_starts = sorted([item for item in trace if item["name"] == "iteration begin" and item["pid"] == rank], key=lambda item: item["ts"])
        reduce = sorted([item for item in trace if item["name"] == "reduce" and item["pid"] == rank], key=lambda item: item["ts"])
        iter_ends = sorted([item for item in trace if item["name"] == "iteration end" and item["pid"] == rank], key=lambda item: item["ts"])
        for iter in range(1, 20):
            reduce_window.append(iter_ends[iter]["ts"] - reduce[iter]["ts"])
            reduce_ratio.append((iter_ends[iter]["ts"] - reduce[iter]["ts"]) / (iter_ends[iter]["ts"] - iter_starts[iter]["ts"]))
            iters.append(iter_ends[iter]["ts"] - iter_starts[iter]["ts"])
        reduce_windows.append(sum(reduce_window) / len(reduce_window) / 1000)
        reduce_ratios.append(gmean(reduce_ratio))
    aggregated_iters.append(sum(iters) / len(iters) / 1000)
    aggregated_reduce_ratios.append(reduce_ratios)
    aggregated_reduce_windows.append(reduce_windows)
ranks = ["0", "1", "2", "3"]
pp_data = aggregated_reduce_windows[0]
dpp_data = aggregated_reduce_windows[1]
x = np.arange(len(ranks))
width = 0.35

plt.figure(figsize=(10, 6))
plt.bar(x - width / 2, pp_data, width, label='pp')
plt.bar(x + width / 2, dpp_data, width, label='dpp')

plt.xlabel("GPU Number")
plt.ylabel("Reduce Window (ms)")
plt.title(args.trace_dir.split("/")[-1] + " pp vs dpp")
plt.xticks(x, ranks)
plt.legend()
plt.tight_layout()
plt.savefig(f"plots/{args.trace_dir.split('/')[-1]}-reduce-windows-aggregated.png")

pp_data = aggregated_reduce_ratios[0]
dpp_data = aggregated_reduce_ratios[1]
x = np.arange(len(ranks))
width = 0.35

plt.figure(figsize=(10, 6))
plt.bar(x - width / 2, pp_data, width, label='pp')
plt.bar(x + width / 2, dpp_data, width, label='dpp')

plt.xlabel("GPU Number")
plt.ylabel("Reduce Ratio")
plt.title(args.trace_dir.split("/")[-1] + " pp vs dpp")
plt.xticks(x, ranks)
plt.legend()
plt.tight_layout()
plt.savefig(f"plots/{args.trace_dir.split('/')[-1]}-reduce-ratios-aggregated.png")

plt.figure(figsize=(10, 6))
width = 0.2
pp_type = ["pp", "dpp"]
plt.bar(pp_type, aggregated_iters, width)
plt.xlabel("PP Algorithm")
plt.ylabel("Iter Time (ms)")
plt.title(args.trace_dir.split("/")[-1] + " pp vs dpp")
plt.tight_layout()
plt.savefig(f"plots/{args.trace_dir.split('/')[-1]}-iter-time.png")
        