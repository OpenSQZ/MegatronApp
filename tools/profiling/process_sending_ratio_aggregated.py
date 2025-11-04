import argparse
import json
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.stats import gmean

parser = argparse.ArgumentParser(description="Process trace file and generate plots.")
parser.add_argument("--trace_dir", help="Directory containing trace (required)", required=True)
args = parser.parse_args()
aggregated_average_sending_ratios = []
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
    for rank in range(pp_ranks):
        forward_compute_ends_by_rank.append(
            sorted(
                [item for item in trace if item["ph"] == "E" and item["args"]["pp_rank"] == rank and item["args"]["iteration"] > 0 and item["name"] == "forward"],
                key=lambda item: item["ts"]
            )
        )
        forward_compute_starts_by_rank.append(
            sorted(
                [item for item in trace if item["ph"] == "B" and item["name"] == "forward" and item["pid"] == rank and item["args"]["iteration"] > 0],
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
                [item for item in trace if item["ph"] == "B" and item["name"] == "backward" and item["pid"] == rank and item["args"]["iteration"] > 0],
                key=lambda item: item["ts"]
            )
        )
        iter_starts = sorted([item for item in trace if item["name"] == "iteration begin" and item["pid"] == rank], key=lambda item: item["ts"])
        iter_ends = sorted([item for item in trace if item["name"] == "iteration end" and item["pid"] == rank], key=lambda item: item["ts"])
    avg_sending_ratios = []
    for forward_send_rank in range(pp_ranks):
        sending_ratios = []
        for iter in range(1, 20):
            sending_window = 0
            for sender_ends in forward_compute_ends_by_rank[forward_send_rank]:
                for i, receiver_ends in enumerate(forward_compute_ends_by_rank[(forward_send_rank + 1) % pp_ranks]):
                    if receiver_ends["args"]["iteration"] == sender_ends["args"]["iteration"] and receiver_ends["args"]["microbatch"] == sender_ends["args"]["microbatch"] and receiver_ends["args"]["model_chunk"] == sender_ends["args"]["model_chunk"] + int(forward_send_rank == pp_ranks - 1) and receiver_ends["args"]["iteration"] == iter:
                        assert forward_compute_starts_by_rank[(forward_send_rank + 1) % pp_ranks][i]["ts"] - sender_ends["ts"] >=0
                        sending_window += forward_compute_starts_by_rank[(forward_send_rank + 1) % pp_ranks][i]["ts"] - sender_ends["ts"]
            sending_ratios.append(sending_window / (iter_ends[iter]["ts"] - iter_starts[iter]["ts"]))
        avg_sending_ratios.append(gmean(sending_ratios))
    for backward_send_rank in range(pp_ranks):
        sending_ratios = []
        for iter in range(1, 20):
            sending_window = 0
            for sender_ends in backward_compute_ends_by_rank[backward_send_rank]:
                for i, receiver_ends in enumerate(backward_compute_ends_by_rank[(backward_send_rank - 1) % pp_ranks]):
                    if receiver_ends["args"]["iteration"] == sender_ends["args"]["iteration"] and receiver_ends["args"]["microbatch"] == sender_ends["args"]["microbatch"] and receiver_ends["args"]["model_chunk"] == sender_ends["args"]["model_chunk"] - int(backward_send_rank == 0) and receiver_ends["args"]["iteration"] == iter:
                        assert backward_compute_starts_by_rank[(backward_send_rank - 1) % pp_ranks][i]["ts"] - sender_ends["ts"] >=0
                        sending_window += backward_compute_starts_by_rank[(backward_send_rank - 1) % pp_ranks][i]["ts"] - sender_ends["ts"]
            sending_ratios.append(sending_window / (iter_ends[iter]["ts"] - iter_starts[iter]["ts"]))
        avg_sending_ratios.append(gmean(sending_ratios))
    aggregated_average_sending_ratios.append(avg_sending_ratios)
pairs = ["0-1", "1-2", "2-3", "3-0", "0-3", "1-0", "2-1", "3-2"]
pp_data = aggregated_average_sending_ratios[0]
dpp_data = aggregated_average_sending_ratios[1]
x = np.arange(len(pairs))
width = 0.35
plt.figure(figsize=(10, 6))
plt.bar(x - width / 2, pp_data, width, label='pp')
plt.bar(x + width / 2, dpp_data, width, label='dpp')

plt.xlabel("GPU pairs (sender rank)-(receiver rank)")
plt.ylabel("Sending Ratio")
plt.title(args.trace_dir.split("/")[-1] + " pp vs dpp")
plt.xticks(x, pairs)
plt.legend()
plt.tight_layout()
plt.savefig(f"plots/{args.trace_dir.split('/')[-1]}-sending-ratios-aggregated.png")
