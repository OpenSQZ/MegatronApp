import argparse
import json
import matplotlib.pyplot as plt
from scipy.stats import gmean

dpp_iter_times = [1048.5, 975.2, 963.2, 934.4, 954.8, 941.4, 935.7, 933.1, 936.8, 958.1, 957.3, 946.6, 942.1, 936.3, 936.5, 1008.1, 947.4, 968.9, 958.3]
pp_iter_times = [757.5, 713.2, 714.6, 717.3, 716.4, 717.2, 716.7, 716.8, 717.0, 718.2, 716.8, 715.7, 716.4, 717.7, 716.5, 786.2, 741.6, 741.3, 742.9]

parser = argparse.ArgumentParser(description="Process trace file and generate plots.")
parser.add_argument("--trace_file", help="File containing trace (required)", required=True)
args = parser.parse_args()
trace_file = args.trace_file
with open(trace_file) as f:
    trace = json.load(f)

iter_times = dpp_iter_times if "dpp" in args.trace_file else pp_iter_times

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
        sending_ratios.append(sending_window / iter_times[iter - 1] / 1000)
    avg_sending_ratios.append(gmean(sending_ratios))
pairs = ["0-1", "1-2", "2-3", "3-0", "0-3", "1-0", "2-1", "3-2"]
plt.bar(pairs, avg_sending_ratios)
plt.xlabel("GPU pairs (sender rank)-(receiver rank)")
plt.ylabel("Send Ratio")
plt.title(trace_file.split("/")[-1][:-5])
plt.savefig(f"plots/{trace_file.split("/")[-1][:-5]}-sending-ratios.png")