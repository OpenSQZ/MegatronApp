import argparse
import json
import matplotlib.pyplot as plt

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
avg_sending_windows = []
total_window = 0
for forward_send_rank in range(pp_ranks):
    sending_windows = []
    for sender_ends in forward_compute_ends_by_rank[forward_send_rank]:
        for i, receiver_ends in enumerate(forward_compute_ends_by_rank[(forward_send_rank + 1) % pp_ranks]):
            if receiver_ends["args"]["iteration"] == sender_ends["args"]["iteration"] and receiver_ends["args"]["microbatch"] == sender_ends["args"]["microbatch"] and receiver_ends["args"]["model_chunk"] == sender_ends["args"]["model_chunk"] + int(forward_send_rank == pp_ranks - 1):
                assert forward_compute_starts_by_rank[(forward_send_rank + 1) % pp_ranks][i]["ts"] - sender_ends["ts"] >=0
                sending_windows.append(forward_compute_starts_by_rank[(forward_send_rank + 1) % pp_ranks][i]["ts"] - sender_ends["ts"])
    avg_sending_windows.append(sum(sending_windows) / len(sending_windows))
for backward_send_rank in range(pp_ranks):
    sending_windows = []
    for sender_ends in backward_compute_ends_by_rank[backward_send_rank]:
        for i, receiver_ends in enumerate(backward_compute_ends_by_rank[(backward_send_rank - 1) % pp_ranks]):
            if receiver_ends["args"]["iteration"] == sender_ends["args"]["iteration"] and receiver_ends["args"]["microbatch"] == sender_ends["args"]["microbatch"] and receiver_ends["args"]["model_chunk"] == sender_ends["args"]["model_chunk"] - int(backward_send_rank == 0):
                assert backward_compute_starts_by_rank[(backward_send_rank - 1) % pp_ranks][i]["ts"] - sender_ends["ts"] >=0
                sending_windows.append(backward_compute_starts_by_rank[(backward_send_rank - 1) % pp_ranks][i]["ts"] - sender_ends["ts"])
    avg_sending_windows.append(sum(sending_windows) / len(sending_windows))
pairs = ["0-1", "1-2", "2-3", "3-0", "0-3", "1-0", "2-1", "3-2"]
plt.bar(pairs, avg_sending_windows)
plt.savefig(f"plots/{trace_file.split("/")[-1][:-5]}-sending-windows.png")