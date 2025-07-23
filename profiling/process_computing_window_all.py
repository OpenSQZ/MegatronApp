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
ranks = ["0", "1", "2", "3"]
avg_forward_compute_windows = []
avg_backward_compute_windows = []
for process_rank in range(0, pp_ranks, 2):
    forward_compute_ends_by_rank.append(
        sorted(
            [item for item in trace if item["ph"] == "E" and item["pid"] == process_rank and item["args"]["iteration"] > 0 and item["name"] == "forward"],
            key=lambda item: item["ts"]
        )
    )
    forward_compute_starts_by_rank.append(
        sorted(
            [item for item in trace if item["ph"] == "B" and item["pid"] == process_rank and item["args"]["iteration"] > 0 and item["name"] == "forward"],
            key=lambda item: item["ts"]
        )
    )
    backward_compute_ends_by_rank.append(
        sorted(
            [item for item in trace if item["ph"] == "E" and item["pid"] == process_rank and item["args"]["iteration"] > 0 and item["name"] == "backward"],
            key=lambda item: item["ts"]
        )
    )
    backward_compute_starts_by_rank.append(
        sorted(
            [item for item in trace if item["ph"] == "B" and item["pid"] == process_rank and item["args"]["iteration"] > 0 and item["name"] == "backward"],
            key=lambda item: item["ts"]
        )
    )
    forward_compute_times = 0
    backward_compute_times = 0
    rank = process_rank // 2
    for i in range(len(forward_compute_ends_by_rank[rank])):
        assert forward_compute_ends_by_rank[rank][i]["ts"] >= forward_compute_starts_by_rank[rank][i]["ts"]
        forward_compute_times += (forward_compute_ends_by_rank[rank][i]["ts"] - forward_compute_starts_by_rank[rank][i]["ts"])
    for i in range(len(backward_compute_ends_by_rank[rank])):
        assert forward_compute_ends_by_rank[rank][i]["ts"] >= forward_compute_starts_by_rank[rank][i]["ts"]
        backward_compute_times += (backward_compute_ends_by_rank[rank][i]["ts"] - backward_compute_starts_by_rank[rank][i]["ts"])
    if forward_compute_starts_by_rank[rank]:
        avg_forward_compute_windows.append(forward_compute_times / len(forward_compute_starts_by_rank[rank]) / 1000)
    if backward_compute_starts_by_rank[rank]:
        avg_backward_compute_windows.append(backward_compute_times / len(backward_compute_starts_by_rank[rank]) / 1000)
if avg_forward_compute_windows:
    print(f"forward: {sum(avg_forward_compute_windows) / len(avg_forward_compute_windows)} ms")
if avg_backward_compute_windows:
    print(f"backward: {sum(avg_backward_compute_windows) / len(avg_backward_compute_windows)} ms")
