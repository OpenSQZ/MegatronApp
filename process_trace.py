import json
import re
from collections import defaultdict
import argparse

parser = argparse.ArgumentParser(description="pipeline and model chunks")
parser.add_argument("--pipeline_parallel", required=True, help="pipeline parallel size")
parser.add_argument("--model_chunks", required=True, help="number of model chunks")
args = parser.parse_args()

file_path = f"trace_pipeline_{args.pipeline_parallel}_model_chunks_{args.model_chunks}.txt"
with open(file_path, "r") as file:
    data = file.read()

lines = data.strip().split("\n")

pipeline = int(args.pipeline_parallel)
num_model_chunks = int(args.model_chunks)

gpu_batch_data = defaultdict(lambda: defaultdict(list))

events = []

for line in lines:
    matches = re.findall(r"(\d+), (\d+), (\w+ \w+), (\d+\.\d+)%", line)
    for match in matches:
        gpu_id = int(match[0]) % 10
        batch_id = int(match[1])
        action = match[2]
        timestamp = float(match[3])
        if action == "can reduce":
            events.append(
                {
                    "name": f"model chunk {batch_id} can reduce",
                    "cname": "thread_state_unknown",
                    "ph": "I",
                    "ts": int(timestamp * 1000),
                    "pid": gpu_id,
                    "tid": 0,
                    "style": {"color": "black"},
                }
            )
        else:
            gpu_batch_data[gpu_id][batch_id].append(
                {"action": action, "timestamp": timestamp}
            )
    matches = re.findall(r"(rank) (\d) (\w+ \w+ \w+ \w+) (\d+\.\d+)%", line)
    for match in matches:
        gpu_id = int(match[1])
        action = match[2]
        timestamp = float(match[3])
        if action == "iteration starts at timestamp":
            events.append(
                {
                    "name": f"gpu {gpu_id} iter starts",
                    "cname": "thread_state_unknown",
                    "ph": "I",
                    "ts": int(timestamp * 1000),
                    "pid": gpu_id,
                    "tid": 0,
                    "style": {"color": "black"},
                }
            )
        else:
            events.append(
                {
                    "name": f"gpu {gpu_id} iter ends",
                    "cname": "thread_state_unknown",
                    "ph": "I",
                    "ts": int(timestamp * 1000),
                    "pid": gpu_id,
                    "tid": 0,
                    "style": {"color": "black"},
                }
            )

for gpu_id, batches in gpu_batch_data.items():
    for batch_id, actions in batches.items():
        forward_start = None
        backward_start = None

        for i, event in enumerate(actions):
            action = event["action"]
            timestamp = event["timestamp"]

            if action == "forward start" and forward_start is None:
                forward_start = timestamp
                cur_chunk = batch_id % (num_model_chunks * pipeline) // pipeline
                cur_batch = (
                    batch_id // (num_model_chunks * pipeline) * pipeline
                    + batch_id % pipeline
                )
                events.append(
                    {
                        "name": f"forward {cur_chunk}-{cur_batch}",
                        "cname": "thread_state_unknown",
                        "ph": "B",
                        "ts": int(forward_start * 1000),
                        "pid": gpu_id,
                        "tid": 0,
                        "style": {"color": "black"},
                    }
                )
            elif action == "backward start" and backward_start is None:
                backward_start = timestamp
                cur_chunk = batch_id % (num_model_chunks * pipeline) // pipeline
                cur_batch = (
                    batch_id // (num_model_chunks * pipeline) * pipeline
                    + batch_id % pipeline
                )
                events.append(
                    {
                        "name": f"backward {cur_chunk}-{cur_batch}",
                        "cname": "thread_state_unknown",
                        "ph": "B",
                        "ts": int(backward_start * 1000),
                        "pid": gpu_id,
                        "tid": 0,
                        "style": {"color": "pink"},
                    }
                )
            elif action == "forward finished" and forward_start is not None:
                cur_chunk = batch_id % (num_model_chunks * pipeline) // pipeline
                cur_batch = (
                    batch_id // (num_model_chunks * pipeline) * pipeline
                    + batch_id % pipeline
                )
                events.append(
                    {
                        "name": f"forward {cur_chunk}-{cur_batch}",
                        "cname": "thread_state_unknown",
                        "ph": "E",
                        "ts": int(timestamp * 1000),
                        "pid": gpu_id,
                        "tid": 0,
                        "style": {"color": "black"},
                    }
                )
                forward_start = None

            elif action == "backward finished" and backward_start is not None:
                cur_chunk = batch_id % (num_model_chunks * pipeline) // pipeline
                cur_batch = (
                    batch_id // (num_model_chunks * pipeline) * pipeline
                    + batch_id % pipeline
                )
                events.append(
                    {
                        "name": f"backward {cur_chunk}-{cur_batch}",
                        "cname": "thread_state_unknown",
                        "ph": "E",
                        "ts": int(timestamp * 1000),
                        "pid": gpu_id,
                        "tid": 0,
                        "style": {"color": "pink"},
                    }
                )
                backward_start = None
            else:
                print(gpu_id, batch_id, action)
                print("ERROR")

result = events

output_file_path = f"{file_path[:-4]}.json"
with open(output_file_path, "w") as f:
    json.dump(result, f, indent=4)

print("JSON file created successfully.")
