import json

input_file = "data-1-pipeline-4-tensor-1-pp-aggregated.json"
output_file = "data-1-pipeline-4-tensor-1-pp-aggregated-iter10.json"

with open(input_file, "r") as f:
    data = json.load(f)

filtered_events = [
    event for event in data
    if "args" in event and isinstance(event["args"], dict)
    and event["args"].get("iteration") == 10
]

min_ts = min(event["ts"] for event in filtered_events if "ts" in event)

for event in filtered_events:
    if "ts" in event:
        event["ts"] -= min_ts
    if "tid" in event:
        event["tid"] = 0

with open(output_file, "w") as f:
    json.dump(filtered_events, f, indent=2)

print(f"Saved {len(filtered_events)} events to {output_file}, with timestamps starting at 0")