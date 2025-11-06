import json

with open("data-1-pipeline-4-tensor-1-dpp-aggregated-iter10.json") as f1, open("data-1-pipeline-4-tensor-1-pp-aggregated-iter10.json") as f2:
    trace1 = json.load(f1)
    trace2 = json.load(f2)

merged = trace1 + trace2

with open("merged_trace.json", "w") as f:
    json.dump(merged, f)