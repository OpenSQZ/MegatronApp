import argparse
import re
import matplotlib.pyplot as plt
from collections import defaultdict

pp_types = ["pp", "dpp"]
parser = argparse.ArgumentParser(description="Process trace file and generate plots.")
parser.add_argument("--trace_dir", help="Directory containing trace (required)", required=True)
args = parser.parse_args()

rank_colors = {
    0: 'tab:blue',
    1: 'tab:orange',
    2: 'tab:green',
    3: 'tab:red'
}

plt.figure(figsize=(12, 6))

for pp_type in pp_types:
    with open(f"{args.trace_dir}-{pp_type}.txt", "r") as f:
        text = f.read()

    pattern = r"rank (\d+) allocated peak memory ([\d\.]+)MB"
    matches = re.findall(pattern, text)

    rank_memory = defaultdict(list)
    for rank, mem in matches:
        rank = int(rank)
        mem = float(mem)
        rank_memory[rank].append(mem)

    linestyle = '--' if pp_type == "pp" else '-'

    for rank, mem_list in sorted(rank_memory.items()):
        iter_idx = list(range(1, len(mem_list) + 1))
        plt.plot(
            iter_idx,
            mem_list,
            label=f"Rank {rank} ({pp_type})",
            linestyle=linestyle,
            marker='o',
            color=rank_colors.get(rank, None)
        )

plt.xlabel("Iteration Index")
plt.ylabel("Peak Memory (MB)")
plt.title("GPU Peak Memory Usage per Rank")
plt.xticks(range(1, 21))
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.ylim(bottom=0)
plt.savefig(f"plots/{args.trace_dir.split('/')[-1]}.png")