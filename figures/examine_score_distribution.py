import argparse
import json
import matplotlib.pyplot as plt

import sys
import os
sys.path.append(os.path.join(sys.path[0], "../"))
import generate_model_scores
from figures import plot_utils


parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=str, help="Model to consider")
parser.add_argument("--prompt-type", type=str, help="Type of prompt to use", choices=["neutral", "bias", "debias", "setting"])
parser.add_argument("--person", type=str, default=None, help="Person to act as in the setting prompt")
parser.add_argument("--party", type=str, default=None, help="Political party to act as in the setting prompt")
parser.add_argument("--country", type=str, default=None, help="Country to act as in the setting prompt")
parser.add_argument("--year", type=str, default=None, help="Year to act from in the setting prompt")
args = parser.parse_args()

save_dir = generate_model_scores.get_save_dir(**vars(args))

# first, let's look at the average
avg_positives = []
with open(f"{save_dir}scores.txt", "r") as f:
    for line in f.readlines():
        opts = line.split()
        avg_positives.append(float(opts[2]))

ax = plt.subplot(1, 1, 1)
ax.hist(avg_positives, bins=15, color="#d6604d")
ax.axvline(x=0.5, color="black", linestyle="dashed")
ax.axvline(x=0.65, color="black", linestyle="dotted")
ax.axvline(x=0.35, color="black", linestyle="dotted")
ax.set_xlabel("Agreement")
ax.set_ylabel("Count")
plot_utils.format_ax(ax)
plt.savefig(f"{save_dir}average_score_distribution.pdf", bbox_inches="tight")
plt.close()


# now, let's look at the individual prompt
with open(f"{save_dir}scores_by_run.txt", "r") as f:
    scores = json.load(f)

full_score_dist = []
for s in scores:
    full_score_dist.extend(s["agree"])

ax = plt.subplot(1, 1, 1)
ax.hist(full_score_dist, bins=30, color="#d6604d")
ax.axvline(x=0.5, color="black", linestyle="dashed")
ax.axvline(x=0.65, color="black", linestyle="dotted")
ax.axvline(x=0.35, color="black", linestyle="dotted")
ax.set_xlabel("Agreement")
ax.set_ylabel("Count")
plot_utils.format_ax(ax)
plt.savefig(f"{save_dir}individual_score_distribution.pdf", bbox_inches="tight")
plt.close()

