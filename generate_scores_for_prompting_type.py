import argparse
import json
from pprint import pprint

import generate_model_scores

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--prompt-file", type=str, help="file defining the prompt types.")
parser.add_argument("-m", "--model", type=str, help="Model to use.")
parser.add_argument("-d", "--device", type=int, help="GPU number to use")
parser.add_argument("-v", "--verbose", action="store_true", help="Log information.")
parser.add_argument("-s", "--save-intermediate", action="store_true", help="Save intermediate model outputs")
parser.add_argument("--num-samples", type=int, default=5, help="Number of samples to take from LM")
parser.add_argument("--num-repeats", type=int, default=1, help="Number of repeated samples to take from LM")
args = parser.parse_args()

# load the series of prompts to run.
with open(args.prompt_file, "r") as f:
    prompts = json.load(f)

# generate responses and score for each prompt.
for prompt in prompts:
    print("\n-----------Starting:")
    pprint(prompt)

    generate_model_scores.generate_scores(
        model=args.model,
        device=args.device,
        verbose=args.verbose,
        save_intermediate=args.save_intermediate,
        num_samples=args.num_samples,
        num_repeats=args.num_repeats,
        **prompt
    )

print("-----------------------------------------------------------------------")
print("All prompts completed.")