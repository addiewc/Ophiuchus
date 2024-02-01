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
args = parser.parse_args()

# load the series of prompts to run.
with open(args.config_file, "r") as f:
    prompts = json.load(f)

for prompt in prompts:
    print("\n-----------Starting:")
    pprint(prompt)

    generate_model_scores.generate_scores(
        model=args.model,
        device=args.device,
        verbose=args.verbose,
        save_intermediate=args.save_intermediate,
        **prompt
    )

print("-----------------------------------------------------------------------")
print("All prompts completed.")
