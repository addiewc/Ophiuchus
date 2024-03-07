import argparse
import json
from pprint import pprint

import step3_testing

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--prompt-file", type=str, help="file defining the prompt types.")
parser.add_argument("-m", "--model", type=str, help="Model to use.")
parser.add_argument("--reverse", action="store_true", help="Prompt with the reverse propositions")
parser.add_argument("-t", "--threshold", type=float, default=0.3,
                    help="the probability threshold between strong and normal (dis)agree")
args = parser.parse_args()


driver = step3_testing.get_driver()

# load the series of prompts to run.
with open(args.prompt_file, "r") as f:
    prompts = json.load(f)

# generate responses and score for each prompt.
for prompt in prompts:
    print("\n-----------Starting:")
    pprint(prompt)

    step3_testing.score_model(
        driver=driver,
        model=args.model,
        threshold=args.threshold,
        reverse=args.reverse,
        **prompt
    )

print("-----------------------------------------------------------------------")
print("All prompts completed.")
