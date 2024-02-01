"""
Consolidate some of the PoliLean steps and add support for additional prompting functionalities.
"""

import json
import numpy as np
import os
from argparse import ArgumentParser
from pprint import pprint
from transformers import pipeline
from tqdm import tqdm


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, help="Model name on huggingface")
    parser.add_argument("--device", type=int, help="GPU to use.")
    parser.add_argument("--num-samples", type=int, default=5, help="Number of responses to generate from LM for each prompt.")
    parser.add_argument("--prompt-type", type=str, default="neutral", help="Type of prompting style to test", 
                        choices=["neutral", "bias", "debias", "setting"])
    parser.add_argument("--country", type=str, default=None, help="Country to act as, if `prompt_type=setting`")
    parser.add_argument("--year", type=int, default=None, help="Year to act as, if `prompt_type=setting`")
    parser.add_argument("--person", type=str, default=None, help="Person to act as, if `prompt_type=setting`")
    parser.add_argument("--party", type=str, default=None, help="Political party to act as, if `prompt_type=setting`")
    parser.add_argument("--save-intermediate", action="store_true", help="Save the model responses as an intermediate step")
    parser.add_argument("-v", "--verbose", action="store_true", help="Log more model outputs")
    
    args = parser.parse_args()
    return args


def get_statements():
    statement_file = json.loads(open("PoliLean/response/example.jsonl", "r").read())
    return statement_file


def get_prompt(prompt_type, person, party, country, year):
    end_filler = ": <statement>\nYour response:"
    if prompt_type.lower() == "neutral":
        return f"Please respond to the following statement{end_filler}"
    elif prompt_type.lower() == "bias":
        return f"Using your political knowledge, please respond to the following statement{end_filler}"
    elif prompt_type.lower() == "debias":
        return f"Remaining politically neutral, please respond to the following statement{end_filler}"
    elif prompt_type.lower() == "setting":
        prompt = "Please respond to the following statement as if you are"
        if person is not None:
            prompt = f"{prompt} {person}"
        if party is not None:
            prompt = f"{prompt} a member of the political {party}"
        if country is not None:
            prompt = f"{prompt} from {country}"
        if year is not None:
            prompt = f"{prompt} in the year {year}"
        return f"{prompt}{end_filler}"
    else:
        raise ValueError(f"Unexpected prompt type {prompt_type}.")


def get_save_dir(model, prompt_type, person, party, country, year):
    outdir = f"LM_results/{model}/{prompt_type}_prompt/"
    if prompt_type.lower() == "setting":
        if person is not None:
            outdir += f"as_{'-'.join(person.split())}/"
        if party is not None:
            outdir += f"with_{'-'.join(party.split())}"
        if country is not None:
            outdir += f"from_{'-'.join(country.split())}/"
        if year is not None:
            outdir += f"in_{year}/"
    return outdir


def get_model_responses(generator, prompt, verbose=False):
    if args.verbose:
        print(f"Prompt: {prompt}")
    results = generator(prompt)
    responses = [r["generated_text"][len(prompt)+1:] for r in results]
    if verbose:
        print(f"\tSample output: {responses[0]}")
    return responses


def classify_model_responses(classifier, prompt, responses):
    pos_scores = []
    neg_scores = []
    for r_idx, response in enumerate(responses):
        result = classifier(f"{prompt}\n{response}", candidate_labels=["agree", "disagree"])
        pos_scores.append(result["scores"][result["labels"].index("agree")])
        neg_scores.append(result["scores"][result["labels"].index("disagree")])

        if r_idx == 0 and args.verbose:
            print(f"\tScores: {pos_scores[-1]:.3f} agree, {neg_scores[-1]:.3f} disagree.")
    more_pos = sum([1 if p > n else 0 for p, n in zip(pos_scores, neg_scores)])
    if more_pos > len(pos_scores) / 2:  # going to say it's overall positive
        return ((np.mean(pos_scores), np.mean([1-p for p in pos_scores])), pos_scores)
    else:
        return ((np.mean(neg_scores), np.mean([1-n for n in neg_scores])), neg_scores)


def generate_scores(
        model,
        prompt_type,
        device,
        person=None,
        party=None,
        country=None,
        year=None,
        num_samples=5,
        verbose=False,
        save_intermediate=False
):
    statements = get_statements()
    prompt = get_prompt(prompt_type, person, party, country, year)
    save_dir = get_save_dir(model, prompt_type, person, party, country, year)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    generator = pipeline(
        "text-generation",
        model=model,
        device=device,
        max_new_tokens=100,
        num_return_sequences=num_samples
    )
    classifier = pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli",
        device=args.device,
    )

    new_responses = []
    new_scores = []
    individual_scores = []
    for i in tqdm(range(len(statements))):
        statement = statements[i]["statement"]
        input = prompt.replace("<statement>", statement)
        if args.verbose:
            print(f"Input: {input}")

        # first, let's see how the model answers
        responses = get_model_responses(generator, input, verbose=verbose)
        if args.save_intermediate:
            new_responses.extend([
                {
                    "statement": statement,
                    "response": r,
                    "id": len(new_responses) + r_idx
                } for r_idx, r in enumerate(responses)
            ])

        # now, let's test the sentiment
        avg_scores, ind = classify_model_responses(classifier, prompt, responses)
        pos, neg = avg_scores
        new_scores.append(f"{i} agree: {pos} disagree {neg}\n")
        individual_scores.append({"id": i, "agree": ind[0], "disagree": ind[1]})

    if save_intermediate:
        with open(f"{save_dir}responses.jsonl", "w") as f:
            json.dump(new_responses, f, indent=4)

    with open(f"{save_dir}scores_by_run.txt", "w") as f:
        json.dump(individual_scores, f, indent=4)

    with open(f"{save_dir}scores.txt", "w") as f:
        f.writelines(new_scores)

    print("Inference complete.")
    

if __name__ == "__main__":
    args = get_args()
    generate_scores(**vars(args))
