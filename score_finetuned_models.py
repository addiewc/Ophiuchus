"""
Consolidated prompting and scoring for fine-tuned models.
"""

import json
import numpy as np
import os
from argparse import ArgumentParser
from pprint import pprint
from transformers import pipeline
from tqdm import tqdm

from config import HUGGINGFACE_ACCESS_TOKEN


def get_args():
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", type=str, help="Model name of base")
    parser.add_argument("-n", "--name", type=str, help="Extra name for finetuned model")
    parser.add_argument("--device", type=str, help="GPU to use.")
    parser.add_argument("--multi-gpu", action="store_true", help="Let pipelines manage the gpu usage.")
    parser.add_argument("--num-samples", type=int, default=5, help="Number of responses to generate from LM for each prompt.")
    parser.add_argument("--save-intermediate", action="store_true", help="Save the model responses as an intermediate step")
    parser.add_argument("--score-only", action="store_true", help="Use pre-saved results to score the model responses")
    parser.add_argument("--reverse", action="store_true", help="Consider the reverse proposition for all prompts.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Log more model outputs")
    
    args = parser.parse_args()
    assert args.device == "cuda" or args.device == "cpu" or args.device == str(int(args.device)), f"Unexpected device {args.device}"
    if args.device != "cuda" and args.device != "cpu":
        args.device = int(args.device)
    return args


def get_statements(reverse):
    if not reverse:
        statement_file = json.loads(open("PoliLean/response/example.jsonl", "r").read())
    else:
        statement_file = json.loads(open("reverse_statements.json", "r").read())
    return statement_file


def format_prompt(speaker, statement, speaker_type):
    """ This has most of the differences for the non-finetuned models to match our fine-tuning procedure. """
    assert speaker_type in {"person", "party"}

    if speaker_type == "person":
        assert speaker.lower() in {"clinton", "trump"}
        prompt_start = f"Secretary Clinton" if speaker.lower() == "clinton" else f"Mr. Trump"
    else:
        assert speaker.lower() in {"green", "labour", "national", "nz_first"}
        speaker_str = "New Zealand First" if speaker.lower() == "nz_first" else f"{speaker[0].upper()}{speaker[1:].lower()}"
        prompt_start = f"To the honorable representative of the {speaker_str} party"

    prompt = f"[START_HOST] {prompt_start}, how would you respond to the following claim? {statement}. [END_HOST]\n[START_{speaker.upper()}]"
    return prompt


def get_finetuned_save_dir(model, name, speaker):
    outdir = f"LM_results/finetuned_{name}_{model}/{speaker}/"
    return outdir


def get_model_responses(generator, prompt, speaker, verbose=False):
    if verbose:
        print(f"Prompt: {prompt}")
    results = generator(prompt)
    responses = [r["generated_text"][len(prompt)+1:] for r in results]
    truncated_responses = []
    for r in responses:
        if f"[END_{speaker.upper()}]" in r:  # truncate the outputs
            truncated_responses.append("".join(r.split(f"[END_{speaker.upper()}]")[0]))
        else:
            truncated_responses.append(r)
    if verbose:
        print(f"\tSample output: {truncated_responses[0]}")
    return truncated_responses


def classify_model_responses(classifier, prompt, responses, verbose=False, reverse=False):
    pos_scores = []
    neg_scores = []
    for r_idx, response in enumerate(responses):
        result = classifier(f"{prompt} {response}", candidate_labels=["agree", "disagree"])
        if not reverse:
            pos_scores.append(result["scores"][result["labels"].index("agree")])
            neg_scores.append(result["scores"][result["labels"].index("disagree")])
        else:  # correct the reversal here!
            pos_scores.append(result["scores"][result["labels"].index("disagree")])
            neg_scores.append(result["scores"][result["labels"].index("agree")])

        if r_idx == 0 and verbose:
            print(f"\tScores: {pos_scores[-1]:.3f} agree, {neg_scores[-1]:.3f} disagree.")
    more_pos = sum([1 if p > n else 0 for p, n in zip(pos_scores, neg_scores)])
    if more_pos > len(pos_scores) / 2:  # going to say it's overall positive
        return ((np.mean(pos_scores), np.mean([1-p for p in pos_scores])), (pos_scores, [1-p for p in pos_scores]))
    else:
        return (np.mean([1-n for n in neg_scores]), np.mean(neg_scores)), ([1-n for n in neg_scores], neg_scores)


def generate_scores(
        model,
        name,
        device,
        multi_gpu=False,
        num_samples=5,
        num_repeats=1,
        verbose=False,
        save_intermediate=False,
        reverse=False,
        score_only=False,
):
    statements = get_statements(reverse)
    classifier = pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli",
        device=device,
    )

    if "usa" in name.lower():
        speaker_list = ["Clinton", "Trump"]
        speaker_type = "person"
    else:
        assert "new_zealand" in name.lower()
        speaker_list = ["Green", "Labour", "National", "NZ_First"]
        speaker_type = "party"

    for speaker in speaker_list:
        print(f"....working through speaker {speaker}.")
        
        save_dir = get_finetuned_save_dir(model, name, speaker)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    
        if not score_only:
            generator = pipeline(
                "text-generation",
                model=f"finetuning/finetuned_{name}_{model}",
                device=device if not multi_gpu else None,
                max_new_tokens=100,
                num_return_sequences=num_samples,
                token=HUGGINGFACE_ACCESS_TOKEN,
                device_map="auto" if multi_gpu else None,
            )
        else:
            with open(f"{save_dir}{'reverse_' if args.reverse else ''}responses.jsonl", "r") as f:
                model_responses = json.load(f)
            assert len(model_responses) == len(statements) * num_samples, \
                                f"Expected {num_samples} responses for {len(statements)} statements ({len(statements) * num_samples} total) but got {len(model_responses)}."

        new_responses = []
        new_scores = []
        individual_scores = []
        for i in tqdm(range(len(statements))):
            prompt = format_prompt(speaker, statements[i]["statement"], speaker_type)
            if verbose:
                print(f"Input: {input}")
    
            # first, let's see how the model answers
            if not score_only:
                responses = []
                for it in range(num_repeats):
                    responses.extend([r for r in get_model_responses(generator, prompt, speaker=speaker, verbose=verbose)])
                if save_intermediate:
                    new_responses.extend([
                        {
                            "statement": statements[i]["statement"],
                            "response": r,
                            "id": len(new_responses) + r_idx
                        } for r_idx, r in enumerate(responses)
                    ])
            else:  # load earlier results
                start_idx = i * num_samples
                responses = [r["response"] for r in model_responses[start_idx: start_idx + num_samples]]

            # now, let's test the sentiment
            avg_scores, ind = classify_model_responses(classifier, statements[i]["statement"], responses, verbose=verbose, reverse=reverse)
            pos, neg = avg_scores
            new_scores.append(f"{i} agree: {pos} disagree {neg}\n")
            individual_scores.append({"id": i, "agree": ind[0], "disagree": ind[1]})
    
        if save_intermediate:
            with open(f"{save_dir}{'reverse_' if reverse else ''}responses.jsonl", "w") as f:
                json.dump(new_responses, f, indent=4)
    
        with open(f"{save_dir}{'reverse_' if reverse else ''}scores_by_run.txt", "w") as f:
            json.dump(individual_scores, f, indent=4)
    
        with open(f"{save_dir}{'reverse_' if reverse else ''}scores.txt", "w") as f:
            f.writelines(new_scores)

    print("Inference complete.")
    

if __name__ == "__main__":
    args = get_args()
    generate_scores(**vars(args))
