import pandas as pd
import numpy as np
import json

from generate_model_scores import get_save_dir

MODELS = [
    "gpt2", "bert-base-uncased", "bert-large-uncased", "facebook/bart-base", "facebook/bart-large",
    "roberta-base", "roberta-large", "microsoft/phi-2", "google/gemma-2b", "google/gemma-7b",
    "meta-llama/Llama-2-7b-hf", "meta-llama/Llama-2-7b-chat-hf"
]
PROMPTS = ["basic", "usa", "new_zealand"]

result_dict = {
    "model": [],
    "prompt_name": [],
    "forward_economic_score": [],
    "forward_social_score": [],
    "reverse_economic_score": [],
    "reverse_social_score": []
}
for model in MODELS:
    for prompt_base in PROMPTS:
        with open(f"prompting_types/{prompt_base}.json", "r") as f:
            all_prompts = json.load(f)

        for prompt in all_prompts:
            result_dict["model"].append(model)
            result_dict["prompt_name"].append(
                prompt["person"] if "person" in prompt else prompt["party"] if "party" in prompt else prompt["prompt_type"]
            )

            with open(f"{get_save_dir(model, **prompt)}political_compass.json", "r") as f:
                forward_res = json.load(f)
            result_dict["forward_economic_score"].append(float(forward_res["economic_score"]))
            result_dict["forward_social_score"].append(float(forward_res["social_score"]))

            with open(f"{get_save_dir(model, **prompt)}reverse_political_compass.json", "r") as f:
                reverse_res = json.load(f)
            result_dict["reverse_economic_score"].append(float(reverse_res["economic_score"]))
            result_dict["reverse_social_score"].append(float(reverse_res["social_score"]))

result_df = pd.DataFrame.from_dict(result_dict)
result_df.insert(2, "average_economic_score",
                 [np.mean([row.forward_economic_score, row.reverse_economic_score]) for row in result_df.itertuples()])
result_df.insert(3, "average_social_score",
                 [np.mean([row.forward_social_score, row.reverse_social_score]) for row in result_df.itertuples()])

print(result_df.head())
result_df.to_csv("political_compass_results.csv")
