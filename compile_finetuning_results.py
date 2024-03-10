import pandas as pd
import numpy as np
import json


MODELS = ["meta-llama/Llama-2-7b-hf", "meta-llama/Llama-2-7b-chat-hf"]
NAMES = {"Trump": "Donald Trump", "Clinton": "Hillary Clinton"}
PARTIES = {"Green": "Green Party", "Labour": "Labour Party", "National": "National Party", "NZ_First": "New Zealand First Party"}

compiled_results = {
    "model": [], "prompt": [],
    "average_economic_score": [], "average_social_score": [],
    "forward_economic_score": [], "forward_social_score": [],
    "reverse_economic_score": [], "reverse_social_score": []
}

for model in MODELS:
    for name in NAMES:
        with open(f"LM_results/finetuned_usa_debates_{model}/{name}/political_compass.json", "r") as f:
            res_fwd = json.load(f)
        with open(f"LM_results/finetuned_usa_debates_{model}/{name}/reverse_political_compass.json", "r") as f:
            res_rvs = json.load(f)

        compiled_results["model"].append(model)
        compiled_results["prompt"].append(NAMES[name])
        compiled_results["forward_economic_score"].append(res_fwd["economic_score"])
        compiled_results["forward_social_score"].append(res_fwd["social_score"])
        compiled_results["reverse_economic_score"].append(res_rvs["economic_score"])
        compiled_results["reverse_social_score"].append(res_rvs["social_score"])
        compiled_results["average_economic_score"].append(np.mean([r["economic_score"] for r in [res_fwd, res_rvs]]))
        compiled_results["average_social_score"].append(np.mean([r["social_score"] for r in [res_fwd, res_rvs]]))
    for party in PARTIES:
        with open(f"LM_results/finetuned_new_zealand_{model}/{party}/political_compass.json", "r") as f:
            res_fwd = json.load(f)
        with open(f"LM_results/finetuned_new_zealand_{model}/{party}/reverse_political_compass.json", "r") as f:
            res_rvs = json.load(f)

        compiled_results["model"].append(model)
        compiled_results["prompt"].append(PARTIES[party])
        compiled_results["forward_economic_score"].append(res_fwd["economic_score"])
        compiled_results["forward_social_score"].append(res_fwd["social_score"])
        compiled_results["reverse_economic_score"].append(res_rvs["economic_score"])
        compiled_results["reverse_social_score"].append(res_rvs["social_score"])
        compiled_results["average_economic_score"].append(np.mean([r["economic_score"] for r in [res_fwd, res_rvs]]))
        compiled_results["average_social_score"].append(np.mean([r["social_score"] for r in [res_fwd, res_rvs]]))

df = pd.DataFrame.from_dict(compiled_results)
print(df.head())
df.to_csv("political_compass_finetuning_results.csv")

