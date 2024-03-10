# Ophiuchus
Implementation of CSE599J course project testing the political biases in LMs. This repository is adopted from and
closely based on the PoliLean repository (https://github.com/BunsenFeng/PoliLean).

## Requirements
This repository requires installation of the PoliLean repository (found at https://github.com/BunsenFeng/PoliLean)
as a git submodule. Furthermore, you must create a `config.py` file containing `HUGGINGFACE_ACCESS_TOKEN` with your
access token for restricted models.

## Reproducing experimental results
### Out-of-the-box LMs
The experiments on out-of-the-box LMs contained in the report can be reproduced as follows.
```
python generate_scores_for_prompting_type.py -f prompting_types/{prompt}.json -m {hf_model} -d {gpu_num} --num-samples {N_sam}
python generate_scores_for_prompting_type.py --reverse -f prompting_types/{prompt}.json -m {hf_model} -d {gpu_num} --num-samples {N_sam}
python generate_political_compass_results_for_prompting_type.py -f prompting_types{prompt}.json -m {hf_model}
python generate_political_compass_results_for_prompting_type.py --reverse -f prompting_types{prompt}.json -m {hf_model}
```
Here, `prompt` is one of basic (encompassing the default, neutralizing/debiasing, and poltiicizing/biasing prompts),
`usa` (encompassing the 2016 US presidential candidates), or `new_zealand` (encompassing several political parites from
the 2017 NZ general election); `hf_model` is the name on the model on HuggingFace, `gpu_num` gives the GPU to use (or 
"cpu" to run without GPU or "cuda" to run on multiple GPUs, in which case the `--multi-gpu` flag should also be set),
and `{N_sam}` is the number of sample generations to use (to reproduce results, `N_sam = 5` for GPT-2 and Llama-2 models,
`4` for the BART models, and `1` for all other models).

After running this for all models and prompts, we compile final results with
```
python compile_results.py
```

###Fine-tuned LMs
The experiments fine-tuning Llama-2 LMs on political debate transcripts can be reproduced as follows.
```
python finetuning/finetune_models_on_political_debates.py -m {hg_model} --dataset {ds_path} -n {ds_name} -d {gpu_num} --num_epochs {epochs} --accumulation-steps 16 --scheduler cosine
python score_finetuned_models.py -m {hg_model} -n {ds_name} --device {gpu_num} --num-samples {N_sam}
python score_finetuned_models.py --reverse -m {hg_model} -n {ds_name} --device {gpu_num} --num-samples {N_sam}
python generate_political_compass_results_for_finetuned_models.py -m {hg_model} -n {ds_name}
python generate_political_compass_results_for_finetuned_models.py --reverse -m {hg_model} -n {ds_name}
```
Here, `ds_path=base_data/usa_presidential_debates/processed_data/` and `ds_name=usa` for US presidential debate data,
and `ds_path=base_data/nz_hansard_reports/processed_data/` and `ds_name=new_zealand` for NZ parliamentary debate data.

After running this for all models and prompts, we compile the fine-tuned results with
```
python compile_finetuning_results.py
```

Additional functionality can be explored using `--help`, such as the `--save-intermediate` flag to store model responses
to prompt statements.
