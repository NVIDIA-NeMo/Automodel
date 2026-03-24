For each subdirectory in examples/llm_finetune:

Do the following for each YAML file in the subdirectory.

Rename it according to the scheme <model>_<task>_<dataset>.yaml, where

* <model> is found in the `pretrained_model_name_or_path` field
* <task> is `peft` if there is a `peft` section in the yaml, otherwise `sft`
* if there is a `peft` section and a `quantization` section, the task becomes `qlora`
* <dataset> is what's found after the slash in the `dataset_name` field

If this results in a name collision, add an identifier to the end of the filename (before the extension) to distinguish between the configs.


Now for each YAML file, make sure there's a section at the top with this schema:
```
benchmark:
  description:
  warmup_steps: 20
  max_steps: 200
  num_nodes: 16
  num_gpus: 128
  compare: False # optional
  cadence: 7 # in days: 7 for weekly, 1 for daily, etc.
  fake_balanced_gate: True # for MoE models only
  nsys_start: -1 # optional
  nsys_end: -1 # optional
  nsys_ranks: [] # optional
  overrides: # anything special want to do in benchmarking but not in the main script (hopefully rare)
    param1: e.g., “compile.activated True” 
    param2: 
```

Set num_nodes to 1 and num_gpus to 8 for each file that doesn't already have a benchmark section.
Set the cadence to 1 day for all models undel 20b and to 7 days for all other models.
Omit fake_balanced_gate unless the string "moe" (case insensitive) occurs somewhere in the file. In that case, ask me.
All the fields starting with nsys can be omitted, as can the overrides.
The compare section is optional.
Generate a description for each benchmark the includes the model, the task, dataset, and any other characteristics needed to uniquely id what's going on.

If there's already a benchmark section, don't eliminate any fields that are already there.