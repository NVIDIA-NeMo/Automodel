# Run on Your Local Workstation

When launching examples locally, `uv` will automatically download and keep your local environment in sync with the package's requirements. 

For example, to finetune a small `Qwen3` model on the HellaSwag dataset, simply run:

```sh
uv run recipes/llm/finetune.py --model.pretrained_model_name_or_path Qwen/Qwen3-0.6B
```

To finetune a slightly larger model on multiple GPUs and sharded using FSDP2, you can make a slight augmentation to the above command. For example, on 2 GPUs simply run:

```sh
uv run torchrun --nproc-per-node=2 recipes/llm/finetune.py --model.pretrained_model_name_or_path Qwen/Qwen3-1.7B
```

`finetune.py` uses the [default config](https://github.com/NVIDIA-NeMo/Automodel/blob/main/recipes/llm/llama_3_2_1b_hellaswag.yaml) file. You can easily customize the config by passing in command-line arguments, editing the config directly, or creating your own configuration file and passing it using `--config /path/to/config`.