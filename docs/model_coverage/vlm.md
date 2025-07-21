# Vision Language Models with NeMo AutoModel

## Introduction

Vision Language Models (VLMs) are advanced models that integrate vision and language processing capabilities. They are trained on extensive datasets containing both interleaved images and text data, allowing them to generate text descriptions of images and answer questions related to images.

NeMo AutoModel LLM APIs can be easily extended to support VLM tasks. While most of the training setup is the same, some additional steps are required to prepare the data and model for VLM training.

In this guide, we will walk through the data preparation steps for two datasets and also provide a table of scripts and configurations that have been tested with NeMo AutoModel. The code for both the datasets is available in `Nemo Repository <https://github.com/NVIDIA/NeMo/blob/main/scripts/vlm/automodel_datasets.py>`__.

## Run LLMs with NeMo Automodel

To run LLMs with NeMo AutoModel, please use at least version `25.07` of the NeMo container.
If the model you want to finetune is available on a newer version of transformers, you may need
to upgrade to the latest NeMo Automodel with:

.. code-block:: bash

   pip3 install --upgrade git+git@github.com:NVIDIA-NeMo/Automodel.git

For other installation options (e.g., uv) please see our [installation guide]([a link](https://github.com/NVIDIA-NeMo/Automodel/blob/main/docs/guides/installation.md)

## Supported Models


While most VLM models from Hugging Face are compatible with NeMo AutoModel, we have specifically tested the following models (e.g., Gemma 3, Qwen 2, etc.) for convergence with the datasets mentioned above. You can find the script for running these models in the `NeMo repository <https://github.com/NVIDIA/NeMo/blob/main/scripts/vlm/automodel.py>`__.

| Model                              | Dataset                     | FSDP2      | PEFT       |
|------------------------------------|-----------------------------|------------|------------|
| Gemma 3-4B & 27B                   | naver-clova-ix & rdr-items  | Supported  | Supported  |
| Gemma 3n                           | naver-clova-ix & rdr-items  | Supported  | Supported  |
| Qwen2-VL-2B-Instruct & Qwen2.5-VL-3B-Instruct | cord-v2          | Supported  | Supported  |
| llava-v1.6                         | cord-v2 & naver-clova-ix    | Supported  | Supported  |

## rdr items dataset

The `rdr items dataset <https://huggingface.co/datasets/quintend/rdr-items>`__ is a small dataset containing 48 images with descriptions. To make sure the data is in correct format, we apply a collate function with user instructions to describe the image.

.. code-block:: python

        def collate_fn(examples, processor):
            def fmt(sample):
                instruction = "Describe accurately the given image."
                conversation = [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": instruction}, {"type": "image", "image": sample["image"]}],
                    },
                    {"role": "assistant", "content": [{"type": "text", "text": sample["text"]}]},
                ]
                return {"conversation": conversation, "images": [sample['image'].convert("RGB")]}

            text = []
            images = []
            for example in map(fmt, examples):

                text.append(
                    processor.apply_chat_template(example["conversation"],tokenize=False,add_generation_prompt=False,)
                )
                images += example['images']

            # Tokenize the text and process the images
            batch = processor(text=text,images=images,padding=True,return_tensors="pt",)

            batch["pixel_values"] = batch["pixel_values"].to(torch.bfloat16)

            labels = batch["input_ids"].clone()
            # Skipped tokens are the padding tokens for both image and text.
            labels[torch.isin(labels, skipped_tokens)] = -100
            batch["labels"] = labels
            return batch

This code block ensures that the images are processed correctly, and the text is tokenized along with the chat template.


## cord-v2 dataset

The `cord-v2 dataset <https://huggingface.co/datasets/naver-clova-ix/cord-v2>`__ is a dataset containing receipts with descriptions in JSON format. 
To ensure the data is in the correct format, we apply the following function to convert the input JSON to text tokens.

.. code-block:: python

    def json2token(obj, sort_json_key: bool = True):
        """
        Convert an ordered JSON object into a token sequence
        """
        if type(obj) == dict:
            if len(obj) == 1 and "text_sequence" in obj:
                return obj["text_sequence"]
            else:
                output = ""
                if sort_json_key:
                    keys = sorted(obj.keys(), reverse=True)
                else:
                    keys = obj.keys()
                for k in keys:
                    output += (
                            fr"<s_{k}>"
                            + json2token(obj[k], sort_json_key)
                            + fr"</s_{k}>"
                    )
                return output
        elif type(obj) == list:
            return r"<sep/>".join(
                [json2token(item, sort_json_key) for item in obj]
            )
        else:
            obj = str(obj)
            return obj

Below is an example input and output of the above function:

.. code-block:: python

    input:

    {'menu': [{'nm': 'Nasi Campur Bali', 'cnt': '1 x', 'price': '75,000'},  {'nm': 'Bebek Street', 'cnt': '1 x', 'price': '44,000'}], 'sub_total': {'subtotal_price': '1,346,000', 'service_price': '100,950', 'tax_price': '144,695', 'etc': '-45'}, 'total': {'total_price': '1,591,600'}}

    output:

    <s_total><s_total_price>1,591,600</s_total_price></s_total><s_sub_total><s_tax_price>144,695</s_tax_price><s_subtotal_price>1,346,000</s_subtotal_price><s_service_price>100,950</s_service_price><s_etc>-45</s_etc></s_sub_total><s_menu><s_price>75,000</s_price><s_nm>Nasi Campur Bali</s_nm><s_cnt>1 x</s_cnt><sep/><s_price>44,000</s_price><s_nm>Bebek Street</s_nm><s_cnt>1 x</s_cnt></s_menu>



We then apply the chat template to these text tokens along with images, and convert them to model tokens using a processor. While we do not add these special tokens to the tokenizer, it is possible to do so.

.. code-block:: python

        def train_collate_fn(examples, processor):
            processed_examples = []
            for example in examples:
                ground_truth = json.loads(example["ground_truth"])
                if "gt_parses" in ground_truth:  # when multiple ground truths are available, e.g., docvqa
                    assert isinstance(ground_truth["gt_parses"], list)
                    gt_jsons = ground_truth["gt_parses"]
                else:
                    assert "gt_parse" in ground_truth and isinstance(ground_truth["gt_parse"], dict)
                    gt_jsons = [ground_truth["gt_parse"]]


                text = random.choice([json2token(gt_json,sort_json_key=True) for gt_json in gt_jsons])
                processed_examples.append((example["image"], text))

            examples = processed_examples
            images = []
            texts = []

            for example in examples:
                image, ground_truth = example
                images.append(image)

                conversation = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": "Extract JSON"},
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": ground_truth},
                        ],
                    }
                ]
                text_prompt = processor.apply_chat_template(conversation)
                texts.append(text_prompt)

            batch = processor(text=texts, images=images, padding=True, truncation=True,
                            return_tensors="pt")

            labels = batch["input_ids"].clone()
            # Skipped tokens are the padding tokens for both image and text.
            labels[torch.isin(labels, skipped_tokens)] = -100
            batch["labels"] = labels
            return batch


## Train the Model

To train the model, we use the NeMo Fine-tuning API. The full script for training is available in `Nemo VLM Automodel <https://github.com/NVIDIA/NeMo/blob/main/scripts/vlm/automodel.py>`_.

You can directly run the fine-tuning script using the following command:
.. code-block:: bash
    python scripts/vlm/automodel.py --model google/gemma-3-4b-it --data_path naver-clova-ix/cord-v2

At the core of the fine-tuning script is the `llm.finetune` function defined below:

.. code-block:: python

    llm.finetune(
        model=model,
        data=dataset_fn(args.data_path, processor, args.mbs, args.gbs),
        trainer=nl.Trainer(
            devices=args.devices,
            max_steps=args.max_steps,
            accelerator=args.accelerator,
            # save only adapters weights if peft is used
            strategy=make_strategy(args.strategy, model, args.devices, args.num_nodes,
                                   adapter_only=True if peft is not None else False),
            log_every_n_steps=1,
            limit_val_batches=0.0,
            num_sanity_val_steps=0,
            accumulate_grad_batches=1,
            gradient_clip_val=1,
            use_distributed_sampler=False,
            enable_checkpointing=args.disable_ckpt,
            precision='bf16-mixed',
            num_nodes=args.num_nodes,
        ),
        # llm.adam.pytorch_adam_with_flat_lr is returns a fiddle
        #  config, so we need to build the object from the config.
        optim=fdl.build(llm.adam.pytorch_adam_with_flat_lr(lr=1e-5)),
        log=nemo_logger,
        # Peft can be lora or none
        peft=peft,
    )
