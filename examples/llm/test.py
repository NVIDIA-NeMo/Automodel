# import torch
# from transformers import pipeline

# model_id = "checkpoints/epoch_0_step_20/model/consolidated/"
# pipe = pipeline(
#     "text-generation",
#     model=model_id,
#     torch_dtype=torch.bfloat16,
#     device_map="auto",
# )

# print(pipe("The key to life is"))

from vllm import LLM, SamplingParams

llm = LLM(model="checkpoints/epoch_0_step_10/model/consolidated/", model_impl="transformers")
params = SamplingParams(max_tokens=20)
outputs = llm.generate("Toronto is a city in Canada.", sampling_params=params)
print(f"Generated text: {outputs[0].outputs[0].text}")
