from nemo.collections.llm.gpt.data.pre_training import PreTrainingDataModule
# from nemo_automodel.components.datasets.llm.megatron_dataset import MegatronPretraining
from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer
from transformers import AutoTokenizer

# Your dataset paths (from test.py)
document_paths = [
    f"fineweb_edu/megatron"
]

# Optional: Use a specific tokenizer (e.g., SentencePiece or Hugging Face); defaults to GPT2BPE if None
# tokenizer = get_nmt_tokenizer("megatron", "GPT2BPETokenizer")  # Or your custom tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")

# Instantiate the DataModule
data_module = PreTrainingDataModule(
    paths=document_paths,  # List of 50 shard prefixes
    seq_length=2048,  # Sequence length (adjust to your needs, e.g., 8192)
    micro_batch_size=4,  # Per-GPU batch size
    global_batch_size=16,  # Total batch size (scale based on your GPUs)
    num_workers=0,  # Data loading workers
    split="0.99, 0.01, 0.01",  # Train/val/test split (90/5/5); or use a dict for custom splits
    seed=1234,  # For reproducibility
    tokenizer=tokenizer,
    trainer_max_steps=1000,
    trainer_val_check_interval=1000,
    trainer_limit_val_batches=1,
    trainer_limit_test_batches=1,
    # Add other params as needed (e.g., reset_position_ids=False)
)
# Build datasets directly using the datamodule's build() method -------------
data_module.build(
    # trainer_max_steps=1000,  # total train steps you plan (any positive int for a test)
    # trainer_val_check_interval=1000,  # how often validation would run
    # trainer_limit_val_batches=1,  # number (or fraction) of val batches
    # trainer_limit_test_batches=1,  # number (or fraction) of test batches
)

# Now dataloaders are ready -------------------------------------------------
# Inject a dummy trainer so PreTrainingDataModule.train_dataloader works outside Lightning
from types import SimpleNamespace

# The datamodule expects a `.trainer` object with a `global_step` attribute.
# We spoof the minimum it needs so we can iterate over the dataloader in a
# stand-alone script.
data_module.trainer = SimpleNamespace(global_step=0)

train_loader = data_module.train_dataloader()

# ---------------------------------------------------------------------
# Iterate over a couple of batches – PreTrainingDataModule returns each
# batch as a *dict* with keys like "tokens"/"labels"/"loss_mask" rather
# than a tuple, so use the keys instead of numeric indices.
# ---------------------------------------------------------------------
breakpoint()
for i, batch in enumerate(train_loader):
    # Keys are modelled after NeMo-Megatron’s default collate_fn
    tokens = batch.get("input_ids")
    labels = batch.get("labels")

    print(
        f"Batch {i}: tokens: {tuple(tokens.shape) if tokens is not None else 'N/A'}, "
        f"labels: {tuple(labels.shape) if labels is not None else 'N/A'}"
        f"input_ids: {tokens}"
    )

    if i >= 2:
        break