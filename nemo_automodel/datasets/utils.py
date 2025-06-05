import torch
import math

def batchify(tensor):
    """Ensures that the input tensor has at least two dimensions by adding an extra batch dimension if necessary.

    Parameters
    ----------
    tensor : torch.Tensor
        The input tensor to be batchified.

    Returns
    -------
    torch.Tensor
        The tensor with an extra dimension added if it was originally 1-dimensional.
        Otherwise, the tensor is returned as-is.
    """
    if tensor.ndim == 1:
        return tensor.unsqueeze_(0)
    return tensor


def extract_key_from_dicts(batch, key):
    """Extracts the value of the given key from each dictionary in a list of dictionaries.

    Parameters
    ----------
    batch : List[dict]
        A list of dictionaries.
    key : str
        The key whose values are to be extracted from each dictionary.

    Returns
    -------
    List
        A list of values associated with the specified key, in the same order as
        the dictionaries in the input batch.
    """
    return list(map(lambda x: x[key], batch))


def pad_within_micro(batch, pad_token_id, pad_seq_len_divisible=None):
    """Pads each list in a batch of lists to the same length with a specified token.

    Parameters
    ----------
    batch : List[List[int]]
        A batch of sequences (e.g., token IDs), where each sequence is a list of integers.
    pad_token_id : int
        The token ID to use for padding shorter sequences.
    pad_seq_len_divisible : int
        The value to use for padding sequence length so that it is divisible by pad_seq_len_divisible.
    Returns
    -------
    List[List[int]]
        A batch of sequences where each inner list has been padded with the pad token
        to match the length of the longest sequence in the batch.
    """
    max_len = max(map(len, batch))
    if pad_seq_len_divisible:
        max_len = (pad_seq_len_divisible - max_len % pad_seq_len_divisible) + max_len
    return [item + [pad_token_id] * (max_len - len(item)) for item in batch]


def default_collater(batch, pad_token_id=0, pad_seq_len_divisible=None):
    """Default batch collator that handles padding and batching.

    Args:
        batch: A batch of examples.
        pad_token_id: The token ID to use for padding.
        pad_seq_len_divisible: If provided, pad sequence length to be divisible by this value.

    Returns:
        dict: A dictionary containing batched tensors.
    """
    return {
        key: batchify(
            torch.LongTensor(
                pad_within_micro(
                    extract_key_from_dicts(batch, key),
                    (
                        0 if key == "attention_mask" or key == "loss_mask" else (
                            -100 if key == "labels" else pad_token_id
                        )
                    ),
                    pad_seq_len_divisible,
                )
            )
        )
        for key in batch[0].keys()
    }


class SFTSingleTurnPreprocessor:
    """
    Generic single-turn text-to-text SFT (supervised-fine-tuning) pre-processor.

    Parameters
    ----------
    args           : argparse.Namespace or similar - must expose the fields
                     `dataset_name`, `model_name_or_path`, `preprocessing_num_workers`,
                     `overwrite_cache`.
    tokenizer      : Pre-trained tokenizer (HF).
    accelerator    : accelerate.Accelerator.
    task_dict      : Dict[str, Task] mapping dataset_name -> task object that
                     provides `get_context()` and `get_target()` callables.
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.block_size = None
        self.preprocessing_num_workers = 1
        self.overwrite_cache = False

    def _tokenize_function(self, examples, dataset):
        ctx = dataset.get_context(examples)
        tgt = dataset.get_target(examples)

        ctx_tok = self.tokenizer(ctx)
        tgt_tok = self.tokenizer(tgt)

        # strip trailing special token from context
        if len(ctx_tok["input_ids"][0]) > 0 and \
           ctx_tok["input_ids"][0][-1] in self.tokenizer.all_special_ids:
            ctx_tok["input_ids"] = [ids[:-1] for ids in ctx_tok["input_ids"]]
            ctx_tok["attention_mask"] = [m[:-1] for m in ctx_tok["attention_mask"]]

        # strip leading special token from target
        if len(tgt_tok["input_ids"][0]) > 0 and \
           tgt_tok["input_ids"][0][0] in self.tokenizer.all_special_ids:
            tgt_tok["input_ids"] = [ids[1:] for ids in tgt_tok["input_ids"]]
            tgt_tok["attention_mask"] = [m[1:] for m in tgt_tok["attention_mask"]]

        out = {}
        out["input_ids"] = [
            c_ids + t_ids for c_ids, t_ids in zip(ctx_tok["input_ids"],
                                                  tgt_tok["input_ids"])
        ]
        out["attention_mask"] = [
            c_m + t_m for c_m, t_m in zip(ctx_tok["attention_mask"],
                                          tgt_tok["attention_mask"])
        ]
        # label: -100 for ctx, true ids for tgt
        out["labels"] = [
            [-100] * (len(c_ids)-1) + t_ids + [-100]
            for c_ids, t_ids in zip(ctx_tok["input_ids"], tgt_tok["input_ids"])
        ]

        out["loss_mask"] = [
            [1 if t != -100 else 0 for t in lbl] for lbl in out["labels"]
        ]
        return out

    def _compute_dataset_max_len(self, tokenized_ds):
        max_len = max(map(lambda x: len(x['input_ids']), tokenized_ds))
        # make multiple of 8
        max_len = math.ceil(max_len / 8) * 8
        # respect model block size
        if self.block_size is not None:
            max_len = min(max_len, self.block_size)
        return max_len

    def _pad_function(self, max_len):
        tk = self.tokenizer

        def _pad(examples):
            pad_id = tk.pad_token_id or 0
            examples["input_ids"] = [
                (ids[:max_len] + [pad_id] * max(0, max_len - len(ids)))
                for ids in examples["input_ids"]
            ]
            examples["attention_mask"] = [
                ([1] * min(len(ids), max_len) + [0] * max(0, max_len - len(ids)))
                for ids in examples["attention_mask"]
            ]
            examples["labels"] = [
                (lbl[:max_len] + [-100] * max(0, max_len - len(lbl)))
                for lbl in examples["labels"]
            ]
            examples["loss_mask"] = [
                (lm[:max_len] + [0] * max(0, max_len - len(lm)))
                for lm in examples["loss_mask"]
            ]
            # return dictionary with sequences all exactly `max_len` long
            return examples

        return _pad

    def process(self, raw_dataset, ds):
        """
        Main entry.

        Parameters
        ----------
        raw_dataset : datasets.DatasetDict  (e.g. returned by load_dataset)
        split        : Which split from raw_dataset to process.

        Returns
        -------
        datasets.DatasetDict  - tokenized + padded datasets (all splits preserved).
        """

        if not hasattr(self.tokenizer, 'pad_token') and hasattr(self.tokenizer, 'bos_token'):
            self.tokenizer.pad_token = self.tokenizer.bos_token

        # 1. tokenise ----------------------------------------------------------------
        tokenized = raw_dataset.map(
            lambda x: self._tokenize_function(x, dataset=ds),
            batched=True,
            num_proc=self.preprocessing_num_workers,
            remove_columns=raw_dataset.column_names,
            load_from_cache_file=False, #not self.overwrite_cache,
            desc="Running tokenizer on dataset",
        )

        # 2. global max len -----------------------------------------------------------
        max_len = self._compute_dataset_max_len(tokenized)

        # 3. pad ----------------------------------------------------------------------
        pad_fn = self._pad_function(max_len)
        tokenized = tokenized.map(
            pad_fn,
            batched=True,
            num_proc=self.preprocessing_num_workers,
            load_from_cache_file=not self.overwrite_cache,
            desc=f"Padding dataset to max length {max_len}",
        )

        return tokenized

