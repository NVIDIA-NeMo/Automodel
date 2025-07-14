from typing import Dict, List, Optional, Type
from pathlib import Path
from nemo_automodel.datasets.llm.megatron.gpt_dataset import GPTDataset
from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec
from megatron.core.datasets.utils import get_blend_from_list

class MegatronPretraining:

    def __init__(
        self,
        paths: Path | List | Dict[str, List],
        seq_length: int = 2048,
        tokenizer: Optional["TokenizerSpec"] = None,
        micro_batch_size: int = 4,
        global_batch_size: int = 8,
        rampup_batch_size: Optional[List[int]] = None,
        num_workers: int = 8,
        pin_memory: bool = True,
        persistent_workers: bool = False,
        reset_position_ids: bool = False,
        create_attention_mask: bool = True,
        reset_attention_mask: bool = False,
        eod_mask_loss: bool = False,
        seed: int = 1234,
        split: str = "900,50,50",
        index_mapping_dir: Optional[str] = None,
        num_dataset_builder_threads: int = 1,
        num_train_samples: Optional[int] = None,
        num_val_samples: Optional[int] = None,
        num_test_samples: Optional[int] = None,
        dataset_cls: Type[MegatronDataset] = GPTDataset,
    ) -> None:
        """Pretraining dataset class for Megatron-LM datasets.
        Args:
            paths (Path | List | Dict[str, List]): Paths of the data distributions. Can be either a
                single path, a list of paths, or a dictionary. If a single path or a list of paths,
                the given paths will be used to generate the train, validation and test datasets. If
                providing a list of paths, the format can be either (1) a list of paths, e.g.
                    ["path/to/dataset_1_prefix", "path/to/dataset_2_prefix"],
                or (2) a flattened, zipped list of weights and paths, e.g.
                    ["30", "path/to/dataset_1_prefix", "70", "path/to/dataset_2_prefix"]
                If a dictionary is provided, it is expected to have the following form:
                    {
                        'train': <TRAIN PATHS>,
                        'validation': <VALID PATHS>,
                        'test': <TEST PATHS>
                    }
                where each value is either a path or a list of paths as described above.
                In this case, each split will be generated using the given paths.
                Note that if limit_val_batches <= 1, we generate the entire validaton dataset, so
                weights should not be provided for the validation split.
            seq_length (int): Sequence length.
            tokenizer (Optional["TokenizerSpec"]): An instance of a TokenizerSpec object.
            micro_batch_size (int): Batch size per GPU.
            global_batch_size (int): Global batch size.
            rampup_batch_size (Optional[List[int]]): Rampup batch size, should be in format of
                [start_global_batch_size, batch_size_increment, ramup_samples].
            num_workers (int): See ``torch.utils.data.DataLoader`` documentation.
            pin_memory (bool): See ``torch.utils.data.DataLoader`` documentation.
            persistent_workers (bool): See ``torch.utils.data.DataLoader`` documentation.
            reset_position_ids (bool): Option to reset the position IDs in the dataset at an interval.
                Not supported with fused and flash attention.
            create_attention_mask (bool): Option to enable the attention masks generation.
                Not supported with fused and flash attention.
            reset_attention_mask (bool): Option to reset the attention mask from the dataset.
                Not supported with fused and flash attention.
            eod_mask_loss (int): Option to enable the EOD mask loss.
            seed (int): Seed for generating the GPT dataset.
            split (str): A string of 3 comma-separated integers denoting how much of the distribution
                to allocate to train, validation, and test sets, respectively. Unused if ``paths`` is a dict.
            index_mapping_dir (Optional[str]): Path to a directory to write index mapping files.
            num_dataset_builder_threads (int): The number of threads to use for dataset building.
            num_train_samples (Optional[int]): The number of samples to use for training, defaults to total
                train steps times global batch size.
            num_val_samples (Optional[int]): The number of samples to use for validation, defaults to total
                validation steps times global batch size.
            num_test_samples (Optional[int]): The number of samples to use for testing, defaults to total
                test steps times global batch size.
            dataset_cls (Optional[Type[MegatronDataset]]): The dataset class to use for the data module.
        """
        self.dataset_cls = dataset_cls
        build_kwargs = {}
        paths, weights = get_blend_from_list(paths)
        if len(paths) == 1:
            weights = None
        build_kwargs["blend"] = [paths, weights]
        build_kwargs["split"] = split

        self.build_kwargs = build_kwargs
        self.seq_length = seq_length
        self.micro_batch_size = micro_batch_size
        self.global_batch_size = global_batch_size
        self.tokenizer = tokenizer
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.reset_position_ids = reset_position_ids
        self.create_attention_mask = create_attention_mask
        self.reset_attention_mask = reset_attention_mask
        self.eod_mask_loss = eod_mask_loss
        self.seed = seed
        self.split = split
        self.index_mapping_dir = index_mapping_dir
        self.num_dataset_builder_threads = num_dataset_builder_threads
        self.init_global_step = 0
        self.num_train_samples = num_train_samples
        self.num_val_samples = num_val_samples
        self.num_test_samples = num_test_samples

        # self.tokenizer = tokenizer or get_nmt_tokenizer("megatron", "GPT2BPETokenizer")
        self.data_sampler = MegatronDataSampler(
            seq_len=self.seq_length,
            micro_batch_size=self.micro_batch_size,
            global_batch_size=self.global_batch_size,
            rampup_batch_size=rampup_batch_size,
        )