# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import glob
import logging
import os
from importlib.util import find_spec
from pathlib import Path
from typing import Dict, List, Optional, Union

from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from nemo_automodel.components.datasets.llm.megatron.builder import BlendedMegatronDatasetBuilder
from nemo_automodel.components.datasets.llm.megatron.gpt_dataset import GPTDatasetConfig
from nemo_automodel.components.datasets.llm.megatron.megatron_utils import compile_helper, get_blend_from_list

logger = logging.getLogger(__name__)


class MegatronPretraining:
    def __init__(
        self,
        paths: Path | List | Dict[str, List],
        seq_length: int = 2048,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        micro_batch_size: int = 4,
        global_batch_size: int = 8,
        create_attention_mask: bool = False,
        seed: int = 1234,
        split: str = "900,50,50",
        index_mapping_dir: Optional[str] = None,
        num_dataset_builder_threads: int = 1,
        num_train_samples: Optional[int] = None,
        num_val_samples: Optional[int] = None,
        num_test_samples: Optional[int] = None,
        trainer_max_steps: Optional[int] = None,
        trainer_val_check_interval: int = 1000,
        trainer_limit_val_batches: Union[int, float] = 1,
        trainer_limit_test_batches: Union[int, float] = 1,
        mmap_bin_files: bool = True,
        splits_to_build: Optional[Union[str, List[str]]] = None,
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
            tokenizer (Optional[PreTrainedTokenizerBase]): An instance of a PreTrainedTokenizerBase object.
            micro_batch_size (int): Batch size per GPU.
            global_batch_size (int): Global batch size.
            create_attention_mask (bool): Option to enable the attention masks generation.
                Not supported with fused and flash attention.
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
            trainer_max_steps (Optional[int]): Maximum training steps. If None or -1, uses full dataset for one epoch.
            trainer_val_check_interval (int): Interval for validation checks.
            trainer_limit_val_batches (Union[int, float]): Limit for validation batches.
            trainer_limit_test_batches (Union[int, float]): Limit for test batches.
            splits_to_build (Optional[Union[str, List[str]]]): Splits to build. If None, builds all splits.
        """
        if find_spec("nemo_automodel.components.datasets.llm.megatron.helpers_cpp") is None:
            try:
                compile_helper()
                assert find_spec("nemo_automodel.components.datasets.llm.megatron.helpers_cpp") is not None
            except AssertionError:
                raise ImportError(
                    "Could not compile megatron dataset C++ helper functions and therefore cannot import helpers python file."
                )

        if not isinstance(paths, (list, tuple, dict)):
            paths = get_list_of_files(paths)
        validate_dataset_asset_accessibility(paths)

        if isinstance(split, (list, tuple)):
            split = [str(s) for s in split]
            split = ", ".join(split)

        build_kwargs = {}
        build_kwargs["mmap_bin_files"] = mmap_bin_files
        if isinstance(paths, dict):
            if split is not None:
                logger.warning(
                    f"{split=} will be ignored since datasets are being created from 3 separate distributions."
                )
            build_kwargs["blend_per_split"] = [
                get_blend_from_list(paths["train"]),
                get_blend_from_list(paths["validation"]),
                get_blend_from_list(paths["test"]),
            ]
        else:
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
        self.create_attention_mask = create_attention_mask
        self.seed = seed
        self.split = split
        self.index_mapping_dir = index_mapping_dir
        self.num_dataset_builder_threads = num_dataset_builder_threads
        self.num_train_samples = num_train_samples
        self.num_val_samples = num_val_samples
        self.num_test_samples = num_test_samples
        if isinstance(splits_to_build, str):
            assert splits_to_build in ["train", "validation", "test"], f"Invalid split: {splits_to_build}"
        elif isinstance(splits_to_build, list):
            assert all(s in ["train", "validation", "test"] for s in splits_to_build), (
                f"Invalid splits: {splits_to_build}"
            )
        self.splits_to_build = splits_to_build

        # Store trainer arguments
        self.trainer_max_steps = trainer_max_steps
        self.trainer_val_check_interval = trainer_val_check_interval
        self.trainer_limit_val_batches = trainer_limit_val_batches
        self.trainer_limit_test_batches = trainer_limit_test_batches

    def build(self):
        """
        Build the datasets using the trainer parameters provided during initialization.
        """
        train_iters = self.trainer_max_steps
        if train_iters is None or train_iters == -1:
            # Full-epoch training: build exhaustive indices
            num_train_samples = None
        else:
            assert train_iters > 0, f"max_steps {train_iters} should be greater than 0"
            num_train_samples = int(train_iters * self.global_batch_size)

        if self.num_train_samples is not None:
            if num_train_samples is not None:
                assert self.num_train_samples >= num_train_samples, (
                    f"num_train_samples must be greater than or equal to {num_train_samples}."
                )
            num_train_samples = self.num_train_samples
            train_iters = int(num_train_samples / self.global_batch_size)

        if self.num_val_samples is not None:
            num_val_samples = self.num_val_samples
        elif train_iters is None or train_iters == -1:
            num_val_samples = None
        else:
            num_val_samples = (
                int(train_iters // self.trainer_val_check_interval)
                * self.trainer_limit_val_batches
                * self.global_batch_size
            )

        if self.num_test_samples is not None:
            num_test_samples = self.num_test_samples
        else:
            num_test_samples = None

        if (
            self.trainer_limit_val_batches > 0.0
            and self.trainer_limit_val_batches <= 1.0
            and isinstance(self.trainer_limit_val_batches, float)
        ):
            assert "blend" not in self.build_kwargs, (
                "When using a single data distribution, limit_val_batches <= 1.0 is not supported. If you'd "
                "like to run with a fractional value of limit_val_batches, please pass in separate datasets for "
                "the train, validation, and test datasets by providing a dictionary of paths, e.g.: \n"
                "    paths={ \n "
                "        'train': [PATHS FOR TRAIN], \n "
                "        'validation': [PATHS FOR VALIDATION], \n "
                "        'test' :[PATHS FOR TEST],  \n"
                "    }"
            )

            # This is to make sure we only have one epoch on every validation iteration
            num_val_samples = None

        train_valid_test_num_samples = [num_train_samples, num_val_samples, num_test_samples]
        self._train_ds, self._validation_ds, self._test_ds = BlendedMegatronDatasetBuilder(
            train_valid_test_num_samples,
            is_built_on_rank=lambda: True,
            config=self.gpt_dataset_config,
            enabled_splits=self.splits_to_build,
        ).build()

    def get_dataset(self, split: str):
        """
        Get the dataset for a given split.
        """
        mapping = {"train": "_train_ds", "validation": "_validation_ds", "test": "_test_ds"}
        assert split in ["train", "validation", "test"], f"Invalid split: {split}"
        if not hasattr(self, mapping[split]) or getattr(self, mapping[split]) is None:
            raise RuntimeError(
                f"Dataset for split {split} was not built. Include '{split}' in splits_to_build to enable it."
            )
        return getattr(self, mapping[split])

    @property
    def gpt_dataset_config(self) -> "GPTDatasetConfig":
        """
        Get the GPT dataset configuration.
        """

        return GPTDatasetConfig(
            random_seed=self.seed,
            sequence_length=self.seq_length,
            tokenizer=self.tokenizer,
            path_to_cache=self.index_mapping_dir,
            reset_position_ids=False,
            create_attention_mask=self.create_attention_mask,
            reset_attention_mask=False,
            eod_mask_loss=False,
            num_dataset_builder_threads=self.num_dataset_builder_threads,
            **self.build_kwargs,
        )


def is_number_tryexcept(s):
    """Returns True if string is a number."""
    if s is None:
        return False
    try:
        float(s)
        return True
    except ValueError:
        return False


def is_zipped_list(paths):
    """
    Check if the paths are zipped.
    """
    # ["30", "path/to/dataset_1_prefix", "70", "path/to/dataset_2_prefix"]
    even = paths[::2]
    if len(even) == 0:
        return False
    is_num = list(map(is_number_tryexcept, even))
    if any(is_num):
        assert all(is_num), "Got malformatted zipped list"
    return is_num[0]


def validate_dataset_asset_accessibility(paths):
    """
    Validate the accessibility of the dataset assets.
    """
    if paths is None:
        raise ValueError("Expected path to have a value.")

    if isinstance(paths, tuple) or isinstance(paths, list):
        if is_zipped_list(paths):
            # remove weights from paths.
            paths = paths[1::2]
        for p in paths:
            validate_dataset_asset_accessibility(p)
        return
    elif isinstance(paths, dict):
        for p in paths.values():
            validate_dataset_asset_accessibility(p)
        return

    if not isinstance(paths, str) and not isinstance(paths, Path):
        raise ValueError("Expected path to be of string or Path type.")

    path = Path(paths)

    suffices = (".bin", ".idx")
    if path.is_dir():
        if not os.access(path, os.R_OK):
            raise PermissionError(f"Expected {str(path)} to be readable.")
        # Will let the downstream class confirm contents are ok.
        return
    if path.exists():
        if not os.access(path, os.R_OK):
            raise PermissionError(f"Expected {str(path)} to be readable.")
        return
    for suffix in suffices:
        file_path = path.with_name(path.name + suffix)
        if not file_path.exists():
            raise FileNotFoundError(f"Expected {str(file_path)} to exist.")
        if not os.access(file_path, os.R_OK):
            raise PermissionError(f"Expected {str(file_path)} to be readable.")


def get_list_of_files(path: str):
    """
    Get the list of unique dataset prefixes (full paths without extension) from a glob pattern.
    """
    if not glob.has_magic(path):
        return [path]
    paths = glob.glob(path)
    unique_paths = set()
    for path in paths:
        assert path.endswith(".bin") or path.endswith(".idx"), f"Expected {path} to be a .bin or .idx file."
        unique_paths.add(str(Path(path).with_suffix("")))
    return sorted(list(unique_paths))
