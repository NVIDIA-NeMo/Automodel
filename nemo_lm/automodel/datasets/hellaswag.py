import sys
import re
from functools import partial

import datasets
from datasets import load_dataset
from nemo_lm.automodel.datasets.utils import SFTSingleTurnPreprocessor

class HellaSwag:
    def __init__(self, path_or_dataset, tokenizer, split):
        print(path_or_dataset)
        raw_datasets = load_dataset(path_or_dataset, split='train[:10]')
        processor = SFTSingleTurnPreprocessor(tokenizer)
        self.dataset = processor.process(raw_datasets, self)


    def get_context(self, examples):
        return examples["ctx"]

    def get_target(self, examples):
        # Pick the correct ending using the gold label
        return [
            endings[int(lbl)] for endings, lbl in zip(examples["endings"], examples["label"])
        ]
    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)