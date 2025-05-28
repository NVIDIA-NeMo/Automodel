from datasets import load_dataset
from automodel.datasets.utils import SFTSingleTurnPreprocessor

class HellaSwag:
    """
    A dataset wrapper for the HellaSwag benchmark, tailored for single-turn supervised fine-tuning (SFT).

    This class loads and preprocesses the HellaSwag dataset using a tokenizer and a custom preprocessing
    pipeline for language model fine-tuning. The dataset consists of context and multiple-choice endings,
    where the goal is to choose the most plausible continuation.

    Attributes:
        dataset (Dataset): The processed dataset ready for model training or evaluation.
    """
    def __init__(self, path_or_dataset, tokenizer, split='train', num_samples_limit=None, trust_remote_code=True):
        """
        Initialize the HellaSwag dataset wrapper.

        Args:
            path_or_dataset (str or Dataset): Path to the dataset or a HuggingFace Dataset object.
            tokenizer (PreTrainedTokenizer): The tokenizer used to process text.
            split (str, optional): Dataset split to use (e.g., 'train', 'validation'). Defaults to 'train'.
            num_samples_limit (int, optional): Maximum number of samples to load. Defaults to None.
            trust_remote_code (bool, optional): Whether to trust remote code. Defaults to True.

        Notes:
            If num_samples_limit is an integer, it limits the dataset size using slicing.
        """
        if isinstance(num_samples_limit, int):
            split = f'{split}[:{num_samples_limit}]'
        raw_datasets = load_dataset(path_or_dataset, split=split, trust_remote_code=trust_remote_code)
        processor = SFTSingleTurnPreprocessor(tokenizer)
        self.dataset = processor.process(raw_datasets, self)


    def get_context(self, examples):
        """
        Extracts the context part of each example.

        Args:
            examples (dict): A dictionary containing example data with a "ctx" key.

        Returns:
            list[str]: List of context strings.
        """
        return examples["ctx"]

    def get_target(self, examples):
        """
        Extracts the correct ending based on the label.

        Args:
            examples (dict): A dictionary with "endings" (list of strings) and "label" (index of correct ending).

        Returns:
            list[str]: The gold target strings based on the label index.
        """
        return [
            endings[int(lbl)] for endings, lbl in zip(examples["endings"], examples["label"])
        ]

    def __getitem__(self, index):
        """
        Get a processed example by index.

        Args:
            index (int): Index of the example.

        Returns:
            dict: A tokenized and preprocessed example.
        """
        return self.dataset[index]

    def __len__(self):
        """
        Get the number of examples in the dataset.

        Returns:
            int: Length of the processed dataset.
        """
        return len(self.dataset)