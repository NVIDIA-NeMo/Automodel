# This directory has been deprecated. Please delete it.
# All classes have been moved to their canonical locations:
# - BiencoderModel, BiencoderOutput, pool, contrastive_scores_and_labels -> nemo_automodel._transformers.biencoder
# - NeMoAutoModelForBiencoder -> nemo_automodel._transformers.auto_model
# - LlamaBidirectionalModel, LlamaBidirectionalConfig -> nemo_automodel.components.models.llama_bidirectional.model
# - BiencoderStateDictAdapter -> nemo_automodel.components.models.common.bidirectional
raise ImportError(
    "This module has been removed. Please import from the canonical locations:\n"
    "  - from nemo_automodel._transformers.biencoder import BiencoderModel\n"
    "  - from nemo_automodel._transformers.auto_model import NeMoAutoModelForBiencoder\n"
    "  - from nemo_automodel.components.models.llama_bidirectional import LlamaBidirectionalModel"
)
