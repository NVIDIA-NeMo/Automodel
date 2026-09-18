# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Check whether training-dataset queries are present in a larger training JSON."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)

DatasetSpec = tuple[int | None, Path]
SummaryRow = dict[str, int | str]

FSW_YBABAKHIN = Path("/lustre/fsw/portfolios/llmservice/users/ybabakhin")
FS12_ROSMULSKI = Path("/lustre/fs12/portfolios/llmservice/projects/llmservice_nlp_retro/users/rosmulski")
ROSMULSKI_SCRATCH = Path("/home/rosmulski/scratch")
RONAYA_TRAINING = Path("/lustre/fsw/portfolios/llmservice/users/ronaya/mteb_8b/training_datasets")

RTEB_ROOT = FSW_YBABAKHIN / "data/rteb"
NV_EMBED_ROOT = FSW_YBABAKHIN / "data/nv-embed-v2-non-retrieval"
NV_EMBED_TRAINING = NV_EMBED_ROOT / "training_datasets"
TEXTQA_TRAIN = FSW_YBABAKHIN / "data/TextQA_train"
MIRACL_FILTERED = FS12_ROSMULSKI / "filtering_repo/datasets/miracl_filtered"
ROSMULSKI_MINED_QWEN = ROSMULSKI_SCRATCH / "filtering_repo/mined_hardnegs/instr_qwen3_8B"
ROSMULSKI_MINED_LLAMA = ROSMULSKI_SCRATCH / "filtering_repo/mined_hardnegs/instr_llama32_8B"


def _path_with_name(root: Path, *name_parts: str) -> Path:
    """Build a path while allowing long filenames to be split across lines."""
    return root / "".join(name_parts)


REFERENCE_DATASET = _path_with_name(
    RTEB_ROOT,
    "multilingual_wiki_ccnews_5hn_withMoreDS_more_reddit_s2orc_allLang_11M_",
    "qwen3_8B_0_95_perc_posdocs_only.json",
)


TRAINING_DATASETS: list[DatasetSpec] = [
    (5999, MIRACL_FILTERED / "miracl_ar_unrolled_8B_ragas2_scores_1_2.json"),
    (3659, MIRACL_FILTERED / "miracl_bn_unrolled_8B_ragas2_scores_1_2.json"),
    (6971, MIRACL_FILTERED / "miracl_en_unrolled_8B_ragas2_scores_1_2.json"),
    (8432, MIRACL_FILTERED / "miracl_es_unrolled_8B_ragas2_scores_1_2.json"),
    (4007, MIRACL_FILTERED / "miracl_fa_unrolled_8B_ragas2_scores_1_2.json"),
    (4767, MIRACL_FILTERED / "miracl_fi_unrolled_8B_ragas2_scores_1_2.json"),
    (2157, MIRACL_FILTERED / "miracl_fr_unrolled_8B_ragas2_scores_1_2.json"),
    (2209, MIRACL_FILTERED / "miracl_hi_unrolled_8B_ragas2_scores_1_2.json"),
    (9365, MIRACL_FILTERED / "miracl_id_unrolled_8B_ragas2_scores_1_2.json"),
    (6676, MIRACL_FILTERED / "miracl_ja_unrolled_8B_ragas2_scores_1_2.json"),
    (1908, MIRACL_FILTERED / "miracl_ko_unrolled_8B_ragas2_scores_1_2.json"),
    (9535, MIRACL_FILTERED / "miracl_ru_unrolled_8B_ragas2_scores_1_2.json"),
    (2378, MIRACL_FILTERED / "miracl_sw_unrolled_8B_ragas2_scores_1_2.json"),
    (3915, MIRACL_FILTERED / "miracl_te_unrolled_8B_ragas2_scores_1_2.json"),
    (4611, MIRACL_FILTERED / "miracl_th_unrolled_8B_ragas2_scores_1_2.json"),
    (3059, MIRACL_FILTERED / "miracl_zh_unrolled_8B_ragas2_scores_1_2.json"),
    (9500, TEXTQA_TRAIN / "mldr_en_perc95_small_train.json"),
    (60000, NV_EMBED_TRAINING / "amazon_reviews_classification.json"),
    (13046, NV_EMBED_TRAINING / "emotion_classification_safe.json"),
    (40000, NV_EMBED_TRAINING / "reddit_clustering_s2s.json"),
    (50000, NV_EMBED_TRAINING / "reddit_clustering_p2p.json"),
    (2495, NV_EMBED_TRAINING / "bioasq.json"),
    (140085, NV_EMBED_TRAINING / "fever.json"),
    (100000, NV_EMBED_TRAINING / "gooaq.json"),
    (170000, NV_EMBED_TRAINING / "hotpotqa.json"),
    (500000, NV_EMBED_TRAINING / "msmarco.json"),
    (3685, NV_EMBED_TRAINING / "nfcorpus.json"),
    (100231, NV_EMBED_TRAINING / "nq.json"),
    (500000, NV_EMBED_TRAINING / "paq.json"),
    (919, NV_EMBED_TRAINING / "scifacts.json"),
    (87599, NV_EMBED_TRAINING / "squad.json"),
    (73346, NV_EMBED_TRAINING / "triviaqa.json"),
    (80001, NV_EMBED_TRAINING / "stackexchange.json"),
    (29721, NV_EMBED_TRAINING / "hover.json"),
    (3547, NV_EMBED_TRAINING / "mrtidy_english.json"),
    (3697, NV_EMBED_TRAINING / "mrtidy_japanese.json"),
    (5366, NV_EMBED_TRAINING / "mrtidy_russian.json"),
    (48494, NV_EMBED_TRAINING / "combined_SynthInstrLongShort.json"),
    (50001, NV_EMBED_TRAINING / "combined_SynthInstrShortLong.json"),
    (39753, NV_EMBED_TRAINING / "combined_SynthInstrShortShort.json"),
    (20000, NV_EMBED_TRAINING / "ToxicConversationsClassification-kaggle.json"),
    (34990, NV_EMBED_TRAINING / "AmazonCounterfactualClassification_gpt-oss-35k-label.json"),
    (49590, NV_EMBED_TRAINING / "LongShort-orig-50k.json"),
    (98922, NV_EMBED_TRAINING / "LongShort-orig-100k.json"),
    (50691, NV_EMBED_TRAINING / "LongShort-orig-gpt-oss-50k.json"),
    (1000, ROSMULSKI_SCRATCH / "datasets/mteb_new_datasets/spartqa.json"),
    (47576, NV_EMBED_TRAINING / "LongShort-orig-llama-3-50k.json"),
    (48238, NV_EMBED_TRAINING / "LongShort-orig-llama-4-50k.json"),
    (44577, NV_EMBED_TRAINING / "LongShort-e5-sample-topics-50k.json"),
    (197265, NV_EMBED_TRAINING / "LongShort-orig-200k.json"),
    (48479, NV_EMBED_TRAINING / "LongShort-orig-qwen-3-50k.json"),
    (47621, NV_EMBED_TRAINING / "LongShort-orig-llama-3-extra-50k.json"),
    (48268, NV_EMBED_TRAINING / "LongShort-orig-llama-4-extra-50k.json"),
    (50682, NV_EMBED_TRAINING / "LongShort-orig-gpt-oss-extra-50k.json"),
    (46589, NV_EMBED_TRAINING / "LongShort-e5-sample-topics-extra-50k.json"),
    (48338, NV_EMBED_TRAINING / "LongShort-orig-llama-4-maverick-50k.json"),
    (
        94354,
        _path_with_name(
            NV_EMBED_TRAINING,
            "LongShort-e5-sample-topics-llama-4-maverick-100k.json",
        ),
    ),
    (
        200000,
        _path_with_name(
            ROSMULSKI_MINED_LLAMA,
            "msmarco_instruct_query_and_inst_with_neg_query_hard_negs_as_pos_single_neg_",
            "llama_32_8B_0.95_perc.json",
        ),
    ),
    (
        50000,
        _path_with_name(
            NV_EMBED_TRAINING,
            "STS-seed-passages-100k-random_qwen3_8B_0_95_perc_no_negs.json",
        ),
    ),
    (49997, NV_EMBED_TRAINING / "STS-news-50k_qwen3_8B_0_95_perc_no_negs.json"),
    (257963, NV_EMBED_TRAINING / "AllNLI-250k_qwen3_8B_0_95_perc_no_negs.json"),
    (2760, ROSMULSKI_MINED_QWEN / "arguana_synth_v2_2760_qwen3_8B_0.95_perc.json"),
    (9660, ROSMULSKI_MINED_QWEN / "arguana_synth_v1_9660_qwen3_8B_0.95_perc.json"),
    (9600, ROSMULSKI_MINED_QWEN / "arguana_synth_gpt_oss_20B_9660_qwen3_8B_0.95_perc.json"),
    (11040, ROSMULSKI_MINED_QWEN / "arguana_synth_mixtral_10k_qwen3_8B_0.95_perc.json"),
    (60000, ROSMULSKI_MINED_QWEN / "arguana_synth_gpt_oss_120B_100k_qwen3_8B_0.95_perc.json"),
    (317180, RONAYA_TRAINING / "tiger_stack_300k_mined_no_scores.json"),
    (3094, RONAYA_TRAINING / "textual_alignment_label.json"),
    (25000, ROSMULSKI_MINED_QWEN / "scidocs_1M_qwen3_8B_0.95_perc.json"),
    (29479, NV_EMBED_TRAINING / "dalaj_classification_filtered.json"),
    (10000, NV_EMBED_TRAINING / "STS13-maverick-mixtral-40k_qwen3_8B_0_95_perc_no_negs.json"),
    (10000, NV_EMBED_TRAINING / "siswati_classification_llama_maverick_15k.json"),
    (19538, NV_EMBED_TRAINING / "massive_intent_classification_llama_maverick_15k.json"),
    (101762, NV_EMBED_TRAINING / "Quora-100k_qwen3_8B_0_95_perc_no_negs.json"),
    (10000, NV_EMBED_TRAINING / "itacasehold_classification_llama_maverick_15k.json"),
    (20000, RONAYA_TRAINING / "toxic_tweet_classification.json"),
    (
        8983,
        _path_with_name(
            NV_EMBED_ROOT,
            "rotten_tomatoes_parallel_en_Balinese_qwen3_8B_0.95_perc_",
            "qwen3_8B_0.95_perc.json",
        ),
    ),
    (
        10054,
        _path_with_name(
            NV_EMBED_ROOT,
            "rotten_tomatoes_parallel_en_Danish_qwen3_8B_0.95_perc_",
            "qwen3_8B_0.95_perc.json",
        ),
    ),
    (
        9911,
        _path_with_name(
            NV_EMBED_ROOT,
            "rotten_tomatoes_parallel_en_French_qwen3_8B_0.95_perc_",
            "qwen3_8B_0.95_perc.json",
        ),
    ),
    (
        8108,
        _path_with_name(
            NV_EMBED_ROOT,
            "rotten_tomatoes_parallel_en_Husa_qwen3_8B_0.95_perc_",
            "qwen3_8B_0.95_perc.json",
        ),
    ),
    (
        7896,
        _path_with_name(
            NV_EMBED_ROOT,
            "rotten_tomatoes_parallel_en_Igbo_qwen3_8B_0.95_perc_",
            "qwen3_8B_0.95_perc.json",
        ),
    ),
    (
        7612,
        _path_with_name(
            NV_EMBED_ROOT,
            "rotten_tomatoes_parallel_en_Indonesian_qwen3_8B_0.95_perc_",
            "qwen3_8B_0.95_perc.json",
        ),
    ),
    (
        8623,
        _path_with_name(
            NV_EMBED_ROOT,
            "rotten_tomatoes_parallel_en_Javanese_qwen3_8B_0.95_perc_",
            "qwen3_8B_0.95_perc.json",
        ),
    ),
    (
        9392,
        _path_with_name(
            NV_EMBED_ROOT,
            "rotten_tomatoes_parallel_en_Nigerian-Pidgin_qwen3_8B_0.95_perc_",
            "qwen3_8B_0.95_perc.json",
        ),
    ),
    (
        8281,
        _path_with_name(
            NV_EMBED_ROOT,
            "rotten_tomatoes_parallel_en_Sundanese_qwen3_8B_0.95_perc_",
            "qwen3_8B_0.95_perc.json",
        ),
    ),
    (
        8250,
        _path_with_name(
            NV_EMBED_ROOT,
            "rotten_tomatoes_parallel_en_Yoruba_qwen3_8B_0.95_perc_",
            "qwen3_8B_0.95_perc.json",
        ),
    ),
    (
        8676,
        _path_with_name(
            NV_EMBED_ROOT,
            "rotten_tomatoes_proper_sentences_parallel_en_Balinese_qwen3_8B_0.95_perc_",
            "qwen3_8B_0.95_perc.json",
        ),
    ),
    (
        9395,
        _path_with_name(
            NV_EMBED_ROOT,
            "rotten_tomatoes_proper_sentences_parallel_en_Danish_qwen3_8B_0.95_perc_",
            "qwen3_8B_0.95_perc.json",
        ),
    ),
    (
        9494,
        _path_with_name(
            NV_EMBED_ROOT,
            "rotten_tomatoes_proper_sentences_parallel_en_French_qwen3_8B_0.95_perc_",
            "qwen3_8B_0.95_perc.json",
        ),
    ),
    (
        7813,
        _path_with_name(
            NV_EMBED_ROOT,
            "rotten_tomatoes_proper_sentences_parallel_en_Husa_qwen3_8B_0.95_perc_",
            "qwen3_8B_0.95_perc.json",
        ),
    ),
    (
        7342,
        _path_with_name(
            NV_EMBED_ROOT,
            "rotten_tomatoes_proper_sentences_parallel_en_Igbo_qwen3_8B_0.95_perc_",
            "qwen3_8B_0.95_perc.json",
        ),
    ),
    (
        6920,
        _path_with_name(
            NV_EMBED_ROOT,
            "rotten_tomatoes_proper_sentences_parallel_en_Indonesian_qwen3_8B_0.95_perc_",
            "qwen3_8B_0.95_perc.json",
        ),
    ),
    (
        7838,
        _path_with_name(
            NV_EMBED_ROOT,
            "rotten_tomatoes_proper_sentences_parallel_en_Javanese_qwen3_8B_0.95_perc_",
            "qwen3_8B_0.95_perc.json",
        ),
    ),
    (
        9237,
        _path_with_name(
            NV_EMBED_ROOT,
            "rotten_tomatoes_proper_sentences_parallel_en_Nigerian-Pidgin_qwen3_8B_0.95_perc_",
            "qwen3_8B_0.95_perc.json",
        ),
    ),
    (
        7808,
        _path_with_name(
            NV_EMBED_ROOT,
            "rotten_tomatoes_proper_sentences_parallel_en_Sundanese_qwen3_8B_0.95_perc_",
            "qwen3_8B_0.95_perc.json",
        ),
    ),
    (
        7901,
        _path_with_name(
            NV_EMBED_ROOT,
            "rotten_tomatoes_proper_sentences_parallel_en_Yoruba_qwen3_8B_0.95_perc_",
            "qwen3_8B_0.95_perc.json",
        ),
    ),
    (20000, NV_EMBED_TRAINING / "AllNLI-jpn-20k_qwen3_8B_0_95_perc_no_negs.json"),
    (10000, NV_EMBED_TRAINING / "multi_lng_2_nld-Latn_qwen3_8B_0_95_perc_no_negs.json"),
    (10000, NV_EMBED_TRAINING / "multi_lng_1_ara-Arab_qwen3_8B_0_95_perc_no_negs.json"),
    (10000, NV_EMBED_TRAINING / "multi_lng_1_deu-Latn_qwen3_8B_0_95_perc_no_negs.json"),
    (10000, NV_EMBED_TRAINING / "multi_lng_2_fra-Latn_qwen3_8B_0_95_perc_no_negs.json"),
    (10000, NV_EMBED_TRAINING / "multi_lng_1_tur-Latn_qwen3_8B_0_95_perc_no_negs.json"),
    (10000, NV_EMBED_TRAINING / "mono_spa-Latn_qwen3_8B_0_95_perc_no_negs.json"),
    (10000, NV_EMBED_TRAINING / "mono_ara-Arab_qwen3_8B_0_95_perc_no_negs.json"),
    (10000, NV_EMBED_TRAINING / "mono_kor-Hang_qwen3_8B_0_95_perc_no_negs.json"),
    (10000, NV_EMBED_TRAINING / "multi_lng_2_spa-Latn_qwen3_8B_0_95_perc_no_negs.json"),
    (10000, NV_EMBED_TRAINING / "multi_lng_2_ita-Latn_qwen3_8B_0_95_perc_no_negs.json"),
    (30000, NV_EMBED_TRAINING / "BigPatentClustering-extra.json"),
    (10000, NV_EMBED_TRAINING / "sprint_mix_synth_15k_qwen3_8B_0_95_perc_no_negs.json"),
]

ADDITIONAL_TRAINING_DATASETS: list[DatasetSpec] = [
    (None, TEXTQA_TRAIN / "miracl_train_ar_llama3_1b_4m_512len_train.json"),
    (None, TEXTQA_TRAIN / "miracl_train_bn_ronaya_llama_train.json"),
    (None, TEXTQA_TRAIN / "miracl_train_en_llama3_1b_4m_512len_train.json"),
    (None, TEXTQA_TRAIN / "miracl_train_es_llama3_1b_4m_512len_train.json"),
    (None, TEXTQA_TRAIN / "miracl_train_fa_llama3_1b_4m_512len_train.json"),
    (None, TEXTQA_TRAIN / "miracl_train_fi_ronaya_llama_train.json"),
    (None, TEXTQA_TRAIN / "miracl_train_fr_llama3_1b_4m_512len_train.json"),
    (None, TEXTQA_TRAIN / "miracl_train_hi_ronaya_llama_train.json"),
    (None, TEXTQA_TRAIN / "miracl_train_id_ronaya_llama_train.json"),
    (None, TEXTQA_TRAIN / "miracl_train_ja_ronaya_llama_train.json"),
    (None, TEXTQA_TRAIN / "miracl_train_ko_ronaya_llama_train.json"),
    (None, TEXTQA_TRAIN / "miracl_train_ru_ronaya_llama_train.json"),
    (None, TEXTQA_TRAIN / "miracl_train_sw_ronaya_llama_train.json"),
    (None, TEXTQA_TRAIN / "miracl_train_te_ronaya_llama_train.json"),
    (None, TEXTQA_TRAIN / "miracl_train_th_ronaya_llama_train.json"),
    (None, TEXTQA_TRAIN / "miracl_train_zh_ronaya_llama_train.json"),
    (None, TEXTQA_TRAIN / "mlqa-translate-train.ar_train.json"),
    (None, TEXTQA_TRAIN / "mlqa-translate-train.de_train.json"),
    (None, TEXTQA_TRAIN / "mlqa-translate-train.es_train.json"),
    (None, TEXTQA_TRAIN / "mlqa-translate-train.hi_train.json"),
    (None, TEXTQA_TRAIN / "mlqa-translate-train.zh_train.json"),
    (None, NV_EMBED_TRAINING / "amazon_counterfactual_classification.json"),
    (None, NV_EMBED_TRAINING / "banking77_classification.json"),
    (None, NV_EMBED_TRAINING / "imdb_classification.json"),
    (None, NV_EMBED_TRAINING / "mtop_intent_classification.json"),
    (None, NV_EMBED_TRAINING / "mtop_domain_classification.json"),
    (None, NV_EMBED_TRAINING / "massive_intent_classification.json"),
    (None, NV_EMBED_TRAINING / "massive_scenario_classification.json"),
    (None, NV_EMBED_TRAINING / "toxic_conversations_classification.json"),
    (None, NV_EMBED_TRAINING / "tweet_sentiment_extraction_classification.json"),
    (None, NV_EMBED_TRAINING / "arxiv_clustering_s2s.json"),
    (None, NV_EMBED_TRAINING / "arxiv_clustering_p2p.json"),
    (None, NV_EMBED_TRAINING / "biorxiv_clustering_p2p.json"),
    (None, NV_EMBED_TRAINING / "biorxiv_clustering_s2s.json"),
    (None, NV_EMBED_TRAINING / "medrxiv_clustering_p2p.json"),
    (None, NV_EMBED_TRAINING / "medrxiv_clustering_s2s.json"),
    (None, NV_EMBED_TRAINING / "stackexchange_clustering_s2s.json"),
    (None, NV_EMBED_TRAINING / "stackexchange_clustering_p2p.json"),
    (None, NV_EMBED_TRAINING / "twenty_newsgroups_clustering.json"),
    (None, NV_EMBED_TRAINING / "STS12.json"),
    (None, NV_EMBED_TRAINING / "STS22.json"),
    (None, NV_EMBED_TRAINING / "STSBenchmark.json"),
    (None, NV_EMBED_TRAINING / "arguana.json"),
    (None, NV_EMBED_TRAINING / "fiqa.json"),
    (None, NV_EMBED_TRAINING / "nli.json"),
]

# ALL_TRAINING_DATASETS = TRAINING_DATASETS + ADDITIONAL_TRAINING_DATASETS
ALL_TRAINING_DATASETS = [
    (expected_records, dataset_path)
    for expected_records, dataset_path in TRAINING_DATASETS + ADDITIONAL_TRAINING_DATASETS
    if dataset_path.is_relative_to(RONAYA_TRAINING)
]

CSV_FIELDNAMES = [
    "dataset_path",
    "expected_records",
    "total_records",
    "question_records",
    "unique_questions",
    "overlap_records",
    "overlap_unique_questions",
    "overlap_record_fraction",
    "overlap_unique_fraction",
    "status",
]


def load_training_records(path: Path) -> list[Any]:
    """Load records from a NeMo Retriever training-format JSON file."""
    with path.open("r", encoding="utf-8") as input_file:
        payload = json.load(input_file)

    if not isinstance(payload, dict):
        raise TypeError(f"expected a JSON object at top level, got {type(payload).__name__}")

    records = payload.get("data")
    if not isinstance(records, list):
        raise TypeError("expected top-level 'data' to be a list")

    return records


def build_question_set(path: Path) -> set[str]:
    """Load all string-valued question fields from a training-format JSON file into a set."""
    records = load_training_records(path)
    questions: set[str] = set()
    skipped_records = 0

    for record in records:
        if not isinstance(record, dict):
            skipped_records += 1
            continue

        question = record.get("question")
        if isinstance(question, str):
            questions.add(question)
        else:
            skipped_records += 1

    LOGGER.info(
        "Loaded %s unique reference questions from %s records in %s; skipped %s records without a string question",
        f"{len(questions):,}",
        f"{len(records):,}",
        path,
        f"{skipped_records:,}",
    )
    return questions


def summarize_dataset(expected_records: int | None, path: Path, reference_questions: set[str]) -> SummaryRow:
    """Compare one training dataset's questions against the reference question set."""
    base_row: SummaryRow = {
        "dataset_path": str(path),
        "expected_records": expected_records if expected_records is not None else "",
        "total_records": 0,
        "question_records": 0,
        "unique_questions": 0,
        "overlap_records": 0,
        "overlap_unique_questions": 0,
        "overlap_record_fraction": "0.000000",
        "overlap_unique_fraction": "0.000000",
        "status": "missing",
    }

    if not path.exists():
        LOGGER.warning("Missing dataset: %s", path)
        return base_row

    try:
        records = load_training_records(path)
    except (json.JSONDecodeError, OSError, TypeError) as error:
        LOGGER.warning("Could not load %s: %s", path, error)
        base_row["status"] = f"error:{error.__class__.__name__}"
        return base_row

    total_records = len(records)
    question_records = 0
    overlap_records = 0
    unique_questions: set[str] = set()

    for record in records:
        if not isinstance(record, dict):
            continue

        question = record.get("question")
        if not isinstance(question, str):
            continue

        question_records += 1
        unique_questions.add(question)
        if question in reference_questions:
            overlap_records += 1

    overlap_unique_questions = len(unique_questions.intersection(reference_questions))
    status = "ok"
    if expected_records is not None and total_records != expected_records:
        status = "count_mismatch"
    if question_records == 0:
        status = "no_questions"

    base_row.update(
        {
            "total_records": total_records,
            "question_records": question_records,
            "unique_questions": len(unique_questions),
            "overlap_records": overlap_records,
            "overlap_unique_questions": overlap_unique_questions,
            "overlap_record_fraction": f"{overlap_records / question_records:.6f}" if question_records else "0.000000",
            "overlap_unique_fraction": (
                f"{overlap_unique_questions / len(unique_questions):.6f}" if unique_questions else "0.000000"
            ),
            "status": status,
        }
    )
    return base_row


def write_csv(rows: list[SummaryRow], output_csv: Path | None) -> None:
    """Write overlap rows to stdout or to an output CSV file."""
    if output_csv is None:
        writer = csv.DictWriter(sys.stdout, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
        return

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Check exact-string overlap between queries in listed training datasets and "
            "the reference 11M training-format JSON."
        )
    )
    parser.add_argument(
        "--reference-json",
        type=Path,
        default=REFERENCE_DATASET,
        help=f"Reference training-format JSON to load into RAM (default: {REFERENCE_DATASET})",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Optional CSV output path. If omitted, writes CSV to stdout.",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
        help="Logging verbosity.",
    )
    return parser.parse_args()


def main() -> int:
    """Run the query-overlap check."""
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s:%(name)s:%(message)s")

    reference_questions = build_question_set(args.reference_json)
    rows: list[SummaryRow] = []

    for expected_records, dataset_path in ALL_TRAINING_DATASETS:
        LOGGER.info("Checking %s", dataset_path)
        rows.append(summarize_dataset(expected_records, dataset_path, reference_questions))

    write_csv(rows, args.output_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
