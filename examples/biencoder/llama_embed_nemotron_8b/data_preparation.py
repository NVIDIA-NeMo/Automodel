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
"""
Data preparation script for embed-nemotron-dataset-v1 datasets.
Downloads and restores datasets from HuggingFace repositories.

Usage:
    python examples/biencoder/llama_embed_nemotron_8b/data_preparation.py --download-path path/to/dataset

Example:
    python examples/biencoder/llama_embed_nemotron_8b/data_preparation.py \
        --download-path ./embed_nemotron_dataset_v1
"""

import argparse
import json
import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from datasets import load_dataset
from huggingface_hub import HfApi, hf_hub_download
from tqdm import tqdm

# Configuration
REPO_ID = "nvidia/embed-nemotron-dataset-v1"
TOKEN = os.environ.get("HF_TOKEN")
api = HfApi(token=TOKEN)


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return super(NumpyEncoder, self).default(obj)


def get_mteb_lookup_dictionaries(source_config: Dict) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Load dataset from MTEB format and create lookup dictionaries for queries and documents.

    Args:
        source_config: Configuration dict containing source_repo path

    Returns:
        Tuple of (query_lookup, doc_lookup) dictionaries
    """
    mteb_queries = load_dataset(source_config["source_repo"], "queries", split="queries")
    mteb_corpus = load_dataset(source_config["source_repo"], "corpus", split="corpus")

    query_lookup = {f"q_{i}": query for i, query in tqdm(enumerate(mteb_queries["text"]), desc="Loading queries")}
    doc_lookup = {f"d_{i}": doc for i, doc in tqdm(enumerate(mteb_corpus["text"]), desc="Loading corpus")}

    return query_lookup, doc_lookup


def get_column_lookup_dictionaries(source_config: Dict) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Load dataset from column format and create lookup dictionaries for queries and documents.

    Args:
        source_config: Configuration dict with source_repo, query_column, and document_column

    Returns:
        Tuple of (query_lookup, doc_lookup) dictionaries
    """
    if "subset" in source_config:
        data = load_dataset(source_config["source_repo"], source_config["subset"], split=source_config["split"])
    else:
        data = load_dataset(source_config["source_repo"], split=source_config["split"])

    query_column = source_config["query_column"]
    document_column = source_config["document_column"]

    query_lookup = {f"q_{i}": query for i, query in tqdm(enumerate(data[query_column]), desc="Loading queries")}
    doc_lookup = {f"d_{i}": doc for i, doc in tqdm(enumerate(data[document_column]), desc="Loading corpus")}

    return query_lookup, doc_lookup


LOADER_REGISTRY = {
    "mteb": get_mteb_lookup_dictionaries,
    "column": get_column_lookup_dictionaries,
}


def restore_dataset(ds_name: str, repo_id: str, local_base_path: str) -> None:
    """
    Restore a complete dataset from HuggingFace repository.

    Args:
        ds_name: Name of the dataset to restore
        repo_id: HuggingFace repository ID
        local_base_path: Base path for storing restored datasets
    """

    print(f"\n--- Processing: {ds_name} ---")

    # 1. Setup Directory
    dataset_root = os.path.join(local_base_path, ds_name)
    corpus_dir = os.path.join(dataset_root, "corpus")
    os.makedirs(corpus_dir, exist_ok=True)

    # 2. Prepare metadata
    local_meta_path = hf_hub_download(
        repo_id=repo_id,
        filename=f"{ds_name}/dataset_metadata.json",
        repo_type="dataset",
        token=TOKEN,
        local_dir=local_base_path,
    )
    os.replace(local_meta_path, os.path.join(corpus_dir, "merlin_metadata.json"))

    with open(os.path.join(corpus_dir, "merlin_metadata.json"), "rb") as file:
        metadata = json.load(file)

    if metadata["ids_only"]:
        config_path = hf_hub_download(
            repo_id=repo_id, filename=f"{ds_name}/source_config.json", repo_type="dataset", token=TOKEN
        )
        with open(config_path, "r") as f:
            config = json.load(f)

    # 3. Get Lookup Maps
    if metadata["ids_only"]:
        source_loader = LOADER_REGISTRY[config["loader_config"]]
        query_lookup, doc_lookup = source_loader(config)

    # 4. Download and Restore Corpus
    print("   -> Downloading Corpus...")
    corpus_url = f"hf://datasets/{repo_id}/{ds_name}/corpus.parquet"
    corpus_df = pd.read_parquet(corpus_url)

    if metadata["ids_only"]:
        corpus_df["text"] = corpus_df["id"].apply(lambda x: doc_lookup[x])

    corpus_df.to_parquet(os.path.join(corpus_dir, "train.parquet"), index=False)

    # 5. Reconstruct Queries
    print("   -> Reconstructing Main JSON...")
    queries_url = f"hf://datasets/{repo_id}/{ds_name}/queries.parquet"
    queries_df = pd.read_parquet(queries_url)

    if metadata["ids_only"]:
        queries_df["question"] = queries_df["question"].apply(lambda x: query_lookup[x])
    data_list = queries_df.to_dict(orient="records")

    # 6. Final JSON Save
    final_structure = {"data": data_list, "corpus": {"path": os.path.abspath(corpus_dir)}}

    main_json_path = os.path.join(dataset_root, f"{ds_name}.json")
    with open(main_json_path, "w", encoding="utf-8") as f:
        json.dump(final_structure, f, indent=2, cls=NumpyEncoder)

    print(f"   [+] Restored {ds_name}")


def main():
    parser = argparse.ArgumentParser(
        description="Download and restore embed-nemotron-dataset-v1 datasets from HuggingFace"
    )
    parser.add_argument(
        "--download-path",
        type=str,
        default="./embed_nemotron_dataset_v1",
        help="Base path for downloading and storing datasets (default: ./embed_nemotron_dataset_v1)",
    )
    args = parser.parse_args()

    base_download_path = args.download_path

    print(f"Scanning repository: {REPO_ID}...")
    tree = api.list_repo_tree(repo_id=REPO_ID, repo_type="dataset", recursive=True)

    dataset_names = set()
    for item in tree:
        if item.path.endswith("corpus.parquet"):
            folder_name = os.path.dirname(item.path)
            if folder_name and folder_name != ".":
                dataset_names.add(folder_name)

    print(f"Found {len(dataset_names)} datasets.")

    for ds in dataset_names:
        restore_dataset(ds, REPO_ID, base_download_path)

    print("\nAll complete.")


if __name__ == "__main__":
    main()
