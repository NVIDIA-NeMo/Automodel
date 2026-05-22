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

"""Materialize one AutoModel retrieval HF subset as local corpus JSON for mining."""

import argparse
import json
import logging
import shutil
import tempfile
from pathlib import Path

from datasets import Dataset

from nemo_automodel.components.datasets.llm.retrieval_dataset import _load_hf_subset

logger = logging.getLogger(__name__)


def main() -> int:
    """Run the HF subset materialization CLI."""
    parser = argparse.ArgumentParser(
        description="Write one AutoModel-schema hf:// retrieval subset as a local corpus-backed JSON file"
    )
    parser.add_argument(
        "repo_id", type=str, help="Hugging Face dataset repo, for example nvidia/embed-nemotron-dataset-v1"
    )
    parser.add_argument("subset", type=str, help="Subset name, for example FEVER")
    parser.add_argument(
        "output_dir", type=str, help="Directory where train.json and the corpus directory will be written"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into a non-empty output directory.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    output_dir = Path(args.output_dir)
    if output_dir.exists():
        if not output_dir.is_dir():
            parser.error(f"output_dir must be a directory: {output_dir}")
        if any(output_dir.iterdir()) and not args.overwrite:
            parser.error(f"output_dir is not empty: {output_dir}. Pass --overwrite to replace existing files.")
    else:
        output_dir.mkdir(parents=True)
    data_list, corpus_info = _load_hf_subset(args.repo_id, args.subset)
    corpus_dir = output_dir / f"{args.subset}_corpus"
    tmp_corpus_dir = Path(tempfile.mkdtemp(prefix=f".{args.subset}_corpus.", dir=output_dir))
    tmp_train_path = None
    backup_corpus_dir = None
    try:
        doc_rows = []
        for doc_id in corpus_info.get_all_ids():
            document = corpus_info.get_document_by_id(doc_id)
            doc_rows.append({"id": str(doc_id), "text": document.get("text", "")})

        Dataset.from_list(doc_rows).to_parquet(str(tmp_corpus_dir / "train.parquet"))
        metadata = {
            **corpus_info.metadata,
            "class": "TextQADataset",
            "corpus_id": corpus_info.corpus_id,
        }
        with open(tmp_corpus_dir / "merlin_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2, sort_keys=True)

        train_json = {
            "corpus": [{"path": f"./{corpus_dir.name}"}],
            "data": data_list,
        }
        with tempfile.NamedTemporaryFile("w", dir=output_dir, prefix=".train.", suffix=".json", delete=False) as f:
            tmp_train_path = Path(f.name)
            json.dump(train_json, f, indent=2)

        if corpus_dir.exists():
            backup_corpus_dir = Path(tempfile.mkdtemp(prefix=f".{corpus_dir.name}.backup.", dir=output_dir))
            backup_corpus_dir.rmdir()
            shutil.move(str(corpus_dir), str(backup_corpus_dir))
        shutil.move(str(tmp_corpus_dir), str(corpus_dir))
        tmp_corpus_dir = None
        tmp_train_path.replace(output_dir / "train.json")
        tmp_train_path = None
    except Exception:
        if backup_corpus_dir is not None and backup_corpus_dir.exists():
            if corpus_dir.exists():
                shutil.rmtree(corpus_dir)
            shutil.move(str(backup_corpus_dir), str(corpus_dir))
        raise
    finally:
        if tmp_corpus_dir is not None and tmp_corpus_dir.exists():
            shutil.rmtree(tmp_corpus_dir)
        if tmp_train_path is not None and tmp_train_path.exists():
            tmp_train_path.unlink()
        if backup_corpus_dir is not None and backup_corpus_dir.exists():
            shutil.rmtree(backup_corpus_dir)

    logger.info("Wrote %s records and %s corpus documents to %s", len(data_list), len(doc_rows), output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
