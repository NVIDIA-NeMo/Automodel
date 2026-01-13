import json
from pathlib import Path
from typing import List, Tuple

import torch


def _read_losses_from_jsonl(file_path: Path) -> List[float]:
    """Read a JSONL file and extract the 'loss' values from each record.

    Blank lines are ignored. Raises AssertionError if a record is missing the
    'loss' field or the value is not numeric.
    """
    losses: List[float] = []
    with file_path.open("r", encoding="utf-8") as f:
        for line_index, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            record = json.loads(stripped)
            assert "loss" in record, f"Missing 'loss' at line {line_index} in {file_path}"
            loss_value = record["loss"]
            assert isinstance(loss_value, (int, float)), (
                f"Non-numeric 'loss' at line {line_index} in {file_path}: {loss_value!r}"
            )
            losses.append(float(loss_value))
    return losses


def test_log_compare_losses_with_ground_truth(jsonl_paths: Tuple[Path, Path]) -> None:
    """Compare the 'loss' field between ground-truth JSONL and provided JSONL.

    This test asserts that each corresponding 'loss' value is torch.allclose() equal to the ground-truth value.
    """
    ground_truth_jsonl_path, compare_jsonl_path = jsonl_paths

    assert ground_truth_jsonl_path.exists(), f"Ground truth file not found: {ground_truth_jsonl_path}"
    assert compare_jsonl_path.exists(), f"Comparison file not found: {compare_jsonl_path}"

    gt_losses = _read_losses_from_jsonl(ground_truth_jsonl_path)
    cmp_losses = _read_losses_from_jsonl(compare_jsonl_path)

    assert len(gt_losses) == len(cmp_losses), (
        f"Files have different number of records: ground-truth={len(gt_losses)} vs compare={len(cmp_losses)}"
    )

    mismatches: List[str] = []
    for index, (gt_loss, cmp_loss) in enumerate(zip(gt_losses, cmp_losses)):
        gt_tensor = torch.tensor(gt_loss, dtype=torch.float32)
        cmp_tensor = torch.tensor(cmp_loss, dtype=torch.float32)
        if not torch.allclose(gt_tensor, cmp_tensor, rtol=1e-5, atol=1e-6):
            mismatches.append(f"idx={index}: gt_loss={gt_loss:.9f} cmp_loss={cmp_loss:.9f}")
            # Keep collecting all mismatches to show a helpful summary

    assert not mismatches, (
        "Loss values differ (torch.allclose == False) for the following records:\n"
        + "\n".join(mismatches[:20])
        + ("\n..." if len(mismatches) > 20 else "")
    )
