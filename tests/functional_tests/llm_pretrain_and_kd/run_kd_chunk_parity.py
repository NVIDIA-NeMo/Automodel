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

"""Compare chunked and unchunked KD loss and student gradients on CPU.

Run from the repository root with:
    uv run python -m tests.functional_tests.llm_pretrain_and_kd.run_kd_chunk_parity
"""

from itertools import product

import torch

from nemo_automodel.components.loss.kd_loss import KDLoss


def main() -> None:
    """Exercise precision, temperature, masking, and uneven token chunks."""
    generator = torch.Generator().manual_seed(42)
    student = torch.randn(2, 7, 17, generator=generator)
    teacher = torch.randn(2, 7, 17, generator=generator)
    completed = 0
    for dtype, fp32_upcast, temperature, masked in product(
        (torch.bfloat16, torch.float32), (False, True), (1.0, 3.0), (False, True)
    ):
        labels = torch.zeros(2, 7, dtype=torch.long)
        if masked:
            labels[0, 1:4] = -100
        reference_student = student.to(dtype).clone().requires_grad_()
        chunked_student = student.to(dtype).clone().requires_grad_()
        teacher_logits = teacher.to(dtype)
        reference_loss = KDLoss(temperature=temperature, fp32_upcast=fp32_upcast)(
            reference_student, teacher_logits, labels
        )
        chunked_loss = KDLoss(temperature=temperature, fp32_upcast=fp32_upcast, chunk_size=4)(
            chunked_student, teacher_logits, labels
        )
        reference_loss.backward()
        chunked_loss.backward()
        case = f"dtype={dtype}, fp32_upcast={fp32_upcast}, temperature={temperature}, masked={masked}"
        if not torch.isfinite(chunked_loss) or not torch.equal(chunked_loss, reference_loss):
            raise RuntimeError(f"KD loss mismatch: {case}")
        if chunked_student.grad is None or reference_student.grad is None:
            raise RuntimeError(f"Missing student gradient: {case}")
        if not torch.equal(chunked_student.grad, reference_student.grad):
            raise RuntimeError(f"KD student gradient mismatch: {case}")
        if torch.count_nonzero(chunked_student.grad[labels == -100]):
            raise RuntimeError(f"Masked tokens have nonzero student gradients: {case}")
        completed += 1
    print(f"KD chunk parity passed: {completed} CPU loss and gradient scenarios")


if __name__ == "__main__":
    main()
