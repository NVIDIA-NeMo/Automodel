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

"""``te_nvfp4`` through real Transformer Engine modules (Blackwell GPUs only)."""

import pytest
import torch

from nemo_automodel.components.models.common.utils import HAVE_TE, TENvfp4Config
from nemo_automodel.shared.import_utils import is_te_min_version


def _nvfp4_capable() -> bool:
    return HAVE_TE and torch.cuda.is_available() and torch.cuda.get_device_capability() >= (10, 0)


pytestmark = pytest.mark.skipif(
    not (_nvfp4_capable() and is_te_min_version("2.19.0")),
    reason="requires Transformer Engine 2.19+ and an NVFP4 GPU",
)

# Row-scaled activations need the random Hadamard transform off; the backward reads dequantized operands so
# that the weights do not have to be frozen.
ROW_SCALED_OPTIONS = {
    "random_hadamard_transform": False,
    "row_scaled_activation": True,
    "backward": "dequantized",
}


def test_large_default_gemms_have_no_wrong_rows():
    """Default NVFP4 GEMMs at 4096 rows x 4608 x 4608, forward and backward: every row within rounding error.

    This is the size at which a row-scaled forward GEMM returns wrong rows with some cuBLASLt versions (next
    test). The default recipe multiplies into a BF16 output and must be correct with every version.
    """
    import transformer_engine.pytorch as te

    torch.manual_seed(0)
    linear = te.Linear(4608, 4608, bias=False, params_dtype=torch.bfloat16).to("cuda")
    x = torch.randn(4096, 4608, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    output_grad = torch.randn(4096, 4608, device="cuda", dtype=torch.bfloat16)
    with TENvfp4Config().maybe_te_autocast():
        output = linear(x)
        output.backward(output_grad)
    weight, inputs, gradient = linear.weight.detach().float(), x.detach().float(), output_grad.float()
    results = {
        "output": (output.detach().float(), inputs @ weight.T),
        "input gradient": (x.grad.float(), gradient @ weight),
        "weight gradient": (linear.weight.grad.float(), gradient.T @ inputs),
    }
    for name, (value, exact) in results.items():
        row_error = (value - exact).norm(dim=1) / exact.norm(dim=1)
        # NVFP4 rounding gives every row an error of 0.15 to 0.19; a row from a defective GEMM is off by about 1.
        assert 0.05 < float(row_error.median()) and float(row_error.max()) < 0.3, (name, row_error.max())


def test_large_row_scaled_gemm_is_right_wherever_the_config_does_not_warn():
    """Row-scaled NVFP4 at 4096 rows x 4608 x 4608: every row is right where the config does not warn.

    Where it warns, wrong rows are expected but not required: cuBLASLt 13.7.0 on GB300 returned 96 wrong rows
    of 4096 here and 13.8.1 none; a release between the two was not measured.
    """
    import transformer_engine.pytorch as te

    from nemo_automodel.components.models.common.utils import row_scaled_nvfp4_gemm_is_unreliable
    from nemo_automodel.shared.import_utils import get_te_cublaslt_version

    torch.manual_seed(0)
    linear = te.Linear(4608, 4608, bias=False, params_dtype=torch.bfloat16).to("cuda")
    x = torch.randn(4096, 4608, device="cuda", dtype=torch.bfloat16)
    config = TENvfp4Config(**ROW_SCALED_OPTIONS)
    with torch.no_grad(), config.maybe_te_autocast():
        output = linear(x).float()
    exact = x.float() @ linear.weight.float().T
    # NVFP4 rounding gives every row an error near 0.13; a row from the defective GEMM is off by about 1.
    row_error = (output - exact).norm(dim=1) / exact.norm(dim=1)
    wrong_rows = int((row_error > 0.5).sum())
    assert float(row_error.median()) < 0.2
    cublaslt_version = get_te_cublaslt_version()
    if not row_scaled_nvfp4_gemm_is_unreliable(cublaslt_version, torch.cuda.get_device_capability()):
        assert wrong_rows == 0, f"{wrong_rows} wrong rows of 4096 with cuBLASLt {cublaslt_version}"
