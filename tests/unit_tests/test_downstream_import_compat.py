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

"""Import-compatibility guard for known downstream consumers.

NeMo-RL and verl import Automodel internals directly, including paths that the
transformers/diffusers split and the subsequent underscore drop relocated.
Those keep working only because of the back-compat aliases installed in
``nemo_automodel/__init__.py``. This module pins the exact imports both
projects perform so that a refactor which quietly drops an alias fails here,
rather than in their CI.

    ############################################################
    #  TEMPORARY -- DELETE THIS FILE once no entry below is
    #  marked ``shim=True``, i.e. once NeMo-RL and verl have
    #  migrated onto canonical paths. Nothing else depends on it.
    ############################################################

Entries were collected by parsing every ``nemo_automodel`` import in each
project at the pinned revisions below; the ``source`` field records where each
one lives so the table can be re-derived:

* NeMo-RL @ 2e026c33 (github.com/NVIDIA-NeMo/RL)
* verl    @ 535c477  (github.com/volcengine/verl)

Deliberately **not** covered, because they are already broken against Automodel
main for reasons unrelated to any relocation -- these are version drift, and
asserting on them would encode someone else's bug as our contract:

* ``components.distributed.cp_utils`` (NeMo-RL) -- module deleted upstream in
  commit 70cb08692, "refactor(distributed): unify CP input prep and dispatch".
* ``components.distributed.mesh_utils.create_device_mesh`` (verl) -- symbol gone.
* ``components.moe.config.MoEParallelizerConfig`` (verl) -- symbol gone.
* ``recipes.llm.train_ft.build_optimizer`` (verl) -- symbol gone.
"""

import importlib
import pathlib
import warnings

import pytest

import nemo_automodel
from nemo_automodel import _LEGACY_MODELS_ALIAS

# Resolution must come from the checkout under test. An editable install of
# nemo_automodel pointing at a *different* checkout installs a meta-path finder
# that supplies any module missing here, which would let a deleted or renamed
# module still appear to import.
_PACKAGE_ROOT = pathlib.Path(nemo_automodel.__file__).resolve().parent

# (module, symbols, source, shim) -- shim=True means the path only resolves via
# the back-compat aliases in nemo_automodel/__init__.py.
_NEMO_RL_IMPORTS = [
    ("nemo_automodel._transformers.registry", ("ModelRegistry",), "nemo_rl/models/automodel/setup.py:21", False),
    (
        "nemo_automodel._transformers.utils",
        ("sliding_window_overwrite",),
        "nemo_rl/models/automodel/setup.py:22",
        False,
    ),
    (
        "nemo_automodel._transformers.auto_model",
        ("NeMoAutoModelForCausalLM", "NeMoAutoModelForImageTextToText", "NeMoAutoModelForTextToWaveform"),
        "nemo_rl/models/policy/utils.py:34",
        False,
    ),
    (
        "nemo_automodel",
        ("NeMoAutoModelForSequenceClassification",),
        "nemo_rl/models/automodel/setup.py:147",
        False,
    ),
    ("nemo_automodel.components.config.loader", ("_resolve_target",), "nemo_rl/models/automodel/setup.py:23", False),
    (
        "nemo_automodel.components.distributed.fsdp2",
        ("FSDP2Manager",),
        "nemo_rl/models/automodel/setup.py:24",
        False,
    ),
    (
        "nemo_automodel.components.distributed.tensor_utils",
        ("get_cpu_state_dict", "to_local_if_dtensor"),
        "nemo_rl/models/policy/dtensor_policy_worker_v2.py:39",
        False,
    ),
    (
        "nemo_automodel.components.moe.parallelizer",
        ("parallelize_model",),
        "nemo_rl/models/automodel/setup.py:298",
        False,
    ),
    (
        "nemo_automodel.components.training.utils",
        ("scale_grads_and_clip_grad_norm",),
        "nemo_rl/models/automodel/train.py:25",
        False,
    ),
    (
        "nemo_automodel.components.checkpoint._backports.filesystem",
        ("SerializationFormat",),
        "nemo_rl/models/policy/dtensor_policy_worker_v2.py:26",
        False,
    ),
    (
        "nemo_automodel.components.checkpoint.checkpointing",
        ("Checkpointer", "CheckpointingConfig"),
        "nemo_rl/models/policy/dtensor_policy_worker_v2.py:29",
        False,
    ),
]

_VERL_IMPORTS = [
    (
        "nemo_automodel._transformers.auto_model",
        ("NeMoAutoModelForCausalLM",),
        "verl/workers/engine/automodel/utils.py:140",
        False,
    ),
    (
        "nemo_automodel._transformers.utils",
        ("apply_cache_compatibility_patches",),
        "verl/workers/engine/automodel/transformer_impl.py:93",
        False,
    ),
    (
        "nemo_automodel.components.models.common.utils",
        ("BackendConfig",),
        "verl/workers/engine/automodel/utils.py:163",
        True,
    ),
    (
        "nemo_automodel.components.checkpoint.checkpointing",
        ("Checkpointer", "CheckpointingConfig"),
        "verl/workers/engine/automodel/transformer_impl.py:31",
        False,
    ),
    (
        "nemo_automodel.components.optim.scheduler",
        ("OptimizerParamScheduler",),
        "verl/workers/engine/automodel/transformer_impl.py:32",
        False,
    ),
    (
        "nemo_automodel.components.training.utils",
        ("prepare_for_final_backward", "prepare_for_grad_accumulation", "scale_grads_and_clip_grad_norm"),
        "verl/workers/engine/automodel/transformer_impl.py:33",
        False,
    ),
    (
        "nemo_automodel.shared.te_patches",
        ("apply_te_patches",),
        "verl/workers/engine/automodel/transformer_impl.py:94",
        False,
    ),
    (
        "nemo_automodel.components.config.loader",
        ("ConfigNode",),
        "verl/workers/engine/automodel/transformer_impl.py:158",
        False,
    ),
    (
        "nemo_automodel.components.moe.megatron.moe_utils",
        ("MoEAuxLossAutoScaler",),
        "verl/workers/engine/automodel/transformer_impl.py:244",
        False,
    ),
    (
        "nemo_automodel.components.distributed.config",
        ("DDPConfig", "FSDP2Config", "MegatronFSDPConfig"),
        "verl/workers/engine/automodel/utils.py:77",
        False,
    ),
    (
        "nemo_automodel.components.quantization.fp8",
        ("FP8Config",),
        "verl/workers/engine/automodel/utils.py:145",
        False,
    ),
    (
        "nemo_automodel.components.utils.compile_utils",
        ("CompileConfig",),
        "verl/workers/engine/automodel/utils.py:150",
        False,
    ),
]

_ALL_IMPORTS = [("NeMo-RL", *row) for row in _NEMO_RL_IMPORTS] + [("verl", *row) for row in _VERL_IMPORTS]

# Prefixes that exist only as back-compat aliases.
_SHIM_PREFIXES = (_LEGACY_MODELS_ALIAS,)


def _import(module: str):
    with warnings.catch_warnings():
        # The relocated paths intentionally warn; that is asserted elsewhere.
        warnings.simplefilter("ignore", DeprecationWarning)
        return importlib.import_module(module)


@pytest.mark.parametrize(
    "project, module, symbols, source, shim",
    _ALL_IMPORTS,
    ids=[f"{p}-{row[0]}" for p, *row in _ALL_IMPORTS],
)
def test_downstream_import_still_resolves(project, module, symbols, source, shim):
    """Each import a downstream project performs must still resolve."""
    try:
        mod = _import(module)
    except ImportError as exc:  # pragma: no cover - only on a real regression
        pytest.fail(f"{project} imports '{module}' ({source}) but it no longer resolves: {exc}")

    origin = getattr(mod, "__file__", None)
    assert origin is not None, f"'{module}' resolved without a __file__"
    assert pathlib.Path(origin).resolve().is_relative_to(_PACKAGE_ROOT), (
        f"{project} imports '{module}' ({source}), but it resolved from {origin} "
        f"instead of the checkout under test ({_PACKAGE_ROOT}). An editable install "
        f"of nemo_automodel pointing elsewhere is masking a missing module."
    )

    missing = [s for s in symbols if not hasattr(mod, s)]
    assert not missing, f"{project} imports {missing} from '{module}' ({source}), but they are gone"


@pytest.mark.parametrize(
    "project, module, symbols, source, shim",
    [row for row in _ALL_IMPORTS if row[4]],
    ids=[f"{p}-{row[0]}" for p, *row in _ALL_IMPORTS if row[3]],
)
def test_shimmed_import_matches_canonical_module(project, module, symbols, source, shim):
    """A shimmed path must yield the *same* module object as its canonical one.

    Guards against the alias silently resolving to a second, divergent copy.
    """
    assert module.startswith(_SHIM_PREFIXES), (
        f"'{module}' is flagged shim=True but does not start with a known alias prefix {_SHIM_PREFIXES}"
    )
    canonical = _import(module).__name__
    assert canonical != module, f"'{module}' was expected to be an alias, but resolved to itself"
    assert _import(module) is importlib.import_module(canonical)


def test_shim_flags_are_accurate():
    """``shim`` must be True exactly for paths under an alias prefix.

    Keeps the table honest: when a project migrates off a relocated path the
    flag flips, and once no ``shim=True`` row remains this whole file can go.
    """
    wrong = [
        (project, module, shim)
        for project, module, _symbols, _source, shim in _ALL_IMPORTS
        if shim != module.startswith(_SHIM_PREFIXES)
    ]
    assert not wrong, f"shim flag disagrees with the alias prefixes {_SHIM_PREFIXES}: {wrong}"
