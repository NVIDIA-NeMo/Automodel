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

"""Widen transformers' ``ALLOWED_LAYER_TYPES`` so legacy custom configs load.

Some community models (e.g. ``nvidia/Nemotron-Flash-1B``) ship a custom
``configuration_*.py`` whose ``layer_types`` entries (e.g. ``'deltanet'``,
``'f'``, ``'m2'``, ``'a'``) are not in the upstream allow-list. Loading them
via ``AutoConfig.from_pretrained`` triggers ``validate_layer_type`` and raises
``StrictDataclassClassValidationError`` before model instantiation.

The validator performs a module-global lookup of ``ALLOWED_LAYER_TYPES`` at
call time, so rebinding it in place takes effect on subsequent validations.
"""

from __future__ import annotations

import importlib.abc
import logging
import sys
from typing import Iterable

logger = logging.getLogger(__name__)

_TARGET_MODULE = "transformers.configuration_utils"

DEFAULT_EXTRA_LAYER_TYPES: tuple[str, ...] = (
    "deltanet",
    "f",
    "m2",
    "a",
)

_PATCHED: bool = False


def patch_allowed_layer_types(extra: Iterable[str] = DEFAULT_EXTRA_LAYER_TYPES) -> bool:
    """Extend ``transformers.configuration_utils.ALLOWED_LAYER_TYPES`` in place.

    Idempotent and best-effort: any failure (missing attribute, transformers
    not installed, unexpected container type) is logged and swallowed so the
    caller's import path is not broken.

    Args:
        extra: Layer-type names to append if not already present.

    Returns:
        ``True`` if the tuple was modified on this call, ``False`` otherwise.
    """
    global _PATCHED
    if _PATCHED:
        return False

    try:
        from transformers import configuration_utils as cu
    except ImportError:
        logger.debug("[v4_patches.layer_types] transformers not importable; skipping.")
        return False
    except Exception as exc:
        logger.warning("[v4_patches.layer_types] transformers import failed: %s", exc)
        return False

    existing = getattr(cu, "ALLOWED_LAYER_TYPES", None)
    if existing is None:
        logger.debug("[v4_patches.layer_types] ALLOWED_LAYER_TYPES missing; nothing to patch.")
        _PATCHED = True
        return False

    try:
        existing_set = set(existing)
    except TypeError:
        logger.warning(
            "[v4_patches.layer_types] ALLOWED_LAYER_TYPES is not iterable (%s); skipping.",
            type(existing).__name__,
        )
        return False

    try:
        additions = tuple(lt for lt in extra if lt not in existing_set)
    except TypeError:
        logger.warning("[v4_patches.layer_types] `extra` is not iterable; skipping.")
        return False

    if not additions:
        _PATCHED = True
        return False

    try:
        cu.ALLOWED_LAYER_TYPES = tuple(existing) + additions
    except Exception as exc:
        logger.warning("[v4_patches.layer_types] failed to rebind ALLOWED_LAYER_TYPES: %s", exc)
        return False

    _PATCHED = True
    logger.info("[v4_patches.layer_types] extended ALLOWED_LAYER_TYPES with %s", additions)
    return True


_HOOK_INSTALLED: bool = False


class _LayerTypesPatchFinder(importlib.abc.MetaPathFinder):
    """Meta-path finder that applies ``patch_allowed_layer_types`` the moment
    ``transformers.configuration_utils`` finishes loading.

    The finder only intercepts the single target module name. For that name it
    delegates to the remaining finders to obtain a real spec, then wraps the
    loader's ``exec_module`` so the patch runs once, post-exec. All deviations
    from the happy path (missing loader, unexpected spec shape, patch failure)
    are swallowed with a log — the finder must never break imports.
    """

    def __init__(self) -> None:
        self._applied = False

    def find_spec(self, fullname, path=None, target=None):
        if fullname != _TARGET_MODULE or self._applied:
            return None

        real_spec = None
        for finder in sys.meta_path:
            # Skip any instance of our own finder class to avoid wrapping our
            # own wrapped loader (can happen if the hook is installed twice).
            if isinstance(finder, _LayerTypesPatchFinder):
                continue
            find_spec = getattr(finder, "find_spec", None)
            if find_spec is None:
                continue
            try:
                real_spec = find_spec(fullname, path, target)
            except Exception:
                continue
            if real_spec is not None:
                break

        if real_spec is None or real_spec.loader is None:
            return real_spec

        loader = real_spec.loader
        original_exec_module = getattr(loader, "exec_module", None)
        if original_exec_module is None:
            return real_spec

        finder_self = self

        def exec_module(module):
            try:
                original_exec_module(module)
            finally:
                finder_self._applied = True
                try:
                    patch_allowed_layer_types()
                except Exception as exc:
                    logger.warning("[v4_patches.layer_types] post-import patch failed: %s", exc)

        try:
            loader.exec_module = exec_module  # type: ignore[method-assign]
        except Exception as exc:
            logger.debug("[v4_patches.layer_types] could not wrap loader.exec_module: %s", exc)

        return real_spec


def install_layer_types_patch_hook() -> bool:
    """Ensure ``patch_allowed_layer_types`` runs before any call to
    ``PreTrainedConfig.validate_layer_type``.

    Two paths:
      * If ``transformers.configuration_utils`` is already loaded, patch
        immediately.
      * Otherwise, register a meta-path finder that patches on first import.

    Idempotent; safe to call multiple times.

    Returns:
        ``True`` if a hook was installed (or the patch was applied directly on
        this call), ``False`` if a previous call already arranged one.
    """
    global _HOOK_INSTALLED
    if _HOOK_INSTALLED:
        return False

    if _TARGET_MODULE in sys.modules:
        _HOOK_INSTALLED = True
        patch_allowed_layer_types()
        return True

    try:
        sys.meta_path.insert(0, _LayerTypesPatchFinder())
    except Exception as exc:
        logger.warning("[v4_patches.layer_types] failed to install import hook: %s", exc)
        return False

    _HOOK_INSTALLED = True
    return True


__all__ = [
    "DEFAULT_EXTRA_LAYER_TYPES",
    "install_layer_types_patch_hook",
    "patch_allowed_layer_types",
]
