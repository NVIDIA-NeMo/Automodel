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

"""Hugging Face config loading through Transformers' native revision resolution."""

import os
import re
from typing import Any

from huggingface_hub import ResolvedRevision, hf_hub_download
from transformers import AutoConfig, PretrainedConfig
from transformers.utils import CONFIG_NAME
from transformers.utils.hub import extract_commit_hash, resolve_revision


class NeMoAutoConfig(AutoConfig):
    """Load configs from one snapshot even when native revision resolution fails.

    For a config followed by a separate model or tokenizer load, resolve the
    revision once with ``transformers.utils.hub.resolve_revision`` and pass it
    to every load as ``revision``. Config objects do not carry Hub loading state.
    """

    @staticmethod
    def _pin_revision(pretrained_model_name_or_path: str | os.PathLike, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Resolve normally, then fall back to the downloaded snapshot for issue #3975."""
        if os.path.exists(pretrained_model_name_or_path):
            return kwargs
        kwargs = kwargs.copy()
        token = kwargs.get("token", kwargs.get("use_auth_token"))
        revision = resolve_revision(
            pretrained_model_name_or_path,
            kwargs.get("revision"),
            cache_dir=kwargs.get("cache_dir"),
            token=token,
            local_files_only=kwargs.get("local_files_only", False),
        )
        commit_hash = revision.resolved if isinstance(revision, ResolvedRevision) else revision
        if not isinstance(commit_hash, str) or re.fullmatch(r"[0-9a-f]{40}", commit_hash) is None:
            # Resolution is best-effort. If it fails, cached_file still re-reads
            # refs/<branch> after downloading, racing other ranks' ref writes.
            # Pin the actual returned snapshot before delegating to Transformers.
            requested_revision = revision.initial if isinstance(revision, ResolvedRevision) else revision
            config_file = hf_hub_download(
                os.fspath(pretrained_model_name_or_path),
                CONFIG_NAME,
                revision=requested_revision,
                cache_dir=kwargs.get("cache_dir"),
                subfolder=kwargs.get("subfolder", ""),
                token=token,
                local_files_only=kwargs.get("local_files_only", False),
                force_download=kwargs.get("force_download", False),
            )
            commit_hash = extract_commit_hash(config_file, None)
            if commit_hash is None:
                raise ValueError(f"Could not resolve the Hub snapshot for {pretrained_model_name_or_path!r}")
            revision = ResolvedRevision(
                resolved=commit_hash, initial=requested_revision, repo_id=os.fspath(pretrained_model_name_or_path)
            )
        kwargs["revision"] = revision
        return kwargs

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path: str | os.PathLike, **kwargs: Any
    ) -> PretrainedConfig | tuple[PretrainedConfig, dict[str, Any]]:
        """Load a config without re-reading a concurrently updated Hub branch ref.

        Args:
            pretrained_model_name_or_path: Hub repository, local directory, or config file.
            **kwargs: Transformers loading options and config overrides.

        Returns:
            The config, or the config and unused options when requested.
        """
        kwargs = cls._pin_revision(pretrained_model_name_or_path, kwargs)
        return super().from_pretrained(pretrained_model_name_or_path, **kwargs)

    @classmethod
    def get_config_dict(
        cls, pretrained_model_name_or_path: str | os.PathLike, **kwargs: Any
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Read the config dictionary for Automodel's custom-config registry.

        Args:
            pretrained_model_name_or_path: Hub repository, local directory, or config file.
            **kwargs: Transformers loading options and config overrides.

        Returns:
            The config dictionary and unused loading options.
        """
        kwargs = cls._pin_revision(pretrained_model_name_or_path, kwargs)
        return PretrainedConfig.get_config_dict(pretrained_model_name_or_path, **kwargs)
