# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for nemo_automodel.components.datasets.vlm.utils."""

from __future__ import annotations

import io
import sys
from unittest.mock import MagicMock

import pytest
from PIL import Image

import nemo_automodel.components.datasets.vlm.utils as vlm_utils

# ---------------------------------------------------------------------------
# _resolve_lmdb_image
# ---------------------------------------------------------------------------


class TestResolveLmdbImage:
    def test_raises_when_lmdb_not_installed(self, monkeypatch):
        monkeypatch.setattr(vlm_utils, "HAVE_LMDB", False)
        with pytest.raises(ImportError, match="lmdb package is required"):
            vlm_utils._resolve_lmdb_image("/db.lmdb::key")

    @pytest.fixture(autouse=True)
    def _setup_fake_lmdb(self, monkeypatch):
        """Set up a fake lmdb module with a working env/txn mock."""
        img = Image.new("RGB", (4, 4), (128, 64, 32))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        self._img_bytes = buf.getvalue()

        self.fake_txn = MagicMock()
        self.fake_txn.get.return_value = self._img_bytes
        self.fake_env = MagicMock()
        self.fake_env.begin.return_value.__enter__ = MagicMock(return_value=self.fake_txn)
        self.fake_env.begin.return_value.__exit__ = MagicMock(return_value=False)

        self.fake_lmdb_mod = MagicMock()
        self.fake_lmdb_mod.open.return_value = self.fake_env

        monkeypatch.setattr(vlm_utils, "HAVE_LMDB", True)
        monkeypatch.setitem(sys.modules, "lmdb", self.fake_lmdb_mod)
        # Inject the fake module as the `lmdb` name in vlm_utils
        monkeypatch.setattr(vlm_utils, "lmdb", self.fake_lmdb_mod, raising=False)
        vlm_utils._lmdb_env_cache.clear()

    def test_returns_rgb_image(self):
        result = vlm_utils._resolve_lmdb_image("/data/db.lmdb::0042")
        assert isinstance(result, Image.Image)
        assert result.mode == "RGB"
        self.fake_txn.get.assert_called_once_with(b"0042")

    def test_raises_on_missing_key(self):
        self.fake_txn.get.return_value = None
        with pytest.raises(KeyError, match="not found"):
            vlm_utils._resolve_lmdb_image("/data/db.lmdb::missing")

    def test_caches_lmdb_env(self):
        vlm_utils._resolve_lmdb_image("/db.lmdb::a")
        vlm_utils._resolve_lmdb_image("/db.lmdb::b")
        # Only one open call — second access uses cache
        self.fake_lmdb_mod.open.assert_called_once()


# ---------------------------------------------------------------------------
# _build_video_metadata
# ---------------------------------------------------------------------------


class TestBuildVideoMetadata:
    def test_empty_conversation(self):
        assert vlm_utils._build_video_metadata([]) == []

    def test_no_video_items(self):
        conv = [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
        assert vlm_utils._build_video_metadata(conv) == []

    def test_builds_metadata_from_preserved_fields(self):
        frames = [Image.new("RGB", (2, 2))] * 4
        conv = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": frames,
                        "_video_fps": 30.0,
                        "_frame_indices": [0, 10, 20, 30],
                    }
                ],
            }
        ]
        result = vlm_utils._build_video_metadata(conv)
        assert len(result) == 1
        assert result[0].total_num_frames == 4
        assert result[0].fps == 30.0
        assert result[0].frames_indices == [0, 10, 20, 30]

    def test_skips_incomplete_metadata(self):
        conv = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": [], "_video_fps": 30.0},  # missing _frame_indices
                ],
            }
        ]
        assert vlm_utils._build_video_metadata(conv) == []
