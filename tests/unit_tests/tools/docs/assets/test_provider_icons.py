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

import io
from pathlib import Path

import pytest
from PIL import Image

from tools.docs.assets import provider_icons


@pytest.fixture
def icon(monkeypatch) -> bytes:
    monkeypatch.setattr(provider_icons, "PROVIDER_ORGS", {"example": "Example"})
    source = io.BytesIO()
    Image.new("RGBA", (80, 40), "red").save(source, format="PNG")
    return provider_icons._build_icons({"example": source.getvalue()})["example"]


def test_provider_icon_is_a_centered_transparent_webp(icon: bytes):
    with Image.open(io.BytesIO(icon)) as image:
        assert image.format == "WEBP"
        assert image.size == (20, 20)
        assert image.getpixel((0, 0))[3] == 0
        assert image.getpixel((10, 10)) == (255, 0, 0, 255)


def test_provider_navigation_update_preserves_other_content(tmp_path: Path, icon: bytes):
    navigation = tmp_path / "nightly.yml"
    navigation.write_text("navigation:\n  - section: Example\n    icon: old\n    slug: example\n  - page: Guide\n")

    assert provider_icons._update_navigation(navigation, {"example": icon}) == 1
    assert "  - page: Guide\n" in navigation.read_text()
    assert provider_icons._check(navigation, tmp_path / "legacy", tmp_path / "sprite.svg") == []


def test_provider_icon_check_rejects_corrupt_image_data(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(provider_icons, "PROVIDER_ORGS", {"example": "Example"})
    navigation = tmp_path / "nightly.yml"
    navigation.write_text("icon: '" + provider_icons._render_icon("bm90LWFuLWltYWdl") + "'\nslug: example\n")

    assert any(
        "invalid base64 WebP" in problem
        for problem in provider_icons._check(navigation, tmp_path / "legacy", tmp_path / "sprite.svg")
    )
