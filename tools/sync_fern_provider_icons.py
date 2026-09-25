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

#!/usr/bin/env python3
"""Build compact embedded provider logos used by Fern navigation."""

from __future__ import annotations

import argparse
import base64
import io
import re
import sys
import tempfile
import urllib.request
from pathlib import Path

from PIL import Image

PROVIDER_ORGS = {
    "allenai": "allenai",
    "baai": "BAAI",
    "baichuan-inc": "baichuan-inc",
    "baidu": "baidu",
    "bigcode": "bigcode",
    "black-forest-labs": "black-forest-labs",
    "bytedance-seed": "ByteDance-Seed",
    "cohere": "CohereForAI",
    "coherelabs": "CohereLabs",
    "deepseek-ai": "deepseek-ai",
    "diffusers": "diffusers",
    "eleutherai": "EleutherAI",
    "google": "google",
    "google-t5": "google-t5",
    "gsai-ml": "GSAI-ML",
    "huggingfacetb": "HuggingFaceTB",
    "hunyuanvideo-community": "hunyuanvideo-community",
    "ibm": "ibm-granite",
    "ibm-ai-platform": "ibm-ai-platform",
    "inceptionai": "inceptionai",
    "inclusionai": "inclusionAI",
    "internlm": "internlm",
    "lgai-exaone": "LGAI-EXAONE",
    "lightricks": "Lightricks",
    "llava-hf": "llava-hf",
    "lmms-lab": "lmms-lab",
    "meta": "meta-llama",
    "microsoft": "microsoft",
    "minimax": "MiniMaxAI",
    "mistralai": "mistralai",
    "moonshotai": "moonshotai",
    "meta-models": "meta-models",
    "nvidia": "nvidia",
    "openai": "openai",
    "openai-community": "openai-community",
    "openbmb": "openbmb",
    "opengvlab": "OpenGVLab",
    "orionstar": "OrionStarAI",
    "parasail-ai": "parasail-ai",
    "poolside": "poolside",
    "qwen": "Qwen",
    "stabilityai": "stabilityai",
    "stepfun-ai": "stepfun-ai",
    "tencent": "tencent",
    "thinkingmachines": "thinkingmachines",
    "thudm": "zai-org",
    "tiiuae": "tiiuae",
    "upstage": "upstage",
    "wan-ai": "Wan-AI",
    "xiaomimimo": "XiaomiMiMo",
    "z-lab": "z-lab",
}

AVATAR_PATTERN = re.compile(rb"cdn-avatars\.huggingface\.co/[^\"&]+")
USER_AGENT = "nemo-automodel-docs/1.0"
ICON_SIZE = 20
MAX_TOTAL_ENCODED_BYTES = 256 * 1024
DATA_URI_PATTERN = re.compile(r'href="data:image/webp;base64,([A-Za-z0-9+/=]+)"')


def _request(url: str, *, accept: str) -> bytes:
    request = urllib.request.Request(
        url,
        headers={"Accept": accept, "User-Agent": USER_AGENT},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read()


def _fetch_avatar(org: str) -> bytes:
    profile = _request(f"https://huggingface.co/{org}", accept="text/html")
    match = AVATAR_PATTERN.search(profile)
    if match is None:
        raise RuntimeError(f"No Hugging Face avatar found for {org}")
    avatar_url = f"https://{match.group().decode('ascii')}"
    source = _request(avatar_url, accept="image/webp")
    if not (source.startswith(b"RIFF") and source[8:12] == b"WEBP"):
        raise RuntimeError(f"Hugging Face did not return WebP for {org}")

    return source


def _build_icons(images: dict[str, bytes]) -> dict[str, bytes]:
    icons = {}
    for provider in sorted(PROVIDER_ORGS):
        with Image.open(io.BytesIO(images[provider])) as source:
            logo = source.convert("RGBA")
            logo.thumbnail((ICON_SIZE, ICON_SIZE), Image.Resampling.LANCZOS)
        icon = Image.new("RGBA", (ICON_SIZE, ICON_SIZE), (0, 0, 0, 0))
        x = (ICON_SIZE - logo.width) // 2
        y = (ICON_SIZE - logo.height) // 2
        icon.alpha_composite(logo, (x, y))
        output = io.BytesIO()
        icon.save(output, format="WEBP", lossless=True, method=6)
        icons[provider] = output.getvalue()
    return icons


def _render_icon(encoded: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {ICON_SIZE} {ICON_SIZE}">'
        f'<image width="{ICON_SIZE}" height="{ICON_SIZE}" '
        f'href="data:image/webp;base64,{encoded}"/></svg>'
    )


def _navigation_icons(navigation_path: Path) -> list[tuple[str, str]]:
    lines = navigation_path.read_text(encoding="utf-8").splitlines()
    icons = []
    for index, line in enumerate(lines[:-1]):
        stripped = line.strip()
        next_line = lines[index + 1].strip()
        if not stripped.startswith("icon: ") or not next_line.startswith("slug: "):
            continue
        provider = next_line.removeprefix("slug: ").strip('"')
        if provider in PROVIDER_ORGS:
            icons.append((provider, stripped.removeprefix("icon: ").strip("'")))
    return icons


def _update_navigation(navigation_path: Path, icons: dict[str, bytes]) -> int:
    lines = navigation_path.read_text(encoding="utf-8").splitlines()
    updated = 0
    for index, line in enumerate(lines[:-1]):
        next_line = lines[index + 1].strip()
        if not line.strip().startswith("icon: ") or not next_line.startswith("slug: "):
            continue
        provider = next_line.removeprefix("slug: ").strip('"')
        if provider not in PROVIDER_ORGS:
            continue
        indent = line[: len(line) - len(line.lstrip())]
        encoded = base64.b64encode(icons[provider]).decode("ascii")
        lines[index] = f"{indent}icon: '{_render_icon(encoded)}'"
        updated += 1
    if updated == 0:
        raise RuntimeError(f"No provider icon entries found in {navigation_path}")
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=navigation_path.parent,
        delete=False,
    ) as temporary:
        temporary.write("\n".join(lines) + "\n")
        temporary_path = Path(temporary.name)
    temporary_path.replace(navigation_path)
    return updated


def _check(navigation_path: Path, legacy_dir: Path, legacy_sprite: Path) -> list[str]:
    problems = []
    icons = _navigation_icons(navigation_path)
    providers = {provider for provider, _ in icons}
    if providers != set(PROVIDER_ORGS):
        missing = sorted(set(PROVIDER_ORGS) - providers)
        problems.append(f"provider icons missing from {navigation_path}: {missing}")
    total_encoded_bytes = 0
    for provider, icon in icons:
        match = DATA_URI_PATTERN.search(icon)
        if match is None:
            problems.append(f"{provider}: icon does not embed a WebP provider logo")
            continue
        encoded = match.group(1)
        total_encoded_bytes += len(encoded)
        if f'viewBox="0 0 {ICON_SIZE} {ICON_SIZE}"' not in icon:
            problems.append(f"{provider}: incorrect icon view box")
        try:
            provider_icon = base64.b64decode(encoded, validate=True)
            with Image.open(io.BytesIO(provider_icon)) as image:
                expected_size = (ICON_SIZE, ICON_SIZE)
                if image.format != "WEBP" or image.size != expected_size:
                    problems.append(f"{provider}: expected WEBP {expected_size}, got {image.format} {image.size}")
        except (ValueError, OSError) as exc:
            problems.append(f"{provider}: invalid base64 WebP: {exc}")
    if total_encoded_bytes > MAX_TOTAL_ENCODED_BYTES:
        problems.append(
            f"provider icon payload is {total_encoded_bytes} bytes; expected at most {MAX_TOTAL_ENCODED_BYTES}"
        )
    legacy_icons = sorted(legacy_dir.glob("*.png")) if legacy_dir.is_dir() else []
    if legacy_icons:
        problems.append(f"legacy provider icons remain in {legacy_dir}")
    if legacy_sprite.exists():
        problems.append(f"legacy external sprite remains at {legacy_sprite}")
    return problems


def main() -> int:
    """Fetch or validate the checked-in provider icon set."""
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--navigation",
        type=Path,
        default=repo_root / "docs" / "fern" / "versions" / "nightly.yml",
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        help="Build from existing <provider>.png files instead of fetching avatars",
    )
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    legacy_dir = repo_root / "docs" / "fern" / "assets" / "providers"
    legacy_sprite = repo_root / "docs" / "fern" / "assets" / "provider-sprite.svg"

    if args.check:
        problems = _check(args.navigation, legacy_dir, legacy_sprite)
        if problems:
            print("\n".join(problems), file=sys.stderr)
            return 1
        print(f"Checked {len(PROVIDER_ORGS)} compact embedded provider icons")
        return 0

    images = {}
    for provider, org in PROVIDER_ORGS.items():
        if args.source_dir is None:
            images[provider] = _fetch_avatar(org)
        else:
            source = args.source_dir / f"{provider}.png"
            if not source.is_file():
                raise FileNotFoundError(source)
            images[provider] = source.read_bytes()

    updated = _update_navigation(args.navigation, _build_icons(images))
    print(f"Updated {updated} provider icons in {args.navigation.relative_to(repo_root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
