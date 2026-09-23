#!/usr/bin/env python3
"""Build the embedded provider-logo sprite used by Fern navigation."""

from __future__ import annotations

import argparse
import base64
import io
import math
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
    "huggingface": "huggingface",
    "hunyuanvideo-community": "hunyuanvideo-community",
    "ibm": "ibm-granite",
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
    "muse": "meta-models",
    "nvidia": "nvidia",
    "openai": "openai",
    "openbmb": "openbmb",
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
SPRITE_COLUMNS = 12
SPRITE_CELL_SIZE = 20
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

    output = io.BytesIO()
    with Image.open(io.BytesIO(source)) as image:
        image.seek(0)
        image.save(output, format="PNG", optimize=True)
    return output.getvalue()


def _build_sprite(images: dict[str, bytes]) -> bytes:
    providers = sorted(PROVIDER_ORGS)
    rows = math.ceil(len(providers) / SPRITE_COLUMNS)
    sprite = Image.new(
        "RGBA",
        (SPRITE_COLUMNS * SPRITE_CELL_SIZE, rows * SPRITE_CELL_SIZE),
        (0, 0, 0, 0),
    )
    for index, provider in enumerate(providers):
        with Image.open(io.BytesIO(images[provider])) as source:
            logo = source.convert("RGBA")
            logo.thumbnail((SPRITE_CELL_SIZE, SPRITE_CELL_SIZE), Image.Resampling.LANCZOS)
        column = index % SPRITE_COLUMNS
        row = index // SPRITE_COLUMNS
        x = column * SPRITE_CELL_SIZE + (SPRITE_CELL_SIZE - logo.width) // 2
        y = row * SPRITE_CELL_SIZE + (SPRITE_CELL_SIZE - logo.height) // 2
        sprite.alpha_composite(logo, (x, y))

    output = io.BytesIO()
    sprite.save(output, format="WEBP", lossless=True, method=6)
    return output.getvalue()


def _render_icon(provider: str, encoded: str) -> str:
    providers = sorted(PROVIDER_ORGS)
    index = providers.index(provider)
    column = index % SPRITE_COLUMNS
    row = index // SPRITE_COLUMNS
    width = SPRITE_COLUMNS * SPRITE_CELL_SIZE
    height = math.ceil(len(providers) / SPRITE_COLUMNS) * SPRITE_CELL_SIZE
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="{column * SPRITE_CELL_SIZE} {row * SPRITE_CELL_SIZE} '
        f'{SPRITE_CELL_SIZE} {SPRITE_CELL_SIZE}">'
        f'<image width="{width}" height="{height}" '
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


def _update_navigation(navigation_path: Path, sprite: bytes) -> int:
    lines = navigation_path.read_text(encoding="utf-8").splitlines()
    encoded = base64.b64encode(sprite).decode("ascii")
    updated = 0
    for index, line in enumerate(lines[:-1]):
        next_line = lines[index + 1].strip()
        if not line.strip().startswith("icon: ") or not next_line.startswith("slug: "):
            continue
        provider = next_line.removeprefix("slug: ").strip('"')
        if provider not in PROVIDER_ORGS:
            continue
        indent = line[: len(line) - len(line.lstrip())]
        lines[index] = f"{indent}icon: '{_render_icon(provider, encoded)}'"
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
    encoded_sprites = set()
    for provider, icon in icons:
        match = DATA_URI_PATTERN.search(icon)
        if match is None:
            problems.append(f"{provider}: icon does not embed the provider sprite")
            continue
        encoded_sprites.add(match.group(1))
        expected_view_box = _render_icon(provider, match.group(1)).split('viewBox="', 1)[1].split('"', 1)[0]
        if f'viewBox="{expected_view_box}"' not in icon:
            problems.append(f"{provider}: incorrect sprite view box")
    if len(encoded_sprites) != 1:
        problems.append(f"expected one shared base64 sprite, found {len(encoded_sprites)}")
    else:
        try:
            sprite = base64.b64decode(next(iter(encoded_sprites)), validate=True)
            with Image.open(io.BytesIO(sprite)) as image:
                expected_rows = math.ceil(len(PROVIDER_ORGS) / SPRITE_COLUMNS)
                expected_size = (SPRITE_COLUMNS * SPRITE_CELL_SIZE, expected_rows * SPRITE_CELL_SIZE)
                if image.format != "WEBP" or image.size != expected_size:
                    problems.append(
                        f"invalid embedded sprite: expected WEBP {expected_size}, got {image.format} {image.size}"
                    )
        except (ValueError, OSError) as exc:
            problems.append(f"invalid base64 sprite: {exc}")
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
        print(f"Checked one embedded sprite for {len(PROVIDER_ORGS)} providers")
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

    updated = _update_navigation(args.navigation, _build_sprite(images))
    print(f"Updated {updated} provider icons in {args.navigation.relative_to(repo_root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
