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
SPRITE_CELL_SIZE = 64
DISPLAY_CELL_REM = 1.25
DATA_URI_PATTERN = re.compile(r'url\("data:image/png;base64,([A-Za-z0-9+/=]+)"\)')


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
    sprite.save(output, format="PNG", optimize=True)
    return output.getvalue()


def _provider_selector(provider: str) -> str:
    return f'a.fern-sidebar-link[href*="/model-coverage/"][href$="/{provider}"]'


def _render_css(sprite: bytes) -> str:
    providers = sorted(PROVIDER_ORGS)
    rows = math.ceil(len(providers) / SPRITE_COLUMNS)
    encoded = base64.b64encode(sprite).decode("ascii")
    selectors = ",\n".join(f"{_provider_selector(provider)}::before" for provider in providers)
    rules = []
    for index, provider in enumerate(providers):
        column = index % SPRITE_COLUMNS
        row = index // SPRITE_COLUMNS
        rules.append(
            f"{_provider_selector(provider)} {{\n"
            f"  --provider-icon-x: {-column * DISPLAY_CELL_REM:g}rem;\n"
            f"  --provider-icon-y: {-row * DISPLAY_CELL_REM:g}rem;\n"
            f"}}"
        )
    width = SPRITE_COLUMNS * DISPLAY_CELL_REM
    height = rows * DISPLAY_CELL_REM
    return (
        "/* Provider logo sprite. */\n"
        f"{selectors} {{\n"
        '  content: "";\n'
        f"  width: {DISPLAY_CELL_REM:g}rem;\n"
        f"  height: {DISPLAY_CELL_REM:g}rem;\n"
        f"  flex: 0 0 {DISPLAY_CELL_REM:g}rem;\n"
        f'  background-image: url("data:image/png;base64,{encoded}");\n'
        "  background-position: var(--provider-icon-x) var(--provider-icon-y);\n"
        "  background-repeat: no-repeat;\n"
        f"  background-size: {width:g}rem {height:g}rem;\n"
        "}\n\n" + "\n\n".join(rules) + "\n"
    )


def _check(output_css: Path, legacy_dir: Path) -> list[str]:
    problems = []
    if not output_css.is_file():
        return [f"missing {output_css}"]
    stylesheet = output_css.read_text(encoding="ascii")
    matches = DATA_URI_PATTERN.findall(stylesheet)
    if len(matches) != 1:
        problems.append(f"expected one embedded PNG sprite in {output_css}, found {len(matches)}")
    else:
        try:
            sprite = base64.b64decode(matches[0], validate=True)
            with Image.open(io.BytesIO(sprite)) as image:
                expected_rows = math.ceil(len(PROVIDER_ORGS) / SPRITE_COLUMNS)
                expected_size = (SPRITE_COLUMNS * SPRITE_CELL_SIZE, expected_rows * SPRITE_CELL_SIZE)
                if image.format != "PNG" or image.size != expected_size:
                    problems.append(
                        f"invalid sprite in {output_css}: expected PNG {expected_size}, got {image.format} {image.size}"
                    )
        except (ValueError, OSError) as exc:
            problems.append(f"invalid base64 sprite in {output_css}: {exc}")
    for provider in PROVIDER_ORGS:
        selector = _provider_selector(provider)
        if selector not in stylesheet:
            problems.append(f"missing provider selector {provider!r} in {output_css}")
    legacy_icons = sorted(legacy_dir.glob("*.png")) if legacy_dir.is_dir() else []
    if legacy_icons:
        problems.append(f"legacy provider icons remain in {legacy_dir}")
    return problems


def main() -> int:
    """Fetch or validate the checked-in provider icon set."""
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-css",
        type=Path,
        default=repo_root / "docs" / "fern" / "provider-icons.css",
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        help="Build from existing <provider>.png files instead of fetching avatars",
    )
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    legacy_dir = repo_root / "docs" / "fern" / "assets" / "providers"

    if args.check:
        problems = _check(args.output_css, legacy_dir)
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

    stylesheet = _render_css(_build_sprite(images))
    args.output_css.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="ascii",
        dir=args.output_css.parent,
        delete=False,
    ) as temporary:
        temporary.write(stylesheet)
        temporary_path = Path(temporary.name)
    temporary_path.replace(args.output_css)
    print(f"Updated {args.output_css.relative_to(repo_root)} with {len(PROVIDER_ORGS)} providers")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
