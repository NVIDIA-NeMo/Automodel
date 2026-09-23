#!/usr/bin/env python3
"""Fetch provider avatars used by the Fern model-coverage navigation."""

from __future__ import annotations

import argparse
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


def _check(output_dir: Path) -> list[str]:
    problems = []
    for provider in PROVIDER_ORGS:
        path = output_dir / f"{provider}.png"
        if not path.is_file():
            problems.append(f"missing {path}")
            continue
        image = path.read_bytes()
        if not image.startswith(b"\x89PNG\r\n\x1a\n"):
            problems.append(f"invalid PNG {path}")
    return problems


def main() -> int:
    """Fetch or validate the checked-in provider icon set."""
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "docs" / "fern" / "assets" / "providers",
    )
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    if args.check:
        problems = _check(args.output_dir)
        if problems:
            print("\n".join(problems), file=sys.stderr)
            return 1
        print(f"Checked {len(PROVIDER_ORGS)} provider icons")
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for provider, org in PROVIDER_ORGS.items():
        destination = args.output_dir / f"{provider}.png"
        image = _fetch_avatar(org)
        with tempfile.NamedTemporaryFile(dir=args.output_dir, delete=False) as temporary:
            temporary.write(image)
            temporary_path = Path(temporary.name)
        temporary_path.replace(destination)
        print(f"Updated {destination.relative_to(repo_root)} from {org}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
