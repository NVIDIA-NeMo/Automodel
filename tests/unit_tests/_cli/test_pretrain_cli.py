import argparse

import nemo_automodel._cli.app as app


def test_cli_accepts_pretrain(tmp_path, monkeypatch):
    parser = app.build_parser()
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("dummy: 1")
    args = parser.parse_args(["pretrain", "llm", "-c", str(cfg)])
    assert args.command == "pretrain"
    assert args.domain == "llm"
    assert args.config == cfg 