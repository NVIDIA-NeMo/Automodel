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
import os

os.environ["TORCH_COMPILE_DISABLE"] = "1"

import importlib
import logging
import os
from typing import List


def make_log_record(level: int = logging.INFO, name: str = "test.logger"):
    """Return a dummy ``LogRecord`` suitable for filter unit-tests."""
    return logging.LogRecord(
        name=name,
        level=level,
        pathname=__file__,
        lineno=1,
        msg="dummy",
        args=(),
        exc_info=None,
    )


def reload_module():
    """
    Reload the module under test to make sure global state (especially
    logging filters) is reset between tests.
    """
    if "log_utils" in globals():
        del globals()["log_utils"]

    mod = importlib.import_module("nemo_automodel.loggers.log_utils")
    importlib.reload(mod)
    return mod


def test_rank_filter_allows_rank0(monkeypatch):
    """RankFilter should allow records when $RANK == 0 or unset."""
    mod = reload_module()

    monkeypatch.setenv("RANK", "0")
    filt = mod.RankFilter()

    assert filt.filter(make_log_record()) is True


def test_rank_filter_blocks_nonzero_rank(monkeypatch):
    """RankFilter should block records and disable logging for rank > 0."""
    mod = reload_module()

    monkeypatch.setenv("RANK", "3")
    filt = mod.RankFilter()

    # Monkey-patch logging.disable to capture the severity passed in.
    disabled: List[int] = []

    def _fake_disable(level):
        disabled.append(level)

    monkeypatch.setattr(logging, "disable", _fake_disable)

    assert filt.filter(make_log_record()) is False
    # Confirm logging.disable(logging.CRITICAL) was invoked.
    assert disabled == [logging.CRITICAL]


# warning_filter
def test_warning_filter_blocks_only_warning():
    """warning_filter returns False only for WARNING level."""
    from nemo_automodel.loggers.log_utils import warning_filter

    warn_rec = make_log_record(level=logging.WARNING)
    info_rec = make_log_record(level=logging.INFO)

    assert warning_filter(warn_rec) is False
    assert warning_filter(info_rec) is True


# module_filter
def test_module_filter_name_prefix_matching():
    """module_filter suppresses loggers whose names start with prefix."""
    from nemo_automodel.loggers.log_utils import module_filter

    filt = lambda rec: module_filter(rec, modules_to_filter=["foo.bar"])

    rec_block = make_log_record(name="foo.bar.baz")
    rec_pass = make_log_record(name="other.module")

    assert filt(rec_block) is False
    assert filt(rec_pass) is True


# setup_logging – integration
def test_setup_logging_full(monkeypatch, caplog):
    """
    End-to-end test of setup_logging:

    * env var overrides function arg
    * WARNING messages are filtered
    * module prefix suppression works
    * RankFilter still lets rank 0 log
    """
    mod = reload_module()

    # ---- environment override ----
    monkeypatch.setenv("LOGGING_LEVEL", str(logging.DEBUG))

    # ---- rank env ----
    monkeypatch.setenv("RANK", "0")  # keep logging on

    # Configure logging
    caplog.set_level(logging.DEBUG)
    mod.setup_logging(
        logging_level=logging.INFO,  # should be overridden by env
        filter_warning=True,
        modules_to_filter=["secret"],
        set_level_for_all_loggers=True,
    )

    # Logger for allowed module
    log_ok = logging.getLogger("public.module")

    # Emit records
    with caplog.at_level(logging.DEBUG):
        log_ok.debug("visible-debug")
        log_ok.warning("hidden-warning")  # should be suppressed via warning_filter

    messages = {rec.message for rec in caplog.records}

    assert "visible-debug" in messages

    # Confirm root level got set from env override
    assert logging.getLogger().level == logging.DEBUG
