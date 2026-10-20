# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native Muse Spark Open Omni model support."""

from nemo_automodel.components.models.muse_spark.config import MuseSpark1xOmniConfig
from nemo_automodel.components.models.muse_spark.processing import (
    MuseSparkProcessor,
    build_muse_spark_processor,
)

__all__ = ["MuseSpark1xOmniConfig", "MuseSparkProcessor", "build_muse_spark_processor"]
