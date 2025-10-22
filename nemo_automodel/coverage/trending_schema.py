from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, ConfigDict, constr


HydraConfig = Dict[str, Any]


class SupportEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")  # reject unknown keys
    is_supported: Optional[bool] = Field(
        ...,
        description="Support status: true=validated success, false=known failure (non-OOM), null=unknown/untested or OOM"
    )
    validated_configs: List[HydraConfig] = Field(default_factory=list)


class SupportMatrix(BaseModel):
    model_config = ConfigDict(extra="forbid")
    fsdp: SupportEntry
    tp:   SupportEntry
    pp:   SupportEntry
    cp:   SupportEntry
    ep:   SupportEntry


class ModelEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model_id: str
    sft: SupportMatrix
    peft: SupportMatrix


class TrendingModelsFile(BaseModel):
    """
    Canonical schema for nemo_automodel/coverage/trending_models.json
    """
    model_config = ConfigDict(extra="forbid")  # keep top-level strict

    spec_version: str = Field(
        default="0.1.0",
        description="Schema version for this report format"
    )
    # YYYYMMDD
    trendinglist_date: constr(pattern=r"^\d{8}$") = Field(
        description="Date the trending list was generated, format YYYYMMDD"
    )
    # 40-hex git SHA
    automodel_commit: Optional[constr(pattern=r"^[0-9a-f]{40}$")] = Field(
        None,
        description="Automodel commit SHA used when generating this list (null if not available)"
    )

    models: List[ModelEntry]
