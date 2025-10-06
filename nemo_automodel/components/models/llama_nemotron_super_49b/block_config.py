import dataclasses
import json
import warnings
from dataclasses import dataclass, MISSING
from functools import partial
from typing import Optional, Any


@partial(dataclass, frozen=True, kw_only=True)
class JsonComparable:
    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self))

    def __eq__(self, other: "JsonComparable") -> bool:
        return self.to_json() == other.to_json()

    def __hash__(self) -> int:
        return hash(self.to_json())

    def __lt__(self, other: "JsonComparable") -> bool:
        return self.to_json() < other.to_json()


@partial(dataclass, frozen=True, kw_only=True)
class SubblockConfig(JsonComparable):
    no_op: bool = False
    replace_with_linear: bool = False
    sparsify: Optional[list[str]] = None

    def __post_init__(self):
        assert not (self.no_op and self.replace_with_linear)

    def _force_setattr(self, name: str, value: Any) -> None:
        """
        Set an attribute even in frozen dataclasses.
        Use only inside __post_init__!
        """
        object.__setattr__(self, name, value)


@partial(dataclass, frozen=True, kw_only=True)
class AttentionConfig(SubblockConfig):
    n_heads_in_group: Optional[int] = None
    window_length: Optional[int] = None
    num_sink_tokens: Optional[int] = None
    use_prefill_window_in_sink_attention: bool = False
    unshifted_sink: bool = False

    def __post_init__(self):
        super().__post_init__()
        assert not (self.no_op and self.replace_with_linear)

        if self.no_op or self.replace_with_linear:
            for irrelevant_att in ["n_heads_in_group", "window_length", "num_sink_tokens"]:
                self._force_setattr(irrelevant_att, None)
        else:
            assert self.n_heads_in_group is not None

        if self.is_sink:
            assert not (self.unshifted_sink and self.use_prefill_window_in_sink_attention), \
                ("Unshifted sink uses its own kind of explicit masking, not standard window. "
                 "Set use_prefill_window_in_sink_attention to False.")
            assert not (self.num_sink_tokens == 0 and not self.unshifted_sink), \
                "Fake sink attention with 0 sink tokens is only supported with unshifted_sink=True"

    @property
    def prefill_sliding_window(self) -> Optional[int]:
        if self.window_length is not None:
            if not self.is_sink or self.use_prefill_window_in_sink_attention:
                return self.window_length
        return None

    @property
    def is_sliding(self) -> bool:
        return self.prefill_sliding_window is not None

    @property
    def is_sink(self) -> bool:
        return (
                (self.window_length is not None)
                and
                (self.num_sink_tokens is not None)
        )


@partial(dataclass, frozen=True, kw_only=True)
class FFNConfig(SubblockConfig):
    ffn_mult: Optional[float] = None

    def __post_init__(self):
        super().__post_init__()
        if self.no_op or self.replace_with_linear:
            self._force_setattr("ffn_mult", None)
        else:
            assert self.ffn_mult is not None
            self._force_setattr("ffn_mult", round(self.ffn_mult, 6))


@partial(dataclass, frozen=True, kw_only=True)
class BlockConfig(JsonComparable):
    attention: AttentionConfig = MISSING
    ffn: FFNConfig = MISSING

    def __post_init__(self):
        """
        Init subblock dataclasses from dicts
        """
        for subblock_name in dataclasses.fields(self):
            subblock_config = getattr(self, subblock_name.name)
            if isinstance(subblock_config, dict):
                subblock_fields = [field.name for field in dataclasses.fields(subblock_name.type)]
                unsupported_fields = [field_name for field_name in subblock_config.keys()
                                      if field_name not in subblock_fields]
                if len(unsupported_fields) > 0:
                    warnings.warn(f"Removed unsupported fields {unsupported_fields} from {subblock_name.type.__name__}")
                subblock_config = {k: v for k, v in subblock_config.items() if k not in unsupported_fields}
                object.__setattr__(self, subblock_name.name,
                                   subblock_name.type(**subblock_config))  # __setattr__ to overcome frozen=True
