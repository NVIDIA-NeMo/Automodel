# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Distributed topology and tensor transport helpers for KD recipes."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor, Shard

from nemo_automodel.components.distributed.config import DDPConfig, DistributedSetup
from nemo_automodel.recipes._dist_utils import create_distributed_setup_from_config, parse_distributed_section

if TYPE_CHECKING:
    from nemo_automodel.components.distributed.context_parallel.sharder import CPTokenLayout

RUN_TEACHER = 1
STOP_TEACHER = 0


class _ConfigLike(Protocol):
    """Minimal recipe-config interface consumed by KD topology setup."""

    def get(self, key: str, default: Any = None) -> Any:
        """Return one configuration value or ``default``."""
        ...


def materialize_teacher_logits(
    logits: torch.Tensor,
    *,
    tokens: "CPTokenLayout",
    sequence_length: int,
) -> torch.Tensor:
    """Reconstruct full teacher logits across TP and CP before mesh transport.

    Args:
        logits: Tensor of global shape ``[batch, sequence, vocab]`` containing
            teacher logits. It may be a vocabulary-sharded ``DTensor`` with
            ``Shard(-1)`` placement and/or have a per-rank load-balanced
            ``local_sequence`` extent under CP.
        tokens: Batch-bound context-parallel token layout used for the teacher
            forward.
        sequence_length: Unpadded global sequence length to retain.

    Returns:
        Detached contiguous tensor of shape
        ``[batch, sequence_length, vocab]`` replicated on every rank in the
        teacher model replica.
    """
    if isinstance(logits, DTensor):
        logits = logits.full_tensor()
    logits = tokens.gather(logits, seq_dim=1, trim=True)
    return logits.narrow(1, 0, sequence_length).detach().contiguous()


def _section_to_dict(section: Any) -> dict:
    if section is None:
        return {}
    if isinstance(section, dict):
        return section.copy()
    if hasattr(section, "to_dict"):
        return section.to_dict()
    return dict(section)


def _mesh_size(distributed_cfg: Any, *, label: str) -> int:
    """Return the explicitly requested mesh size for a separate KD model."""
    cfg_dict = _section_to_dict(distributed_cfg)
    parsed = parse_distributed_section(cfg_dict)
    sizes = parsed["parallelism_sizes"]
    if sizes.dp_size is None:
        raise ValueError(
            f"{label}.dp_size must be set when separate_meshes=true so the non-overlapping rank split is explicit"
        )
    return sizes.dp_size * sizes.tp_size * sizes.pp_size * sizes.cp_size


@dataclass(frozen=True)
class KDDistributedSetups:
    """Student/teacher setups plus their global-rank assignments.

    Attributes:
        student: Resolved student topology and policies.
        teacher: Resolved teacher topology and policies.
        student_ranks: Ordered global ranks assigned to the student.
        teacher_ranks: Ordered global ranks assigned to the teacher.
        separate: Whether the assignments are disjoint.
    """

    student: DistributedSetup
    teacher: DistributedSetup
    student_ranks: tuple[int, ...]
    teacher_ranks: tuple[int, ...]
    separate: bool


def create_kd_distributed_setups(cfg: _ConfigLike, *, world_size: int) -> KDDistributedSetups:
    """Build shared or explicitly disjoint student and teacher setups.

    Args:
        cfg: Recipe configuration containing ``distributed`` and optional
            ``teacher_distributed`` sections.
        world_size: Total global process count available to both models.

    Returns:
        Resolved model setups and their ordered global-rank assignments.
    """
    separate = bool(cfg.get("separate_meshes", False))
    teacher_cfg = cfg.get("teacher_distributed", None)
    if not separate:
        if teacher_cfg is not None:
            raise ValueError("teacher_distributed requires separate_meshes=true")
        shared = create_distributed_setup_from_config(cfg, world_size=world_size)
        ranks = tuple(range(world_size))
        return KDDistributedSetups(shared, shared, ranks, ranks, False)

    if teacher_cfg is None:
        raise ValueError("separate_meshes=true requires a teacher_distributed section")

    student_cfg = _section_to_dict(cfg.get("distributed", None))
    teacher_cfg = _section_to_dict(teacher_cfg)
    student_size = _mesh_size(student_cfg, label="distributed")
    teacher_size = _mesh_size(teacher_cfg, label="teacher_distributed")
    if student_size + teacher_size != world_size:
        raise ValueError(
            "Separate KD mesh sizes must use every rank exactly once: "
            f"student={student_size} + teacher={teacher_size} != world_size={world_size}"
        )

    student_ranks = tuple(range(student_size))
    teacher_ranks = tuple(range(student_size, world_size))
    student = create_distributed_setup_from_config(
        student_cfg,
        world_size=student_size,
        ranks=student_ranks,
    )
    teacher = create_distributed_setup_from_config(
        teacher_cfg,
        world_size=teacher_size,
        ranks=teacher_ranks,
    )
    if isinstance(student.strategy_config, DDPConfig) or isinstance(teacher.strategy_config, DDPConfig):
        raise ValueError("Separate KD meshes currently require mesh-backed strategies; DDP is not supported")
    return KDDistributedSetups(student, teacher, student_ranks, teacher_ranks, True)


@dataclass(frozen=True)
class _Replica:
    ranks: tuple[int, ...]
    input_rank: int
    output_rank: int


@dataclass(frozen=True)
class _Route:
    src: int
    ranks: tuple[int, ...]
    group: dist.ProcessGroup


def _model_replicas(setup: DistributedSetup) -> list[_Replica]:
    mesh = setup.mesh_context.device_mesh
    if mesh is None:
        raise ValueError("Separate KD meshes require a DeviceMesh")

    names = tuple(name.value if hasattr(name, "value") else str(name) for name in mesh.mesh_dim_names)
    rank_mesh = mesh.mesh.cpu()
    dp_axes = tuple(i for i, name in enumerate(names) if name in {"dp", "dp_replicate", "dp_shard"})
    if not dp_axes:
        raise ValueError(f"KD mesh has no data-parallel axis: {names}")

    grouped: dict[tuple[int, ...], list[tuple[tuple[int, ...], int]]] = {}
    for coordinate in itertools.product(*(range(size) for size in rank_mesh.shape)):
        dp_coordinate = tuple(coordinate[i] for i in dp_axes)
        grouped.setdefault(dp_coordinate, []).append((coordinate, int(rank_mesh[coordinate].item())))

    replicas = []
    pp_axis = names.index("pp") if "pp" in names else None
    cp_axis = names.index("cp") if "cp" in names else None
    tp_axis = names.index("tp") if "tp" in names else None
    for dp_coordinate in sorted(grouped):
        entries = grouped[dp_coordinate]

        def is_input(coordinate: tuple[int, ...]) -> bool:
            return all(coordinate[axis] == 0 for axis in (pp_axis, cp_axis, tp_axis) if axis is not None)

        def is_output(coordinate: tuple[int, ...]) -> bool:
            if pp_axis is not None and coordinate[pp_axis] != rank_mesh.shape[pp_axis] - 1:
                return False
            return all(coordinate[axis] == 0 for axis in (cp_axis, tp_axis) if axis is not None)

        input_ranks = [rank for coordinate, rank in entries if is_input(coordinate)]
        output_ranks = [rank for coordinate, rank in entries if is_output(coordinate)]
        if len(input_ranks) != 1 or len(output_ranks) != 1:
            raise RuntimeError(f"Could not identify KD replica endpoints for DP coordinate {dp_coordinate}")
        replicas.append(
            _Replica(
                ranks=tuple(rank for _, rank in entries),
                input_rank=input_ranks[0],
                output_rank=output_ranks[0],
            )
        )
    return replicas


def _tree_spec(value: Any, tensors: list[torch.Tensor]) -> Any:
    """Describe a nested value while extracting tensor leaves without copies.

    Tensor leaves may have arbitrary rank and axis semantics; each leaf's exact
    shape and dtype are recorded for allocation on receiving ranks.

    Args:
        value: Nested dictionaries, lists, tuples, scalar values, and tensor
            leaves with arbitrary shape and axis order.
        tensors: Output list populated with aliases of tensor leaves in traversal
            order.

    Returns:
        Pickle-compatible nested metadata describing ``value``.
    """
    if isinstance(value, torch.Tensor):
        index = len(tensors)
        tensors.append(value)
        return ("tensor", index, tuple(value.shape), str(value.dtype).removeprefix("torch."))
    if isinstance(value, dict):
        return ("dict", [(key, _tree_spec(item, tensors)) for key, item in value.items()])
    if isinstance(value, list):
        return ("list", [_tree_spec(item, tensors) for item in value])
    if isinstance(value, tuple):
        return ("tuple", [_tree_spec(item, tensors) for item in value])
    return ("value", value)


def _tree_from_spec(spec: Any, tensors: list[torch.Tensor]) -> Any:
    """Rebuild a nested value from tensor leaves described by ``_tree_spec``.

    Tensor leaves preserve the exact shapes and dtypes encoded in ``spec`` and
    alias the corresponding entries in ``tensors``.

    Args:
        spec: Metadata returned by ``_tree_spec``.
        tensors: Tensor leaves with the exact recorded shapes and dtypes.

    Returns:
        Reconstructed nested value whose tensor leaves alias ``tensors``.
    """
    kind = spec[0]
    if kind == "tensor":
        return tensors[spec[1]]
    if kind == "dict":
        return {key: _tree_from_spec(item, tensors) for key, item in spec[1]}
    if kind == "list":
        return [_tree_from_spec(item, tensors) for item in spec[1]]
    if kind == "tuple":
        return tuple(_tree_from_spec(item, tensors) for item in spec[1])
    return spec[1]


class KDMeshBridge:
    """Move batches and teacher logits between disjoint model meshes.

    Args:
        setups: Resolved disjoint student and teacher setups.
        device: Device used for transport tensors and collectives.
    """

    def __init__(self, setups: KDDistributedSetups, *, device: torch.device) -> None:
        if not setups.separate:
            raise ValueError("KDMeshBridge is only used for separate meshes")
        self.device = device
        self.rank = dist.get_rank()
        self.student_ranks = setups.student_ranks
        self.teacher_ranks = setups.teacher_ranks
        self.control_group = dist.new_group(ranks=list(self.student_ranks + self.teacher_ranks))
        self.student_group = dist.new_group(ranks=list(self.student_ranks))
        self.teacher_group = dist.new_group(ranks=list(self.teacher_ranks))
        self.student_replicas = _model_replicas(setups.student)
        self.teacher_replicas = _model_replicas(setups.teacher)
        self.num_waves = (len(self.student_replicas) + len(self.teacher_replicas) - 1) // len(self.teacher_replicas)

        self.input_routes: list[list[_Route]] = []
        self.output_routes: list[list[_Route]] = []
        for wave in range(self.num_waves):
            wave_inputs = []
            wave_outputs = []
            for teacher_index, teacher_replica in enumerate(self.teacher_replicas):
                student_index = wave * len(self.teacher_replicas) + teacher_index
                source_index = student_index if student_index < len(self.student_replicas) else 0
                source = self.student_replicas[source_index].input_rank
                input_ranks = tuple(dict.fromkeys((source, *teacher_replica.ranks)))
                wave_inputs.append(_Route(source, input_ranks, dist.new_group(ranks=list(input_ranks))))
                if student_index < len(self.student_replicas):
                    output_ranks = tuple(
                        dict.fromkeys((teacher_replica.output_rank, *self.student_replicas[student_index].ranks))
                    )
                    wave_outputs.append(
                        _Route(
                            teacher_replica.output_rank,
                            output_ranks,
                            dist.new_group(ranks=list(output_ranks)),
                        )
                    )
            self.input_routes.append(wave_inputs)
            self.output_routes.append(wave_outputs)

    @property
    def is_student(self) -> bool:
        return self.rank in self.student_ranks

    @property
    def is_teacher(self) -> bool:
        return self.rank in self.teacher_ranks

    def move_to_device(self, value: Any) -> Any:
        """Move every tensor leaf in a nested value to the bridge device.

        Tensor leaves may have arbitrary shapes and axis order.

        Args:
            value: Nested dictionaries, lists, tuples, scalar values, and tensor
                leaves.

        Returns:
            Equivalent nested value whose tensors preserve shape and dtype on
            ``self.device``. Input containers are not mutated.
        """
        if isinstance(value, torch.Tensor):
            return value.to(self.device, non_blocking=True)
        if isinstance(value, dict):
            return {key: self.move_to_device(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self.move_to_device(item) for item in value]
        if isinstance(value, tuple):
            return tuple(self.move_to_device(item) for item in value)
        return value

    @staticmethod
    def match_student_vocab_shard(
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Match full teacher logits to a TP-sharded student vocabulary.

        Args:
            student_logits: Tensor of global shape
                ``[batch, sequence, vocab]`` containing student logits. A
                vocabulary-sharded ``DTensor`` has local shape
                ``[batch, sequence, local_vocab]`` and ``Shard(-1)`` placement.
            teacher_logits: Replicated tensor of shape
                ``[batch, sequence, vocab]`` containing teacher logits.

        Returns:
            Replicated tensor of shape ``[batch, sequence, vocab]`` for a
            non-TP student, otherwise a contiguous tensor of shape
            ``[batch, sequence, local_vocab]`` matching this rank's shard.
        """
        if not isinstance(student_logits, DTensor):
            return teacher_logits
        vocab_dim = student_logits.ndim - 1
        for mesh_dim, placement in enumerate(student_logits.placements):
            if isinstance(placement, Shard) and placement.dim in {-1, vocab_dim}:
                group = student_logits.device_mesh.get_group(mesh_dim)
                chunks = torch.tensor_split(teacher_logits, dist.get_world_size(group), dim=-1)
                return chunks[dist.get_rank(group)].contiguous()
        return teacher_logits

    def broadcast_command(self, command: int | None = None) -> int:
        """Broadcast one worker command from the first student rank.

        Args:
            command: Command supplied on student ranks. Teacher ranks pass
                ``None`` while waiting for the broadcast.

        Returns:
            Broadcast command value on every student and teacher rank.
        """
        value = command if self.rank == self.student_ranks[0] else 0
        tensor = torch.tensor(value, dtype=torch.int32, device=self.device)
        dist.broadcast(tensor, src=self.student_ranks[0], group=self.control_group)
        return int(tensor.item())

    def synchronize(self) -> None:
        """Wait for both model roles without using the default process group."""
        dist.barrier(group=self.control_group)

    def _broadcast_tree(self, value: Any, route: _Route) -> Any:
        """Broadcast a nested tensor tree along one route.

        Tensor leaves may have arbitrary shapes and axis order.

        Args:
            value: Nested source value on ``route.src`` and ``None`` elsewhere.
            route: Source, membership, and process group for the broadcast.

        Returns:
            Reconstructed value on route members and ``None`` elsewhere.
            Receiving tensors preserve source shape and dtype on ``self.device``.
        """
        if self.rank not in route.ranks:
            return None
        source_tensors: list[torch.Tensor] = []
        objects = [None]
        if self.rank == route.src:
            objects[0] = _tree_spec(value, source_tensors)
        dist.broadcast_object_list(objects, src=route.src, group=route.group, device=self.device)
        spec = objects[0]
        if self.rank == route.src:
            tensors = [tensor.to(self.device).contiguous() for tensor in source_tensors]
        else:
            tensor_specs: dict[int, tuple[tuple[int, ...], str]] = {}

            def collect(node: Any) -> None:
                if node[0] == "tensor":
                    tensor_specs[node[1]] = (node[2], node[3])
                elif node[0] == "dict":
                    for _, child in node[1]:
                        collect(child)
                elif node[0] in {"list", "tuple"}:
                    for child in node[1]:
                        collect(child)

            collect(spec)
            tensors = [
                torch.empty(shape, dtype=getattr(torch, dtype_name), device=self.device)
                for _, (shape, dtype_name) in sorted(tensor_specs.items())
            ]
        for tensor in tensors:
            dist.broadcast(tensor, src=route.src, group=route.group)
        return _tree_from_spec(spec, tensors)

    def send_batch(self, wave: int, batch: Any | None) -> Any | None:
        """Send one nested batch from each active student replica to a teacher.

        Batch tensor leaves preserve their original shapes and axis order.

        Args:
            wave: Routing wave index.
            batch: Nested student batch on student ranks and ``None`` on teacher
                ranks.

        Returns:
            Assigned nested batch on teacher ranks and ``None`` on student ranks.
        """
        teacher_batch = None
        for route in self.input_routes[wave]:
            received = self._broadcast_tree(batch if self.rank == route.src else None, route)
            if self.is_teacher and self.rank in route.ranks:
                teacher_batch = received
        return teacher_batch

    def send_logits(self, wave: int, logits: torch.Tensor | None) -> torch.Tensor | None:
        """Send full teacher logits back to the assigned student replica.

        Args:
            wave: Routing wave index.
            logits: Tensor of shape ``[batch, sequence, vocab]`` containing full
                replicated teacher logits on the teacher output rank, otherwise
                ``None``.

        Returns:
            Replicated tensor of shape ``[batch, sequence, vocab]`` on ranks in
            the assigned student replica, otherwise ``None``.
        """
        student_logits = None
        for route in self.output_routes[wave]:
            received = self._broadcast_tree(logits if self.rank == route.src else None, route)
            if self.is_student and self.rank in route.ranks:
                student_logits = received
        return student_logits
