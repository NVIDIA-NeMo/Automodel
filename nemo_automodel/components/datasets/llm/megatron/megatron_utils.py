from typing import List, Optional, Tuple

def get_blend_from_list(
    blend: Optional[List[str]],
) -> Optional[Tuple[List[str], Optional[List[float]]]]:
    """Get the megatron.core.datasets.blended_megatron_dataset_config.BlendedMegatronDatasetConfig blend from the blend list

    Args:
        blend (Optional[List[str]]): The blend list, which can be either (1) a list of prefixes, e.g. ["path/to/dataset_1_prefix", "path/to/dataset_2_prefix"], or (2) a flattened, zipped list of weights and prefixes, e.g. ["30", "path/to/dataset_1_prefix", "70", "path/to/dataset_2_prefix"]

    Returns:
        Optional[Tuple[List[str], Optional[List[float]]]]: The blend, consisting of a list of dataset prefixes and optionally a list of dataset weights, e.g. [["path/to/dataset_1_prefix", "path/to/dataset_2_prefix"], [30.0, 70.0]].
    """
    if blend is None:
        return None

    if len(blend) % 2 == 1:
        weight_per_dataset = None
        raw_prefix_per_dataset = blend
    else:
        raw_weight_per_dataset, raw_prefix_per_dataset = zip(
            *[(blend[i], blend[i + 1]) for i in range(0, len(blend), 2)]
        )

        weight_per_dataset = []
        for rwpd in raw_weight_per_dataset:
            try:
                weight = float(rwpd)
            except ValueError:
                weight = None
            weight_per_dataset.append(weight)

        is_none = map(lambda _: _ is None, weight_per_dataset)
        if any(is_none):
            assert all(is_none)
            weight_per_dataset = None
            raw_prefix_per_dataset = blend

    prefix_per_dataset = [rppd.strip() for rppd in raw_prefix_per_dataset]

    return prefix_per_dataset, weight_per_dataset