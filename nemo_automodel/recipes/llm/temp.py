from nemo_automodel.components.checkpoint._backports.consolidate_hf_safetensors import consolidate_safetensors_files
from nemo_automodel.components.checkpoint._backports.hf_storage import get_fqn_to_file_index_mapping


fqn_mapping = get_fqn_to_file_index_mapping("/lustre/fsw/coreai_dlalgo_genai/adasif/models/models--moonshotai--Moonlight-16B-A3B/snapshots/ce8bc137e6e29c3b7540ebdd515bbc5bdb20d915/")

consolidate_safetensors_files(
    input_dir="/lustre/fsw/coreai_dlalgo_genai/adasif/checkpoints/moonlight_dclm/epoch_0_step_57200/model/",
    output_dir="/lustre/fsw/coreai_dlalgo_genai/adasif/checkpoints/moonlight_dclm/epoch_0_step_57200/model/consolidated",
    fqn_to_index_mapping=fqn_mapping,
    num_threads=max(fqn_mapping.values()),
)
