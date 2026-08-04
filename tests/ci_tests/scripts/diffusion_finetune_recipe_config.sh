# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

configure_diffusion_finetune_recipe() {
    local recipe_name="$1"

    case "$recipe_name" in
        wan2_1_t2v_flow*)
            MEDIA_TYPE="video"
            PROCESSOR="wan"
            GENERATE_CONFIG="examples/diffusion/generate/configs/generate_wan.yaml"
            MODEL_NAME="Wan-AI/Wan2.1-T2V-14B-Diffusers"
            INFER_NUM_FRAMES=9
            PREPROCESS_EXTRA_ARGS=""
            ;;
        hunyuan_t2v_flow*)
            MEDIA_TYPE="video"
            PROCESSOR="hunyuan"
            GENERATE_CONFIG="examples/diffusion/generate/configs/generate_hunyuan.yaml"
            MODEL_NAME="hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_t2v"
            INFER_NUM_FRAMES=5
            PREPROCESS_EXTRA_ARGS="--target_frames 13"
            ;;
        ltx2_3_t2v_flow*)
            MEDIA_TYPE="video"
            PROCESSOR="ltx2"
            GENERATE_CONFIG="examples/diffusion/generate/configs/generate_ltx2.yaml"
            MODEL_NAME="diffusers/LTX-2.3-Diffusers"
            INFER_NUM_FRAMES=9
            PREPROCESS_EXTRA_ARGS="--num_frames 9 --output_format pt"
            ;;
        flux_t2i_flow*)
            MEDIA_TYPE="image"
            PROCESSOR="flux"
            GENERATE_CONFIG="examples/diffusion/generate/configs/generate_flux.yaml"
            MODEL_NAME="black-forest-labs/FLUX.1-dev"
            PREPROCESS_EXTRA_ARGS=""
            ;;
        qwen_image_t2i_flow*)
            MEDIA_TYPE="image"
            PROCESSOR="qwen_image"
            GENERATE_CONFIG="examples/diffusion/generate/configs/generate_qwen_image.yaml"
            MODEL_NAME="Qwen/Qwen-Image"
            PREPROCESS_EXTRA_ARGS=""
            ;;
        *)
            echo "ERROR: Unknown recipe '$recipe_name'. Add a case to diffusion_finetune_recipe_config.sh." >&2
            return 1
            ;;
    esac
}
