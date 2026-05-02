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

import os

import nemo_run as run

# List of tuples (string to pass to --dataset argument, additional arguments specific to the dataset)
eval_datasets = {}

eval_datasets['debug_text'] = [
    ('fiqa', ''),
    ('techqa_beir', ''),
]

"""eval_datasets['vision_split'] = [
    ('bo767_fullpage', ''),
    ('earnings', ''),
    ('vidore_infovqa_test_subsampled vidore_arxivqa_test_subsampled vidore_docvqa_test_subsampled vidore_shiftproject_test vidore_syntheticDocQA_artificial_intelligence_test vidore_syntheticDocQA_energy_test vidore_syntheticDocQA_government_reports_test vidore_syntheticDocQA_healthcare_industry_test vidore_tabfquad_test_subsampled vidore_tatdqa_test', ''),
]"""
eval_datasets['vision_split'] = [
    ('earnings_v2', ''),
    ('dc10k', ''),
    ('earnings', ''),
    ('bo767_fullpage', ''),
    ('vidore_all', ''),
]

eval_datasets['text_short_context'] = [
    ('fiqa nq_sample35 hotpotqa_sample35 miracl_ar_sample miracl_de_sample miracl_en_sample miracl_es_sample miracl_fi_sample miracl_fr_sample miracl_ja_sample miracl_ko_sample miracl_ru_sample miracl_th_sample miracl_zh_sample techqa_beir bo767_text earnings_text ohrbench_text_en ohrbench_text_zh', ''),
]

eval_datasets['small_textqa_eval_20250508'] = [
    ('bo767_text earnings_text fiqa nq_sample35 hotpotqa_sample35 techqa_beir miracl_ar_sample miracl_de_sample miracl_en_sample miracl_es_sample miracl_fi_sample miracl_fr_sample miracl_ja_sample miracl_ko_sample miracl_ru_sample miracl_th_sample miracl_zh_sample test-context-de-question-ar test-context-ar-question-es test-context-en-question-zh test-context-hi-question-de test-context-vi-question-ar test-context-zh-question-en test-context-es-question-vi', ''),
    ('mldr_en_sample_bge_400query mldr_it_full mldr_pt_full mldr_ja_full mldr_de_full', ' --batch_size 2 --max_length 8192 '),
]

eval_datasets['full_text_eval'] = [
    ('bo767_text earnings_text', ' --batch_size 1 --max_length 8192 '),
    ('ohrbench_text_en ohrbench_text_zh', ' --batch_size 1 --max_length 8192 '),
    ('fiqa nq_sample35 hotpotqa_sample35 techqa_beir', ''),
    ('nq', ''),
    ('hotpotqa', ''),
    ('mldr_fr_full mldr_pt_full', ' --batch_size 1 --max_length 8192 '),
    ('mldr_ja_full mldr_de_full', ' --batch_size 1 --max_length 8192 '),
    ('mldr_it_full mldr_es_full mldr_ko_full', ' --batch_size 1 --max_length 8192 '),
    ('mldr_zh_full', ' --batch_size 1 --max_length 8192 '),
    ('mldr_en_full', ' --batch_size 1 --max_length 8192 '),
    ('mldr_ar_full mldr_hi_full', ' --batch_size 1 --max_length 8192 '),
    ('mldr_th_full mldr_ru_full', ' --batch_size 1 --max_length 8192 '),
    ('miracl_ar_sample miracl_id_sample miracl_fi_sample miracl_sw_sample miracl_bn_sample miracl_de_sample miracl_en_sample miracl_ja_sample miracl_hi_sample miracl_ko_sample miracl_yo_sample miracl_te_sample miracl_es_sample miracl_fa_sample miracl_th_sample miracl_ru_sample miracl_zh_sample miracl_fr_sample miracl_en-nl-helsinki_sample miracl_en-pt-helsinki_sample miracl_en-it-helsinki_sample miracl_en-sv-helsinki_sample miracl_en-cs-helsinki_sample miracl_en-he-helsinki_sample miracl_es-no-helsinki_sample miracl_es-pl-helsinki_sample miracl_en-tr-helsinki_sample miracl_en-da-helsinki_sample miracl_en-hu-helsinki_sample', ''),
    ('test-context-ar-question-ar test-context-ar-question-de test-context-ar-question-en test-context-ar-question-es test-context-ar-question-hi test-context-ar-question-vi test-context-ar-question-zh test-context-de-question-ar test-context-de-question-de test-context-de-question-en test-context-de-question-es test-context-de-question-hi test-context-de-question-vi test-context-de-question-zh test-context-en-question-ar test-context-en-question-de test-context-en-question-en test-context-en-question-es test-context-en-question-hi test-context-en-question-vi', ''),
    ('test-context-en-question-zh test-context-es-question-ar test-context-es-question-de test-context-es-question-en test-context-es-question-es test-context-es-question-hi test-context-es-question-vi test-context-es-question-zh test-context-hi-question-ar test-context-hi-question-de test-context-hi-question-en test-context-hi-question-es test-context-hi-question-hi test-context-hi-question-vi test-context-hi-question-zh test-context-vi-question-ar test-context-vi-question-de test-context-vi-question-en test-context-vi-question-es test-context-vi-question-hi', ''),
    ('test-context-vi-question-vi test-context-vi-question-zh test-context-zh-question-ar test-context-zh-question-de test-context-zh-question-en test-context-zh-question-es test-context-zh-question-hi test-context-zh-question-vi test-context-zh-question-zh', ''),
    ('test-context-de-question-zh test-context-en-question-ar test-context-en-question-de test-context-en-question-en', ''),
    ('test-context-en-question-es test-context-en-question-hi test-context-en-question-vi test-context-hi-question-hi', ''),
    ('test-context-hi-question-vi test-context-hi-question-zh test-context-vi-question-ar test-context-vi-question-de', ''),
    ('test-context-vi-question-en test-context-vi-question-es test-context-vi-question-hi', '')
]

eval_datasets['MTEB_ALL_QUICK'] = [
    ('MMTEB_QUICK', ' --max_length 512 --batch_size 32 '),
    ('RTEB_QUICK', ' --max_length 4096 --batch_size 16 '),
    ('MTEB_ENG', ' --max_length 2048 --batch_size 16 '),
]


eval_datasets["vidore_v3_complex_ocr"] = [
    ("vidore_v3_finance_bank_reports_en_complex_ocr vidore_v3_military_technical_reports_complex_ocr vidore_v3_education_computer_science_en_complex_ocr vidore_v3_healthcare_fda_en_complex_ocr vidore_v3_hr_european_reports_en_complex_ocr vidore_v3_energy_reports_fr_complex_ocr vidore_v3_education_physics_fr_complex_ocr vidore_v3_luxury_financial_reports_fr_complex_ocr",
     " --eval_type text_embedding_eval"),
]

eval_datasets["text_embedding_eval"] = [
    ("dc10k_complex_ocr earnings_v2_complex_ocr fiqa nq_sample35 hotpotqa_sample35 techqa_beir miracl_ar_sample miracl_de_sample miracl_en_sample miracl_es_sample miracl_fi_sample miracl_fr_sample miracl_ja_sample miracl_ko_sample miracl_ru_sample miracl_th_sample miracl_zh_sample mldr_en_sample_bge_400query mldr_it_full mldr_pt_full mldr_ja_full mldr_de_full test-context-de-question-ar test-context-ar-question-es test-context-en-question-zh test-context-hi-question-de test-context-vi-question-ar test-context-zh-question-en test-context-es-question-vi",
     " --eval_type text_embedding_eval"),
]

def get_script(
    script,
    args: str
):
    script = run.Script(
        inline=f"""
            {script} {args}
            """
    )
    return(script)

def extract_args_by_prefix(
    args: dict,
    prefix = 'train_',
    exceptions = []
):
    out = {}
    for key in args.keys():
        if key.startswith(prefix) and key not in exceptions:
            out[key.replace(prefix, '', 1)] = args[key]
    return(out)

def validate_args(args, add_args):
    if not os.path.isfile(args.train_script) and args.num_training_rounds>=1:
        raise ValueError('Training Script does not exists: ' + str(args.train_script))
    if not os.path.isfile(args.eval_script) and args.eval_datasets is not None and len(args.eval_datasets) > 0:
        raise ValueError(f'Evaluation script is required when eval_datasets are specified, but script does not exist: {args.eval_script} (datasets: {args.eval_datasets})')
    if args.num_nodes>1 and 'interactive' in str(args.partition):
        raise ValueError('We cannot run multinode runs on interactive partition')
    if 'model_name_or_path' in args.more_eval_args and args.num_training_rounds>0:
        raise ValueError('Evaluation args contain model_name_or_path parameter, which means an existing checkpoint should be evaluated, but num_training_rounds>0, which means that a new model is trained.')

    if len(add_args)>0:
        if args.more_train_args.strip() == "":
            # This is necessary for hyperperameter tuning with wandb sweeps, as it does not allow
            # injecting sampled CLI args inside "${args}". So in that case we feed the sampled
            # training hparams as regular CLI args and capture them here
            args.more_train_args = " ".join([f"--{k} {v}" for k, v in add_args.items() ])
        else:
            print(args.more_train_args)
            print(add_args)
            raise ValueError('Additional CLI args can be used to provide the training args, but cannot be used in conjunction with --more_train_args is set.')
    # ToDo: How to check that train_scripts and eval_scripts are executable (chmod -R 777) 

def draco(
    account: str = "datascience_nemo_retriever",
    container_image: str = "/lustre/fsw/portfolios/llmservice/users/mengyaox/inform/images/inform-python3.10-vidore-beir-flashattn.sqsh",
    partition: str = None,
    nodes: int=1,
    gpus_per_node: int = 8,
    time: str = "03:57:00",
    username: str = os.getenv("USER"),
    dependency_type: str = "afterok",
    job_name_prefix: str = "",
    master_port: int = 29500,
) -> run.Config[run.Executor]:
    """
    Create a Slurm executor configuration for the DRACO cluster.

    Args:
        account (str): The account to use for the job.
        partition (str): The partition to use for the job.
        container_image (str): The container image to use for the job.
        gpus_per_node (int): The number of GPUs per node to use for the job.
        time (str): The time to use for the job.
        username (str): The username to use for the job.
        master_port (str): The port for torchrun rendezvous endpoint.

    Returns:
        run.Config[run.SlurmExecutor]: The Slurm executor configuration for the DRACO cluster.
    """
    env_vars = {}
    num_processes = nodes * gpus_per_node 
    env_vars['N_NODES'] = str(nodes)
    env_vars['GPUS_PER_NODE'] = str(gpus_per_node)
    env_vars['CUDA_VISIBLE_DEVICES'] = ",".join([str(i) for i in list(range(gpus_per_node))])
    env_vars['N_PROCESSES'] = str(nodes * gpus_per_node)
    env_vars["MASTER_ADDR"] = "$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)"
    env_vars["MASTER_PORT"] = str(master_port)
    
    job_dir = f"/lustre/fsw/portfolios/llmservice/users/{username}/jobs"
    ssh_tunnel = run.SSHTunnel(
        host="draco-oci-login-01.draco-oci-iad.nvidia.com",
        user=username,
        job_dir=job_dir,
    )
    slurm_executor = run.SlurmExecutor(
        account=account,
        partition=partition,
        nodes=nodes,
        container_image=container_image,
        gpus_per_node=gpus_per_node,
        time=time,
        container_mounts=[
            "/lustre:/lustre",'/home/:/home/'
        ],
        job_dir=job_dir,
        tunnel=ssh_tunnel,
        env_vars=env_vars,
        job_name_prefix=job_name_prefix,
        #Job will start after the job terminated (regardless of exit code)
        # Default is afterok, which requires the job to finish successfully,
        # but the first job will be killed after 4 hours
        dependency_type=dependency_type,
        # avoid git commit every time and everything
        packager = run.GitArchivePackager(include_untracked=True, include_uncommitted=True)
    )
    return slurm_executor
