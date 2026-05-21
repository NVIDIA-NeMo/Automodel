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

# Usage: python3 submit_nemo_run.py -n test_qwen  -tr scripts/mengyaox/local_qwen2_5_vl_train_script.sh -e scripts/mengyaox/local_qwen2_5_eval_script.sh --eval_datasets vision_split small_textqa_eval_20250508  --run_local -r 2

import os
import argparse
from datetime import datetime

import nemo_run as run

from automation.utils import eval_datasets, draco, get_script, validate_args
from torchx.specs.api import AppState
import csv
import wandb
import time
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
def get_args():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument(
        '-n', '--experiment_name',
        type=str,
        help='Name of the Experiment',
        default=None
    )
    parser.add_argument(
        '--container_image',
        type=str,
        help='Container used for training and evaluation',
        default='/home/bschifferer/llm2/team_folder/03_sqsh/inform-python3.10-vidore-beir-flashattn-nemorun-ncclupdate.sqsh'
    )
    parser.add_argument(
        '-ckpd', '--checkpoint_directory',
        type=str,
        help='Path to the checkpoint directory, if not provided, the checkpoint directory will be set later',
        default=None
    )
    parser.add_argument(
        '--partition',
        type=str,
        help='Partition - e.g. interactive',
        default="batch_block1,batch_block3,batch_block4"
    )
    parser.add_argument(
        '--num_nodes',
        type=int,
        help='Number of nodes for training jobs',
        default=1
    )
    parser.add_argument(
        '-r', '--num_training_rounds',
        type=int,
        help='Recurrent training runs to resume from checkpoint',
        default=1
    )
    parser.add_argument(
        '-yc', '--yaml_config',
        type=str,
        help='YAML config file for the experiment',
        default=None
    )
    parser.add_argument(
        '-tr', '--train_script',
        type=str,
        help='Bash script to start training models',
        default='./scripts/demo_start_training.sh'
    )
    parser.add_argument(
        '--more_train_args',
        type=str,
        help='Addition train arguments passed to the script',
        default=''
    )
    parser.add_argument(
        '-e', '--eval_script',
        type=str,
        help='Bash script to evaluate models',
        default='./scripts/demo_eval_script.sh'
    )
    parser.add_argument(
        '--more_eval_args',
        type=str,
        help='Addition eval arguments passed to the script',
        default=''
    )
    parser.add_argument(
        '--eval_datasets',
        type=str,
        nargs='*',
        help='Datasets to evaluate (can specify multiple datasets)',
        default=None
    )

    parser.add_argument(
        '--run_local',
        help='Run locally',
        action='store_true',
    )

    parser.add_argument(
        '-nt', '--no_append_timestamp',
        action='store_true',
        help='Disable appending timestamp to experiment name'
    )
    parser.add_argument(
        '--wait_until_finished',
        type=str2bool,
        nargs='?', 
        const=True, 
        default=False,
        help='If the script should wait until all tasks (Slurm jobs) are finished',
    )
    parser.add_argument(
        '--log_wandb',
        type=str2bool,
        nargs='?', 
        const=True, 
        default=False,
        help='If W&B tracking should be disabled',
    )
    parser.add_argument(
        '--wandb_project_name',
        type=str,
        help='Project name for wandb log',
        default="inform-simlm-embedding"
    )
    parser.add_argument(
        '--results_dir_name',
        type=str,
        help='Subfolder inside the checkpoint that might contain the metrics of all evaluation runs (the recall5_results.tsv file)',
        default="eval_results"
    )

    
    args, extra_args = parser.parse_known_args()

    def convert_value(value):
        if value.replace('.', '', 1).isdigit():  # Check if it's a number (int or float)
            return int(value) if '.' not in value else float(value)
        return value  # Keep as string if not numeric

    # Treat extra args separated by "="  (like "--seed=123"), as that is how wandb sweeps injects args
    extra_args_dict = {}
    arg_index = 0
    while arg_index < len(extra_args):
        arg = extra_args[arg_index]
        if "=" in arg:
            key, value = arg.split("=", 1)
        else:
            if arg_index + 1 >= len(extra_args):
                parser.error(f"Expected a value after extra argument {arg!r}.")
            key = arg
            value = extra_args[arg_index + 1]
            arg_index += 1

        key = key.lstrip("-")
        if not key:
            parser.error(f"Unexpected empty extra argument name in {arg!r}.")
        extra_args_dict[key] = convert_value(value)
        arg_index += 1
    
    return args, extra_args_dict

def experiment_tasks_status(exp):
    tasks_status = exp.status(return_dict=True) 
    tasks_status = {task: tasks_status[task]["status"] for task in tasks_status.keys()}
    return tasks_status        

def experiment_has_finished(tasks_status):
    all_finished = (all(tasks_status[task] in [AppState.SUCCEEDED, AppState.FAILED, AppState.CANCELLED] for task in tasks_status.keys()))
    return all_finished

def read_eval_metrics(experiment_checkpoint_directory, results_dir_name):
    metrics_file = os.path.join(experiment_checkpoint_directory, results_dir_name, "recall5_results.tsv")
    if not os.path.exists(metrics_file):
        logger.info(f"Metrics file not exist: {metrics_file}")
        return
    
    with open(metrics_file, 'r') as file:
        csv_reader = csv.reader(file, delimiter="\t")
        header = next(csv_reader)
        first_row = next(csv_reader)
        first_row = [float(v) for v in first_row]
        eval_metrics = dict(zip(header, first_row))
    return eval_metrics


def log_metrics_to_wandb(eval_metrics):
    if eval_metrics is None:
        raise Exception("Eval metrics not found to be logged to W&B.")
    
    valid_metric_names = ['Bo767', 'Earnings', 'Earnings Overall', 'ArxivQA', 'DocVQA', 'InfoVQA', 'Shiftproject', 'synthetic AI', 'synthetic Energy', 'synthetic Government', 'synthetic Health', 'TabfQUAD', 'TatDQA']
    valid_metrics = {metric: value for metric, value in eval_metrics.items() if metric in valid_metric_names}

    if len(valid_metrics) == len(valid_metric_names):
        avg_recall_at_5 = sum(valid_metrics.values()) / len(valid_metrics)
        eval_metrics["avg_recall_at_5"] = avg_recall_at_5
        wandb.log(eval_metrics)
    else:
        logger.info(
            "The average metrics cannot be logged because some metrics were not found: %s",
            set(valid_metric_names) - set(valid_metrics.keys()),
        )

def main():
    username = os.getenv("USER")
    args, add_args = get_args()    
    validate_args(args, add_args)
    
    logger.info('User: ' + str(username))
    logger.info("########################## " + "args" + " #######################")
    logger.info(args)
      

    if args.experiment_name is None:
        experiment_id = "train_model_experiment"
    else:
        experiment_id = args.experiment_name
        
    if not args.no_append_timestamp:
        experiment_id = f"{experiment_id}_{datetime.now().strftime('%m-%d-%H%M%S')}"  
        run_experiment_id = f"{experiment_id}"  
    else:
        # To avoid the same nemorun experiment id if resuming from ckp
        # e.g.FileExistsError: [Errno 17] File exists: '.nemo_run/experiments/...'
        run_experiment_id = f"{experiment_id}_{datetime.now().strftime('%m-%d-%H%M%S')}"

    if args.run_local:
        train_executor = run.LocalExecutor()
        eval_executor = run.LocalExecutor()
    else:
        train_executor = draco(
            container_image = args.container_image,
            partition = args.partition,
            nodes = int(args.num_nodes),        
            dependency_type="afterany",
            job_name_prefix=experiment_id,
            gpus_per_node=0 if args.partition == "cpu" else 8,
        )
        eval_executor = draco(
            container_image = args.container_image,
            partition = args.partition,
            nodes = 1,
            job_name_prefix='e_' + experiment_id,
        )
    
    
    if args.checkpoint_directory is None:
        if isinstance(train_executor, run.SlurmExecutor):
            checkpoints_directory = f"/lustre/fsw/portfolios/llmservice/users/{username}/model_checkpoints"
        else:
            checkpoints_directory = "/model_checkpoints"
    else:
        checkpoints_directory = args.checkpoint_directory
    
    experiment_checkpoint_directory = f"{checkpoints_directory}/{experiment_id}"
    args.more_train_args += f' --config {args.yaml_config} --checkpoint.checkpoint_dir {experiment_checkpoint_directory} --wandb.name {experiment_id} '
    if 'model_name_or_path' not in args.more_eval_args: 
        args.more_eval_args += f' --model_name_or_path {experiment_checkpoint_directory}/LATEST/model/consolidated'

    logger.info("########################## " + "more_train_args" + " #######################")
    logger.info(args.more_train_args)
    logger.info("########################## " + "more_eval_args" + " #######################")
    logger.info(args.more_eval_args)

    if args.log_wandb:
        if not args.wait_until_finished:
            raise Exception("If you use --log_wandb True you need to enable --wait_until_finished")
        
        run_config = vars(args)
        run_config['model_path'] = experiment_checkpoint_directory
        wandb.init(
                entity='nvidia-merlin',
                project=args.wandb_project_name,
                resume='allow',
                group=experiment_id,
                name=f"{experiment_id}-eval_final_metrics",
                job_type="evaluation",
                config=run_config
            )     
    
    try:
        with run.Experiment(experiment_id, id=run_experiment_id) as exp:
            train_job_name = None
            for i in range(args.num_training_rounds):
                script = get_script(
                    args.train_script,
                    args.more_train_args
                )
                if i==0:
                    logger.info('Training Script')
                    logger.info(script)
                train_job_name = exp.add(
                    script,
                    name="train_" + str(i), 
                    executor=train_executor,
                    dependencies=[train_job_name] if train_job_name else None
                )

            if args.eval_datasets is not None and len(args.eval_datasets) > 0:
                eval_commands = []
                for eval_dataset in args.eval_datasets:
                    if eval_dataset not in eval_datasets.keys():
                        eval_datasets[eval_dataset] = [(eval_dataset, "")]
                    for (dataset_list, extra_eval_args) in eval_datasets[eval_dataset]:
                        logger.info(f"Evaluating on {dataset_list}")
                        eval_args = args.more_eval_args
                        if "--eval_group_name" not in eval_args and "--eval_group_name" not in extra_eval_args:
                            eval_args += f" --eval_group_name {eval_dataset}"
                        if "--results_dir_name" not in eval_args and "--results_dir_name" not in extra_eval_args:
                            eval_args += f" --results_dir_name {args.results_dir_name}/{eval_dataset}"
                        if "--dataset_mapping" not in eval_args and "--dataset_mapping" not in extra_eval_args:
                            eval_args += f" --dataset_mapping {eval_dataset}"
                            logger.info("Dataset mappings module not available, using default mapping")
                        eval_args += " --dataset " + dataset_list + extra_eval_args
                        eval_commands.append(f"{args.eval_script} {eval_args}")

                combined_eval_inline = "\n".join(eval_commands)
                script = run.Script(inline=combined_eval_inline)
                logger.info("Combined Evaluation Script")
                logger.info(script)
                eval_job_name = exp.add(
                    script,
                    name="eval",
                    executor=eval_executor,
                    dependencies=[train_job_name] if train_job_name else None
                )

            # Launches the experiment Slurm tasks 
            if args.run_local:
                exp.run( sequential=False, tail_logs=True, direct=False)
            else:
                exp.run(detach=True, sequential=False, tail_logs=True, direct=False)

            if args.wait_until_finished:
                tasks_status = None
                while tasks_status is None or not experiment_has_finished(tasks_status):
                    time.sleep(30)                
                    tasks_status = experiment_tasks_status(exp)
                    logger.info("Tasks status: %s", tasks_status)

                failed_tasks = [task for task in tasks_status.keys() if tasks_status[task] != AppState.SUCCEEDED]
                if failed_tasks:                    
                    raise Exception(f"Some tasks failed: {tasks_status}")
            
                logger.info("All tasks succeeded")

                eval_metrics = read_eval_metrics(experiment_checkpoint_directory, args.results_dir_name)
                logger.info("Eval metrics:\n%s", eval_metrics)
                if args.log_wandb:                    
                    log_metrics_to_wandb(eval_metrics)
            else:
                logger.info("Experiment tasks launched")
    
    finally:
        if args.log_wandb:
            wandb.finish()

if __name__ == '__main__':    
    main()
