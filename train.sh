#!/bin/bash
#SBATCH -p inspur           # 指定分区
#SBATCH --gres=gpu:A100:1   # 申请1张A100 GPU
#SBATCH -c 8                # 申请8个CPU核心
#SBATCH -o /home/LAB/chenlb24/PromptMRG/logs/output_%j.log    # 输出日志文件，%j代表作业号
#SBATCH -e /home/LAB/chenlb24/PromptMRG/logs/error_%j.log     # 错误日志文件

conda activate promptmrg

python /home/LAB/chenlb24/PromptMRG/main_train.py --save_dir results/A100_1