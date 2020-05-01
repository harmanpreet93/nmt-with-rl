#!/bin/bash
#SBATCH --time=480:00
#SBATCH --gres=gpu:k80:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
# --reservation=IFT6759_2020-04-10

#module load python/3.7
#source /home/guest140/harman_venv/bin/activate

echo
date
echo ~~~~~~~~~~~~Training Transformer model
echo

python -u ../code/train_model.py \
            --config ../code/user_config.json \
            2>&1 | tee -a ../logs/DL_50_RL_50_Pos.log \
