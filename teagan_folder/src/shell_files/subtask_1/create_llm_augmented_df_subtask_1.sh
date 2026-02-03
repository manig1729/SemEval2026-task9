#! /bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --account=blanca-clearlab1
#SBATCH --partition=blanca-clearlab1
#SBATCH --qos=blanca-clearlab1
#SBATCH --gres=gpu:1
#SBATCH --output=job_logs/create_llm_df_output-%j.out
#SBATCH --time=05:00:00

# 1. Changing to project directory
cd /projects/tejo9855/Projects/SemEval2026-task9/teagan_folder/

# 2. Loading Modules
module load anaconda
module load cuda/12.1.1

# 3. Loading conda environment
conda activate teagan-conda-env-curc

# 6. Run python script
python src/scripts/subtask_1/run_llm_counterfactual_aug.py