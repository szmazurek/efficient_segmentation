#!/bin/bash -l
## Nazwa zlecenia
#SBATCH -J energy_eff_training
## Liczba alokowanych węzłów
#SBATCH -N 1
## Liczba zadań per węzeł (domyślnie jest to liczba alokowanych rdzeni na węźle)
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
## Ilość pamięci przypadającej na jeden rdzeń obliczeniowy (domyślnie 4GB na rdzeń)
#SBATCH --mem-per-cpu=5GB
## Maksymalny czas trwania zlecenia (format HH:MM:SS)
#SBATCH --time=00:10:00
## Nazwa grantu do rozliczenia zużycia zasobów
#SBATCH -A plgsanoathena-gpu-a100
## Specyfikacja partycji
#SBATCH --partition plgrid-gpu-a100
#SBATCH --gpus=4
## Plik ze standardowym wyjściem
#SBATCH --output="output_files/stdout.out"
## Plik ze standardowym wyjściem błędó0w
#SBATCH --error="output_files/stderr.err"



ml CUDA/11.7
# ml GCC/11.2.0
# ml OpenMPI/4.1.2-CUDA-11.6.0

# conda activate /net/tscratch/people/plgmazurekagh/energy_efficient_ai/energy_efficient_env
conda activate /net/tscratch/people/plgmazurekagh/conda_envs/lightning_bagua_env
cd $SCRATCH/energy_efficient_ai/E2MIP_Challenge_FetalBrainSegmentation
export WANDB_API_KEY=$(cat "wandb_api_key.txt")
export OMPI_MCA_opal_cuda_support=true


srun -u python  src/main.py \
    --train \
    --training_data_path data/training_data \
    --lr 0.001 \
    --num_classes 1 \
    --epochs 100 \
    --batch_size 32 \
    --model Unet \
    --loss_function MCCLoss \
    --exp_name "Adam" \
    --wandb
