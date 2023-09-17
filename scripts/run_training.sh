#!/bin/bash -l
## Nazwa zlecenia
#SBATCH -J energy_eff_training
## Liczba alokowanych węzłów
#SBATCH -N 1
## Liczba zadań per węzeł (domyślnie jest to liczba alokowanych rdzeni na węźle)
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=20
## Ilość pamięci przypadającej na jeden rdzeń obliczeniowy (domyślnie 4GB na rdzeń)
#SBATCH --mem=500GB
## Maksymalny czas trwania zlecenia (format HH:MM:SS)
#SBATCH --time=0:30:00
## Nazwa grantu do rozliczenia zużycia zasobów
#SBATCH -A plgsano4-gpu-a100
## Specyfikacja partycjii da
#SBATCH --partition plgrid-gpu-a100
#SBATCH --gpus=4
## Plik ze standardowym wyjściem
#SBATCH --output="output_files/stdout.out"
## Plik ze standardowym wyjściem błędó0w
#SBATCH --error="output_files/stderr.err"



ml CUDA/11.7
# ml GCC/11.2.0
# ml OpenMPI/4.1.2-CUDA-11.6.0
nvidia-smi
# conda activate /net/tscratch/people/plgmazurekagh/energy_efficient_ai/energy_efficient_env
conda activate /net/tscratch/people/plgmazurekagh/conda_envs/lightning_bagua_env
cd $SCRATCH/energy_efficient_ai/efficient_segmentation
export WANDB_API_KEY=$(cat "wandb_api_key.txt")
export OMPI_MCA_opal_cuda_support=true
# export NCCL_DEBUG=INFO

srun -u python  src/main.py \
    --train \
    --training_data_path data/open_neuro_mixed/ \
    --lr 0.001 \
    --num_classes 1 \
    --epochs 150 \
    --img_size 256 \
    --batch_size 128 \
    --model AttSqueezeUnet \
    --loss_function DiceLoss \
    --exp_name "att-squeezeu-unet-128" \
    --n_workers 6 \
    --wandb


