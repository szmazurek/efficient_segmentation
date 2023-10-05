#!/bin/bash -l
## Nazwa zlecenia
#SBATCH -J energy_eff_training
## Liczba alokowanych węzłów
#SBATCH -N 1
## Liczba zadań per węzeł (domyślnie jest to liczba alokowanych rdzeni na węźle)
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=22
## Ilość pamięci przypadającej na jeden rdzeń obliczeniowy (domyślnie 4GB na rdzeń)
#SBATCH --mem=200GB
## Maksymalny czas trwania zlecenia (format HH:MM:SS)
#SBATCH --time=0:40:00
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
conda activate /net/tscratch/people/plgmazurekagh/conda_envs/lightning_bagua_env
cd $SCRATCH/energy_efficient_ai/efficient_segmentation
export WANDB_API_KEY=$(cat "wandb_api_key.txt")
export OMPI_MCA_opal_cuda_support=true

srun -u python  src/main.py \
    --train \
    --training_data_path data/open_neuro_mixed/ \
    --validation_data_path data/validation_data/ \
    --testing_data_path  data/testing_data/ \
    --test_results_save_path data/test_results/ \
    --model_path checkpoints/best_model.ckpt \
    --lr 0.001 \
    --num_classes 1 \
    --epochs 1 \
    --img_size 256 \
    --batch_size 128 \
    --model AttSqueezeUnet \
    --loss_function DiceLoss 


