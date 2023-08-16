#!/bin/bash -l
## Nazwa zlecenia
#SBATCH -J energy_eff_eval
## Liczba alokowanych węzłów
#SBATCH -N 1
## Liczba zadań per węzeł (domyślnie jest to liczba alokowanych rdzeni na węźle)
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
## Ilość pamięci przypadającej na jeden rdzeń obliczeniowy (domyślnie 4GB na rdzeń)
#SBATCH --mem-per-cpu=15GB
## Maksymalny czas trwania zlecenia (format HH:MM:SS)
#SBATCH --time=00:10:00
## Nazwa grantu do rozliczenia zużycia zasobów
#SBATCH -A plgsanoathena-gpu-a100
## Specyfikacja partycji
#SBATCH --partition plgrid-gpu-a100
#SBATCH --gpus-per-task=1
## Plik ze standardowym wyjściem
#SBATCH --output="output_files/stdout_test.out"
## Plik ze standardowym wyjściem błędó0w
#SBATCH --error="output_files/stderr_test.err"
ml CUDA/11.7
conda activate $SCRATCH/energy_efficient_ai/energy_efficient_env
cd $SCRATCH/energy_efficient_ai/E2MIP_Challenge_FetalBrainSegmentation
srun python3.10 src/main.py \
    --test \
    --testing_data_path data/testing_data \
    --model_path "lightning_logs/version_6/checkpoints/epoch=80-step=243.ckpt" \
    --test_results_save_path data/test_results \
    --model Unet \
    --loss_function MCCLoss