#!/bin/bash -l
## Nazwa zlecenia
#SBATCH -J energy_eff_eval
## Liczba alokowanych węzłów
#SBATCH -N 1
## Liczba zadań per węzeł (domyślnie jest to liczba alokowanych rdzeni na węźle)
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
## Ilość pamięci przypadającej na jeden rdzeń obliczeniowy (domyślnie 4GB na rdzeń)
#SBATCH --mem-per-cpu=2GB
## Maksymalny czas trwania zlecenia (format HH:MM:SS)
#SBATCH --time=00:10:00
## Nazwa grantu do rozliczenia zużycia zasobów
#SBATCH -A plgsano4-gpu-a100
## Specyfikacja partycji
#SBATCH --partition plgrid-gpu-a100
#SBATCH --gpus-per-task=4
## Plik ze standardowym wyjściem
#SBATCH --output="output_files/stdout_test.out"
## Plik ze standardowym wyjściem błędó0w
#SBATCH --error="output_files/stderr_test.err"
ml CUDA/11.7
# conda activate $SCRATCH/energy_efficient_ai/energy_efficient_env
conda activate /net/tscratch/people/plgmazurekagh/conda_envs/lightning_bagua_env
cd $SCRATCH/energy_efficient_ai/efficient_segmentation
srun -u python  src/main.py \
    --test \
    --training_data_path data/open_neuro_mixed/ \
    --validation_data_path data/validation_data/ \
    --testing_data_path  data/testing_data/ \
    --test_results_save_path data/test_results/ \
    --model_path checkpoints/best_model.ckpt \
    --lr 0.001 \
    --num_classes 1 \
    --epochs 150 \
    --img_size 256 \
    --batch_size 128 \
    --model AttSqueezeUnet \
    --loss_function DiceLoss 