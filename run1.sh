##!/bin/bash
export CUDA_VISIBLE_DEVICES=0
export MASTER_PORT=12355
train_files=(
  "./data/TREC/train_clean.csv"
#  "./data/AGNEWS/train_20.0_asym.csv"
)
eval_files=(
  "./data/TREC/test_clean.csv"
#  "./data/AGNEWS/test_clean.csv"
)
dataset_names=(
  "TREC"
#  "AGNEWS"
)
noise_types=(
  "sym"
#  "asym"
)
model_load_paths=(
  "./checkpoints/pretrained/TREC_100_100_.pth"
#  "./checkpoints/pretrained/AGNEWS_100_100_.pth"
)

min_synonyms_list=(1 2 4 5 6 3)

epochs=40
lr=1e-5

num_files=${#train_files[@]}

for ((i=0; i<$num_files; i++)); do
  train_file=${train_files[$i]}
  eval_file=${eval_files[$i]}
  dataset_name=${dataset_names[$i]}
  noise_type=${noise_types[$i]}
  model_load_path=${model_load_paths[$i]}

  echo "Running experiments for:"
  echo "Train file: $train_file"
  echo "Eval file: $eval_file"
  echo "Dataset: $dataset_name"
  echo "Noise type: $noise_type"

  for min_synonyms in "${min_synonyms_list[@]}"; do
    echo "  Running with min_synonyms=$min_synonyms"

    python train.py \
      --train_file_path "${train_file}" \
      --eval_file_path "${eval_file}" \
      --dataset_name "$dataset_name" \
      --min_synonyms $min_synonyms \
      --train_epochs $epochs \
      --lr $lr \
      --noise_type "$noise_type" \
      --model_load_path "$model_load_path" \
      --noise_ratio 40

    echo "  Finished min_synonyms=$min_synonyms"
    echo "----------------------------------------"
  done
done