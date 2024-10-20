
export CUDA_VISIBLE_DEVICES=1
model=Transformer
root_path=/TimerBed/Datasets
model_id=RCW
python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path $root_path \
  --model_id $model_id \
  --model $model \
  --data UEA \
  --e_layers 3 \
  --batch_size 4 \
  --d_model 32 \
  --d_ff 32 \
  --top_k 3 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.001 \
  --train_epochs 30 \
  --patience 3
