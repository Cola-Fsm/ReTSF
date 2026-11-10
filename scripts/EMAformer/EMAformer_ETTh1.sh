

model_name=EMAformer

python -u ../../run.py \
  --is_training 1 \
  --root_path ../../dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_320_96 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 320 \
  --pred_len 96 \
  --e_layers 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 256 \
  --n_heads 4 \
  --d_ff 256 \
  --output_proj_dropout 0.3 \
  --itr 1 \
  --cycle 24 

python -u ../../run.py \
  --is_training 1 \
  --root_path ../../dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_320_192 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 320 \
  --pred_len 192 \
  --e_layers 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 256 \
  --n_heads 4 \
  --d_ff 256 \
  --output_proj_dropout 0.3 \
  --itr 1 \
  --cycle 24 

python -u ../../run.py \
  --is_training 1 \
  --root_path ../../dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_320_336 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 320 \
  --pred_len 336 \
  --e_layers 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 256 \
  --n_heads 4 \
  --d_ff 256 \
  --output_proj_dropout 0.3 \
  --itr 1 \
  --cycle 24 

python -u ../../run.py \
  --is_training 1 \
  --root_path ../../dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_320_720 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 320 \
  --pred_len 720 \
  --e_layers 2 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 256 \
  --n_heads 4 \
  --d_ff 256 \
  --output_proj_dropout 0.5 \
  --itr 1 \
  --patience 5 \
  --cycle 24 
