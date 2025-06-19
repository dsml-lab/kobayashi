python viz_loss_eval.py \
    --gpu 1 \
    --model cnn_5layers \
    --model_width 1 \
    --model_path /workspace/save_model/EMNIST/cnn_5layers_width1_emnist_digits_lr0.01_batch_size256_epoch2000_LabelNoiseRate0.0_Optimsgd_momentum0.0 \
    --epoch 2000 \
    --m 100 \
    --output_txt high_loss_indices.txt \
    --data_path /workspace/data/EMNIST/EMNIST_0.1 \
    --batch_size 256