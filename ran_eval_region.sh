
python dd_scratch_model_eval_region.py \
  --folder save_model/EMNIST/cnn_5layers_width1_emnist_digits_lr0.01_batch_size256_epoch2000_LabelNoiseRate0.2_Optimsgd_momentum0.0 \
  --data_dir /workspace/data/EMNIST \
  --label_noise 0.2 \
  --sample_type noisy \
  --npoints 100 \
  --epsilon 1 \
  --nsample 1000\
  --auto \
  --gpu 1 \
  --model cnn_5layers \
  --model_width 1

    experiment_name = f"model_{args.model}_npoint{args.npoints}_epsilon{args.epsilon}_{args.nsample}_seed{args.seed}"
    out_dir = os.path.join("region",experiment_name, f"EMNIST{args.label_noise}", args.sample_type)
