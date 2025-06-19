python viz_colored_spatial_stability.py \
    --dirs \
        alpha_test/closet_cnn_5layers_distribution_colored_emnist_variance3162_combined_lr0.01_batch256_epoch2000_LabelNoiseRate0.0_Optimsgd_Momentum0.0/no_noise_no_noise/1/1/ \
        alpha_test/closet_cnn_5layers_distribution_colored_emnist_variance3162_combined_lr0.01_batch256_epoch2000_LabelNoiseRate0.2_Optimsgd_Momentum0.0/no_noise_no_noise/2/2/ \
       alpha_test/test_closet_cnn_5layers_distribution_colored_emnist_variance3162_combined_lr0.01_batch256_epoch2000_LabelNoiseRate0.5_Optimsgd_Momentum0.0/no_noise_no_noise/1/1 \
    --target digit \
    --entity dsml-kernel24 \
    --project kobayashi_save_model \
    --output combined_plot_digit.pdf