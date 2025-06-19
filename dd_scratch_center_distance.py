# main.py
import math
import os
import torch
import torch.optim as optim
import warnings
import torch.nn as nn
import numpy as np
import csv
from config import parse_args
from utils import set_seed, set_device, clear_memory
from torch.utils.data import  DataLoader, TensorDataset
from datasets import load_datasets, BalancedBatchSampler, NoisyDataset
from models import load_models
from train import (
    train_model,
    test_model,
    add_label_noise,
    select_data_points,
    alpha_interpolation_test,
    compute_class_centroid_and_variance,
    find_closest_data_point_to_centroid
)
from logger import setup_wandb, setup_alpha_csv_logging, log_alpha_test_results, log_to_wandb
import matplotlib.pyplot as plt

import wandb

def main():
    """
    Main training loop with comprehensive error handling and logging
    """
    print('Start session')
    wandb_run = None

    # Ignore warnings
    warnings.filterwarnings("ignore")
    torch.backends.cudnn.benchmark = True

    try:
        # Parse arguments and set initial configurations
        args = parse_args()
        set_seed(args.fix_seed)
        
        # Set device
        device = set_device(args.gpu)
        print(f'Using device: {device}')

        # Load datasets with error handling
        print('Loading datasets...')
        try:
            train_dataset, test_dataset, imagesize, num_classes, in_channels = load_datasets(
                args.dataset, args.target, args.gray_scale, args
            )
        except FileNotFoundError as e:
            print(f"Error loading dataset: {e}")
            return
        except Exception as e:
            print(f"Unexpected error loading dataset: {e}")
            return

        # Set number of digit and color classes
        if args.target == 'combined':
            num_digits = 10
            num_colors = 10
        else:
            num_digits = 10
            num_colors = 1

        # Add label noise and create NoisyDataset
        print(f'Preparing dataset with label noise rate: {args.label_noise_rate}')
        if args.label_noise_rate > 0.0:
            if hasattr(train_dataset, 'tensors'):
                x_train, y_train = train_dataset.tensors
                y_train_noisy, noise_info = add_label_noise(y_train, args.label_noise_rate, num_digits, num_colors)
                train_dataset = TensorDataset(x_train, y_train_noisy)
                train_dataset = NoisyDataset(train_dataset, noise_info)
            else:
                y_train = torch.tensor(train_dataset.targets)
                y_train_noisy, noise_info = add_label_noise(y_train, args.label_noise_rate, num_digits, num_colors)
                train_dataset.targets = y_train_noisy.tolist()
                train_dataset = NoisyDataset(train_dataset, noise_info)
        else:
            if hasattr(train_dataset, 'tensors'):
                x_train, y_train = train_dataset.tensors
                y_train_noisy = y_train.clone()
                noise_info = torch.zeros(len(train_dataset), dtype=torch.int)
                train_dataset = TensorDataset(x_train, y_train_noisy)
                train_dataset = NoisyDataset(train_dataset, noise_info)
            else:
                y_train = torch.tensor(train_dataset.targets)
                y_train_noisy = y_train.clone()
                noise_info = torch.zeros(len(train_dataset), dtype=torch.int)
                train_dataset.targets = y_train_noisy.tolist()

        # Extract indices for clean and noisy samples
        clean_indices = [i for i, label in enumerate(noise_info) if label == 0]
        noisy_indices = [i for i, label in enumerate(noise_info) if label == 1]

        if hasattr(train_dataset.dataset, 'tensors'):
            original_targets = y_train  # Before label noise
        else:
            original_targets = torch.tensor(y_train)  # Before label noise

        # Get noisy targets
        if hasattr(train_dataset.dataset, 'tensors'):
            noisy_targets = y_train_noisy
        else:
            noisy_targets = torch.tensor(train_dataset.dataset.targets)

        # ------------------- New Code Starts Here -------------------
        # Requirement 2: Compute centroids and variances for two labels
        n1 = 0  # First class label
        n2 = 10  # Second class label

        print(f"Computing centroids and variances for class labels {n1} and {n2}...")
        try:
            centroid1, variance1 = compute_class_centroid_and_variance(
                train_dataset.dataset,
                noisy_targets,
                class_label=n1
            )
            centroid2, variance2 = compute_class_centroid_and_variance(
                train_dataset.dataset,
                noisy_targets,
                class_label=n2
            )
            print(f"Centroids computed for class labels {n1} and {n2}")
            #print(f"{n1}_{variance1} |||||{n2}_{variance2}")
        except ValueError as e:
            print(e)
            return

        # Requirement 3: Find closest data points to the centroids
        mode1 = 'no_noise'  # 'no_noise' or 'noise'
        mode2 = 'noise'
        print(f"Finding closest data points to centroids for class labels {n1} and {n2} in mode '{mode1} and {mode2}'...")
        try:
            x1, idx1, distance1 = find_closest_data_point_to_centroid(
                centroid1,
                train_dataset.dataset,
                noisy_targets,
                class_label=n1,
                mode=mode1,
                noise_info=noise_info
            )
            x2, idx2, distance2 = find_closest_data_point_to_centroid(
                centroid2,
                train_dataset.dataset,
                noisy_targets,
                class_label=n1,
                mode=mode2,
                noise_info=noise_info
            )
            print(f"Closest data point indices: {idx1} (class {n1}), {idx2} (class {n2})")
            print(f"Distances to centroids: {distance1}, {distance2}")
        except ValueError as e:
            print(e)
            return

        # Requirement 4: Run alpha_interpolation_test using the two data points
        # Get labels for the data points
        label_x1 = noisy_targets[idx1].item()
        label_x2 = noisy_targets[idx2].item()

        digit_label_x1 = label_x1 // num_colors
        color_label_x1 = label_x1 % num_colors
        digit_label_x2 = label_x2 // num_colors
        color_label_x2 = label_x2 % num_colors

        combined_label_x1 = label_x1
        combined_label_x2 = label_x2

        # Set up alpha test logging
        experiment_name = f'{args.model}_{args.dataset}_variance{args.variance}_{args.target}_lr{args.lr}_batch{args.batch_size}_epoch{args.epoch}_LabelNoiseRate{args.label_noise_rate}_Optim{args.optimizer}_class{n1}_distance{distance1}_class{n2}_distance{distance2}{mode1}{mode2}'
        alpha_csv_path = setup_alpha_csv_logging(experiment_name, 0)  # Epoch 0 for initial test
        
        # Save the selected images
        save_dir = (os.path.join(alpha_csv_path,experiment_name))
        save_dir - os.path.join(save_dir,"image")
        os.makedirs(save_dir, exist_ok=True)
        torch.save(x1, os.path.join(save_dir, f"class_{n1}_data_point_{idx1}.pt"))
        torch.save(x2, os.path.join(save_dir, f"class_{n2}_data_point_{idx2}.pt"))
        
        #################################################################################
        img_tensor = torch.load(os.path.join(save_dir, f"class_{n1}_data_point_{idx1}.pt"))
        # もしGPU上のテンソルの場合はCPUへ移動
        img_tensor = img_tensor.cpu()
        img_np = img_tensor.permute(1, 2, 0).numpy()
        # matplotlibで画像表示
        plt.imshow(img_np)
        plt.axis('off')  # 軸を表示しない
        plt.savefig(f"class_{n1}_data_point_{idx1}.pt")


        img_tensor = torch.load(os.path.join(save_dir, f"class_{n2}_data_point_{idx2}.pt"))
        # もしGPU上のテンソルの場合はCPUへ移動
        img_tensor = img_tensor.cpu()
        # テンソルが(CHW)形式の場合、imshowには(H, W, C)形式が必要
        # 一般的な画像テンソルは(C, H, W)。
        # その場合、permute(1, 2, 0) で(H, W, C)に変換
        img_np = img_tensor.permute(1, 2, 0).numpy()
        # matplotlibで画像表示
        plt.imshow(img_np)
        plt.axis('off')  # 軸を表示しない
        plt.savefig(f"class_{n2}_data_point_{idx2}.pt")
        #################################################################################
        
        # Initialize model
        print('Initializing model...')
        model = load_models(in_channels, args, imagesize, num_classes)
        model = model.to(device)

        # Set optimizer
        print('Setting up optimizer...')
        if args.optimizer == "sgd":
            optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
        else:
            optimizer = optim.Adam(model.parameters(), lr=args.lr)

        # Set loss function
        criterion = nn.CrossEntropyLoss(reduction='none')

        # Initialize wandb
        if args.wandb:
            print('Initializing wandb...')
            wandb_run = setup_wandb(args, experiment_name)

        # Prepare data loaders (existing code)
        # Validate batch size
        if args.batch_size % 2 != 0:
            raise ValueError("Batch size must be even for balanced batches")

        # Create data loaders
        print('Setting up data loaders...')
        if args.label_noise_rate in [0.0, 1.0]:
            train_loader = DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.num_workers,
                pin_memory=True
            )
        else:
            batch_sampler = BalancedBatchSampler(
                clean_indices,
                noisy_indices,
                args.batch_size,
                drop_last=False
            )
            train_loader = DataLoader(
                train_dataset,
                batch_sampler=batch_sampler,
                num_workers=args.num_workers,
                pin_memory=True
            )

        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True
        )

        # ------------------- New Code Continues -------------------
        # Run alpha interpolation test before training
        print("Running alpha interpolation test before training...")
        alpha_logs = alpha_interpolation_test(
            model,
            x1,
            x2,
            digit_label_x1,
            digit_label_x2,
            color_label_x1,
            color_label_x2,
            combined_label_x1,
            combined_label_x2,
            num_digits,
            num_colors,
            device,
        )
        # Log alpha test results
        log_alpha_test_results(alpha_logs, alpha_csv_path)
        if args.wandb:
            log_data = {'epoch': epoch}  # Include epoch in log_data
            valid_alpha_values = np.arange(-0.50, 1.60, 0.1).tolist()
            tolerance = 1e-5  # Tolerance for matching alpha values

            if alpha_logs is not None:
                for i, alpha in enumerate(alpha_logs['alpha_values']):
                    if any(math.isclose(alpha, valid_alpha, abs_tol=tolerance) for valid_alpha in valid_alpha_values):
                        alpha_str = f"{alpha:.2f}"
                        # Use a nested dictionary for better organization
                        log_data.setdefault(f'alpha_{alpha_str}', {})
                        log_data[f'alpha_{alpha_str}'].update({
                            'digit_losses': alpha_logs['digit_losses'][i],
                            'color_losses': alpha_logs['color_losses'][i],
                            'combined_losses': alpha_logs['combined_losses'][i],
                            'predicted_digits': alpha_logs['predicted_digits'][i],
                            'predicted_colors': alpha_logs['predicted_colors'][i],
                        })

                        # Log digit probabilities
                        digit_probs = alpha_logs['digit_probabilities'][i]
                        for j, prob in enumerate(digit_probs):
                            log_data[f'alpha_{alpha_str}'][f'digit_probability_{j}'] = prob

                        # Log color probabilities
                        color_probs = alpha_logs['color_probabilities'][i]
                        for j, prob in enumerate(color_probs):
                            log_data[f'alpha_{alpha_str}'][f'color_probability_{j}'] = prob

            # Log the alpha test results to WandB with epoch
            log_to_wandb(wandb_run, log_data)
        # ------------------- End of New Code -------------------

        # Training loop (existing code)
        print('Starting training...')
        best_test_accuracy = 0.0
        for epoch in range(1, args.epoch + 1):
            try:
                # Training phase
                train_metrics = train_model(
                    model=model,
                    train_loader=train_loader,
                    optimizer=optimizer,
                    criterion=criterion,
                    weight_noisy=args.weight_noisy,
                    weight_clean=args.weight_clean,
                    device=device,
                    num_colors=num_colors,
                    num_digits=num_digits
                )

                # Testing phase
                test_metrics = test_model(
                    model=model,
                    test_loader=test_loader,
                    device=device,
                    num_colors=num_colors,
                    num_digits=num_digits
                )

                # Run alpha interpolation test at specified intervals
                if epoch % 10 == 0:
                    print(f"Running alpha interpolation test at epoch {epoch}...")
                    alpha_logs = alpha_interpolation_test(
                        model,
                        x1,
                        x2,
                        digit_label_x1,
                        digit_label_x2,
                        color_label_x1,
                        color_label_x2,
                        combined_label_x1,
                        combined_label_x2,
                        num_digits,
                        num_colors,
                        device,
                    )
                    # Log alpha test results
                    alpha_csv_path = setup_alpha_csv_logging(experiment_name, epoch)
                    log_alpha_test_results(alpha_logs, alpha_csv_path)

                    if args.wandb:
                        valid_alpha_values = np.arange(-0.50, 1.60, 0.1).tolist()
                        tolerance = 1e-5  # Tolerance for matching alpha values

                        if alpha_logs is not None:
                            for i, alpha in enumerate(alpha_logs['alpha_values']):
                                if any(math.isclose(alpha, valid_alpha, abs_tol=tolerance) for valid_alpha in valid_alpha_values):
                                    alpha_str = f"{alpha:.2f}"
                                    log_data[f'{alpha_str}/digit_losses'] = alpha_logs['digit_losses'][i]
                                    log_data[f'{alpha_str}/color_losses'] = alpha_logs['color_losses'][i]
                                    log_data[f'{alpha_str}/combined_losses'] = alpha_logs['combined_losses'][i]
                                    log_data[f'{alpha_str}/predicted_digits'] = alpha_logs['predicted_digits'][i]
                                    log_data[f'{alpha_str}/predicted_colors'] = alpha_logs['predicted_colors'][i]

                                    # Log digit probabilities
                                    digit_probs = alpha_logs['digit_probabilities'][i]
                                    for j, prob in enumerate(digit_probs):
                                        log_data[f'{alpha_str}/digit_probability_{j}'] = prob

                                    # Log color probabilities
                                    color_probs = alpha_logs['color_probabilities'][i]
                                    for j, prob in enumerate(color_probs):
                                        log_data[f'{alpha_str}/color_probability_{j}'] = prob

                # ------------------- Modification Starts Here -------------------
                # Log the specified metrics to WandB
                if args.wandb:
                    # Prepare log data
                    log_data = {
                        'epoch': epoch,
                        'train_loss': train_metrics['avg_loss'],
                        'train_loss_variance': train_metrics['var_loss'],
                        'train_accuracy': train_metrics['accuracy_total'],
                        'train_accuracy_noisy': train_metrics['accuracy_noisy'],
                        'train_accuracy_clean': train_metrics['accuracy_clean'],
                        'train_accuracy_digit_total': train_metrics['accuracy_digit_total'],
                        'train_accuracy_color_total': train_metrics['accuracy_color_total'],
                        'train_accuracy_digit_noisy': train_metrics['accuracy_digit_noisy'],
                        'train_accuracy_color_noisy': train_metrics['accuracy_color_noisy'],
                        'train_accuracy_digit_clean': train_metrics['accuracy_digit_clean'],
                        'train_accuracy_color_clean': train_metrics['accuracy_color_clean'],
                        'test_loss': test_metrics['avg_loss'],
                        'test_accuracy': test_metrics['accuracy_total'],
                        'test_accuracy_digit_total': test_metrics['accuracy_digit_total'],
                        'test_accuracy_color_total': test_metrics['accuracy_color_total']
                    }
                    # Log to WandB
                    log_to_wandb(wandb_run, log_data)
                # ------------------- Modification Ends Here -------------------

                # Print progress
                print(f"\nEpoch: {epoch}/{args.epoch}")
                print(f"Train Loss: {train_metrics['avg_loss']:.4f}, "
                      f"Train Loss Variance: {train_metrics['var_loss']:.4f}")
                print(f"Train Accuracy: {train_metrics['accuracy_total']:.2f}%, "
                      f"Test Accuracy: {test_metrics['accuracy_total']:.2f}%")
                print(f"Train Digit/Color Accuracy: {train_metrics['accuracy_digit_total']:.2f}%/"
                      f"{train_metrics['accuracy_color_total']:.2f}%")
                print(f"Test Digit/Color Accuracy: {test_metrics['accuracy_digit_total']:.2f}%/"
                      f"{test_metrics['accuracy_color_total']:.2f}%")

                # Save best model
                if test_metrics['accuracy_total'] > best_test_accuracy:
                    best_test_accuracy = test_metrics['accuracy_total']
                    torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))

            except Exception as e:
                print(f"Error in epoch {epoch}: {str(e)}")
                continue

    except Exception as e:
        print(f"Error in main: {e}")
        return 
    
    finally:
        # Cleanup
        if wandb_run:
            wandb_run.finish()
        print('Training completed')

if __name__ == '__main__':
    main()
