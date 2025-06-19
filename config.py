# config.py

import argparse
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
def parse_args():
    """
    Parse the command-line arguments.
    
    Returns:
        argparse.Namespace: The parsed command-line arguments.
    """
    arg_parser = argparse.ArgumentParser(description="PyTorch Training Script")
    
    # Set seed
    arg_parser.add_argument("-seed", "--fix_seed", type=int, default=42, help="Random seed for reproducibility")
    
    # Model settings
    arg_parser.add_argument("--model", type=str, choices=[
        "cnn_2layers", "cnn_3layers", "cnn_4layers",
        "cnn_5layers", "cnn_8layers", "cnn_16layers", "resnet18"
    ], required=True, help="Model architecture to use")
    arg_parser.add_argument("-model_width", "--model_width", type=int, default=1, help="Width multiplier for the model")
    arg_parser.add_argument("-epoch", "--epoch", type=int, default=1000, help="Number of training epochs")
    
    # Dataset settings
    arg_parser.add_argument("-datasets", "--dataset", type=str, choices=[
        "mnist", "emnist", "emnist_digits", "cifar10", "cifar100",
        "tinyImageNet", "colored_emnist", "distribution_colored_emnist"
    ], default="cifar10", help="Dataset to use")
    arg_parser.add_argument("-variance", "--variance", type=int, default=10000, help="Variance parameter for distribution datasets")
    arg_parser.add_argument("-correlation", "--correlation", type=float, default=0.5, help="Correlation parameter for distribution datasets")
    arg_parser.add_argument("-label_noise_rate", "--label_noise_rate", type=float, default=0.0, help="Label noise rate")
    arg_parser.add_argument("-gray_scale", "--gray_scale", action='store_true', help="Convert images to grayscale")
    arg_parser.add_argument("-batch_size", "--batch_size", type=int, default=128, help="Batch size for training")
    arg_parser.add_argument("-img_size", "--img_size", type=int, default=32, help="Image size")
    arg_parser.add_argument("-target", "--target", type=str, choices=["color", "digit", "combined"], default='color', help="Target for colored EMNIST")
    
    # Optimizer settings
    arg_parser.add_argument("-lr", "--lr", type=float, default=0.1, help="Learning rate")
    arg_parser.add_argument("-optimizer", "--optimizer", type=str, choices=["sgd", "adam"], default="adam", help="Optimizer to use")
    arg_parser.add_argument("-momentum", "--momentum", type=float, default=0.9, help="Momentum for SGD")
    
    # Loss function settings
    arg_parser.add_argument("-loss", "--loss", type=str, choices=["cross_entropy", "focal_loss"], default="cross_entropy", help="Loss function")
    
    # Device settings
    arg_parser.add_argument("-gpu", "--gpu", type=int, default=0, help="GPU device ID")
    arg_parser.add_argument("-num_workers", "--num_workers", type=int, default=4, help="Number of data loader workers")
    
    # WandB settings
    arg_parser.add_argument("-wandb", "--wandb", action='store_true', default=True, help="Use Weights & Biases for logging")
    arg_parser.add_argument("-wandb_project", "--wandb_project", type=str, default="dd_scratch_models", help="WandB project name")
    arg_parser.add_argument("--wandb_entity", type=str, default="dsml-kernel24", help="WandB entity name")
    
    # Loss weighting
    arg_parser.add_argument("-weight_noisy", "--weight_noisy", type=float, default=1.0, help="Weight for noisy samples")
    arg_parser.add_argument("-weight_clean", "--weight_clean", type=float, default=1.0, help="Weight for clean samples")
    
    return arg_parser.parse_args()

def parse_args_save_clo():
    """
    Parse the command-line arguments.
    
    Returns:
        argparse.Namespace: The parsed command-line arguments.
    """
    arg_parser = argparse.ArgumentParser(description="PyTorch Training Script")
    
    # Set seed
    arg_parser.add_argument("-fix_seed", "--fix_seed", type=int, default=42, help="Random seed for reproducibility")
    
    # Model settings
    arg_parser.add_argument("--model", type=str, choices=[
        "cnn_2layers", "cnn_3layers", "cnn_4layers",
        "cnn_5layers", "cnn_8layers", "cnn_16layers", "resnet18","cnn_5layers_cus"
    ], required=True, help="Model architecture to use")
    arg_parser.add_argument("-model_width", "--model_width", type=int, default=1, help="Width multiplier for the model")
    arg_parser.add_argument("-epoch", "--epoch", type=int, default=1000, help="Number of training epochs")
    
    # Dataset settings
    arg_parser.add_argument("-datasets", "--dataset", type=str, choices=[
        "mnist", "emnist", "emnist_digits", "cifar10", "cifar100",
        "tinyImageNet", "colored_emnist", "distribution_colored_emnist"
    ], default="cifar10", help="Dataset to use")
    arg_parser.add_argument("-variance", "--variance", type=int, default=10000, help="Variance parameter for distribution datasets")
    arg_parser.add_argument("-correlation", "--correlation", type=float, default=0.5, help="Correlation parameter for distribution datasets")
    arg_parser.add_argument("-label_noise_rate", "--label_noise_rate", type=float, default=0.0, help="Label noise rate")
    arg_parser.add_argument("-gray_scale", "--gray_scale", action='store_true', help="Convert images to grayscale")
    arg_parser.add_argument("-batch_size", "--batch_size", type=int, default=128, help="Batch size for training")
    arg_parser.add_argument("-img_size", "--img_size", type=int, default=32, help="Image size")
    arg_parser.add_argument("-target", "--target", type=str, choices=["color", "digit", "combined"], default='color', help="Target for colored EMNIST")
    
    # Optimizer settings
    arg_parser.add_argument("-lr", "--lr", type=float, default=0.1, help="Learning rate")
    arg_parser.add_argument("-optimizer", "--optimizer", type=str, choices=["sgd", "adam"], default="adam", help="Optimizer to use")
    arg_parser.add_argument("-momentum", "--momentum", type=float, default=0.9, help="Momentum for SGD")
    
    # Loss function settings
    arg_parser.add_argument("-loss", "--loss", type=str, choices=["cross_entropy", "focal_loss"], default="cross_entropy", help="Loss function")
    
    arg_parser.add_argument("-n1","--n1",type=int,default=0)
    arg_parser.add_argument("-n2","--n2",type=int,default=0)
    arg_parser.add_argument("-mode1","--mode1",type=str,default="no_noise")
    arg_parser.add_argument("-mode2","--mode2",type=str,default="no_noise")


    # Device settings
    arg_parser.add_argument("-gpu", "--gpu", type=int, default=0, help="GPU device ID")
    arg_parser.add_argument("-num_workers", "--num_workers", type=int, default=4, help="Number of data loader workers")
    
    # WandB settings
    arg_parser.add_argument("-wandb", "--wandb", action='store_true', default=True, help="Use Weights & Biases for logging")
    arg_parser.add_argument("-wandb_project", "--wandb_project", type=str, default="dd_scratch_models", help="WandB project name")
    arg_parser.add_argument("--wandb_entity", type=str, default="dsml-kernel24", help="WandB entity name")
    
    # Loss weighting
    arg_parser.add_argument("-weight_noisy", "--weight_noisy", type=float, default=1.0, help="Weight for noisy samples")
    arg_parser.add_argument("-weight_clean", "--weight_clean", type=float, default=1.0, help="Weight for clean samples")
        
    
    return arg_parser.parse_args()


def parse_args_save():
    """
    Parse the command-line arguments.
    
    Returns:
        argparse.Namespace: The parsed command-line arguments.
    """
    arg_parser = argparse.ArgumentParser(description="PyTorch Training Script")
    
    # Set seed
    arg_parser.add_argument("-seed", "--fix_seed", type=int, default=42, help="Random seed for reproducibility")
    
    # Model settings
    arg_parser.add_argument("--model", type=str, choices=[
        "cnn_2layers", "cnn_3layers", "cnn_4layers",
        "cnn_5layers", "cnn_8layers", "cnn_16layers", "resnet18","resnet18k"
    ], required=True, help="Model architecture to use")
    arg_parser.add_argument("-model_width", "--model_width", type=int, default=1, help="Width multiplier for the model")
    arg_parser.add_argument("-epoch", "--epoch", type=int, default=1000, help="Number of training epochs")
    
    # Dataset settings
    arg_parser.add_argument("-datasets", "--dataset", type=str, choices=[
        "mnist", "emnist", "emnist_digits", "cifar10", "cifar100",
        "tinyImageNet", "colored_emnist", "distribution_colored_emnist"
    ], default="cifar10", help="Dataset to use")
    arg_parser.add_argument("-variance", "--variance", type=int, default=10000, help="Variance parameter for distribution datasets")
    arg_parser.add_argument("-correlation", "--correlation", type=float, default=0.5, help="Correlation parameter for distribution datasets")
    arg_parser.add_argument("-label_noise_rate", "--label_noise_rate", type=float, default=0.0, help="Label noise rate")
    arg_parser.add_argument("-gray_scale", "--gray_scale", action='store_true', help="Convert images to grayscale")
    arg_parser.add_argument("-batch_size", "--batch_size", type=int, default=128, help="Batch size for training")
    arg_parser.add_argument("-img_size", "--img_size", type=int, default=32, help="Image size")
    arg_parser.add_argument("-target", "--target", type=str, choices=["color", "digit", "combined"], default='combined', help="Target for colored EMNIST")
    
    # Optimizer settings
    arg_parser.add_argument("-lr", "--lr", type=float, default=0.1, help="Learning rate")
    arg_parser.add_argument("-optimizer", "--optimizer", type=str, choices=["sgd", "adam"], default="adam", help="Optimizer to use")
    arg_parser.add_argument("-momentum", "--momentum", type=float, default=0.9, help="Momentum for SGD")
    
    # Loss function settings
    arg_parser.add_argument("-loss", "--loss", type=str, choices=["cross_entropy", "focal_loss"], default="cross_entropy", help="Loss function")
    
    arg_parser.add_argument("-n1","--n1",type=int,default=0)
    arg_parser.add_argument("-n2","--n2",type=int,default=0)
    arg_parser.add_argument("-mode1","--mode1",type=str,default="no_noise")
    arg_parser.add_argument("-mode2","--mode2",type=str,default="no_noise")


    # Device settings
    arg_parser.add_argument("-gpu", "--gpu", type=int, default=0, help="GPU device ID")
    arg_parser.add_argument("-num_workers", "--num_workers", type=int, default=4, help="Number of data loader workers")
    
    # WandB settings
    arg_parser.add_argument("-wandb", "--wandb", action='store_true', default=True, help="Use Weights & Biases for logging")
    arg_parser.add_argument("-wandb_project", "--wandb_project", type=str, default="dd_scratch_models", help="WandB project name")
    arg_parser.add_argument("--wandb_entity", type=str, default="dsml-kernel24", help="WandB entity name")
    
    # Loss weighting
    arg_parser.add_argument("-weight_noisy", "--weight_noisy", type=float, default=1.0, help="Weight for noisy samples")
    arg_parser.add_argument("-weight_clean", "--weight_clean", type=float, default=1.0, help="Weight for clean samples")
    
    return arg_parser.parse_args()


def parse_args_model_save():
    """
    Parse the command-line arguments.
    
    Returns:
        argparse.Namespace: The parsed command-line arguments.
    """
    arg_parser = argparse.ArgumentParser(description="PyTorch Training Script")
    
    # Set seed
    arg_parser.add_argument("-seed", "--fix_seed", type=int, default=42, help="Random seed for reproducibility")
    
    # Model settings
    arg_parser.add_argument("--model", type=str, choices=[
        "cnn_2layers", "cnn_3layers", "cnn_4layers",
        "cnn_5layers", "cnn_8layers", "cnn_16layers", "resnet18","resnet18k","cnn_5layers_cus"
    ], required=True, help="Model architecture to use")
    arg_parser.add_argument("-model_width", "--model_width", type=int, default=1, help="Width multiplier for the model")
    arg_parser.add_argument("-epoch", "--epoch", type=int, default=1000, help="Number of training epochs")
    
    # Dataset settings
    arg_parser.add_argument("-datasets", "--dataset", type=str, choices=[
        "mnist", "emnist", "emnist_digits", "cifar10", "cifar100",
        "tinyImageNet", "colored_emnist", "distribution_colored_emnist"
    ], default="cifar10", help="Dataset to use")
    arg_parser.add_argument("-variance", "--variance", type=int, default=0, help="Variance parameter for distribution datasets")
    arg_parser.add_argument("-correlation", "--correlation", type=float, default=0.5, help="Correlation parameter for distribution datasets")
    arg_parser.add_argument("-label_noise_rate", "--label_noise_rate", type=float, default=0.0, help="Label noise rate")
    arg_parser.add_argument("-gray_scale", "--gray_scale", action='store_true', help="Convert images to grayscale")
    arg_parser.add_argument("-batch_size", "--batch_size", type=int, default=128, help="Batch size for training")
    arg_parser.add_argument("-img_size", "--img_size", type=int, default=32, help="Image size")
    arg_parser.add_argument("-target", "--target", type=str, choices=["color", "digit", "combined"], default='color', help="Target for colored EMNIST")
    
    # Optimizer settings
    arg_parser.add_argument("-lr", "--lr", type=float, default=0.1, help="Learning rate")
    arg_parser.add_argument("-optimizer", "--optimizer", type=str, choices=["sgd", "adam"], default="adam", help="Optimizer to use")
    arg_parser.add_argument("-momentum", "--momentum", type=float, default=0.9, help="Momentum for SGD")
    
    # Loss function settings
    arg_parser.add_argument("-loss", "--loss", type=str, choices=["cross_entropy", "focal_loss"], default="cross_entropy", help="Loss function")
    
    # Device settings
    arg_parser.add_argument("-gpu", "--gpu", type=int, default=0, help="GPU device ID")
    arg_parser.add_argument("-num_workers", "--num_workers", type=int, default=4, help="Number of data loader workers")
    
    # WandB settings
    arg_parser.add_argument("-wandb", "--wandb", action='store_true', default=True, help="Use Weights & Biases for logging")
    arg_parser.add_argument("-wandb_project", "--wandb_project", type=str, default="dd_scratch_models", help="WandB project name")
    arg_parser.add_argument("--wandb_entity", type=str, default="dsml-kernel24", help="WandB entity name")
    
    # Loss weighting
    arg_parser.add_argument("-weight_noisy", "--weight_noisy", type=float, default=1.0, help="Weight for noisy samples")
    arg_parser.add_argument("-weight_clean", "--weight_clean", type=float, default=1.0, help="Weight for clean samples")
    arg_parser.add_argument('--use_saved_data', type=str2bool, default=True, help='Use pre-saved data') 
    return arg_parser.parse_args()