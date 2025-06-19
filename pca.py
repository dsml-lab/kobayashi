import numpy as np
import csv
import torch
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision import datasets
import os
import gzip
def apply_transform(data, transform):
    # Apply transformation to numpy array and convert to torch tensor
    transformed_data = [transform(image) for image in data]
    return torch.stack(transformed_data)

# PCA and visualization functions
def plot_cumulative_explained_variance(pca, args):
    seed = args.fix_seed
    variance = args.variance
    correlation = args.correlation

    # フォルダ作成
    output_dir = f"pca/{args.dataset}_{args.variance}/"
    os.makedirs(output_dir, exist_ok=True)

    # 累積寄与率を計算
    explained_variance = np.cumsum(pca.explained_variance_ratio_)

    # CSVへの保存
    csv_filename = os.path.join(output_dir, f'explained_variance_{args.dataset}_Seed{seed}_Var{variance}_Corr{correlation}.csv')
    with open(csv_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Component", "Cumulative_Explained_Variance"])
        for i, val in enumerate(explained_variance):
            writer.writerow([i+1, val])
    print(f"Saved cumulative explained variance to: {csv_filename}")

    # グラフの保存
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(explained_variance) + 1), explained_variance, marker='o', linestyle='--')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('PCA Cumulative Explained Variance')
    plt.grid(True)
    plt.show()
    plt.savefig(os.path.join(output_dir, f'kiyoritu_{args.dataset}_Seed{seed}_Var{variance}_Corr{correlation}.png'))


def plot_pca_2d(pca_transformed, labels, n, m, label_type, args):
    seed = args.fix_seed
    variance = args.variance
    correlation = args.correlation

    # フォルダ作成
    output_dir = "pca"
    os.makedirs(output_dir, exist_ok=True)

    # プロット
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(pca_transformed[:, n], pca_transformed[:, m], c=labels, cmap=plt.cm.tab10, alpha=0.7)
    plt.xlabel(f'Principal Component {n + 1}')
    plt.ylabel(f'Principal Component {m + 1}')
    plt.title(f'PCA 2D Plot (Components {n + 1} and {m + 1})')
    plt.colorbar(scatter, label=label_type)
    plt.grid(True)
    plt.show()
    plt.savefig(os.path.join(output_dir, f'pca_2d_{label_type}_Seed{seed}_Var{variance}_Corr{correlation}.png'))
# Load and process dataset
def load_datasets(dataset, target, gray_scale, args):
    """
    Load the specified dataset and apply transformations based on the dataset type and grayscale option.
    
    Args:
        dataset (str): The name of the dataset to load. Supported options are "mnist", "emnist", "cifar10", "cifar100", and "tinyImageNet".
        gray_scale (bool): Flag indicating whether to convert the images to grayscale.
        
    Returns:
        tuple: A tuple containing the train dataset, test dataset, image size, and number of classes.
    """
    if dataset == "mnist":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((32, 32)),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        imagesize = (32, 32)
        num_classes = 10
        in_channels = 1
    elif dataset == "emnist":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((32, 32)),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_dataset = datasets.EMNIST(root='./data', split='balanced', train=True, download=True, transform=transform)
        test_dataset = datasets.EMNIST(root='./data', split='balanced', train=False, download=True, transform=transform)
        imagesize = (32, 32)
        num_classes = 47
        in_channels = 1
    elif dataset == "emnist_digits":
        emnist_path = './data/EMNIST'
        def load_gz_file(file_path, is_image=True):
            with gzip.open(file_path, 'rb') as f:
                if is_image:
                    return np.frombuffer(f.read(), dtype=np.uint8, offset=16).reshape(-1, 28, 28)
                else:
                    return np.frombuffer(f.read(), dtype=np.uint8, offset=8)

        x_train = load_gz_file(os.path.join(emnist_path, 'emnist-digits-train-images-idx3-ubyte.gz'))
        y_train = load_gz_file(os.path.join(emnist_path, 'emnist-digits-train-labels-idx1-ubyte.gz'), is_image=False)
        x_test = load_gz_file(os.path.join(emnist_path, 'emnist-digits-test-images-idx3-ubyte.gz'))
        y_test = load_gz_file(os.path.join(emnist_path, 'emnist-digits-test-labels-idx1-ubyte.gz'), is_image=False)
        # 変換関数が必要な場合はここで定義
        transform = transforms.Compose([
            transforms.ToPILImage(),  # Convert numpy array to PIL Image
            transforms.Resize((32, 32)),  # Same size as original, adjust if needed
            transforms.ToTensor()
        ])

        # Apply transformation
        x_train_tensor = apply_transform(x_train, transform)
        x_test_tensor = apply_transform(x_test, transform)

        y_train_tensor = torch.tensor(y_train, dtype=torch.long)
        y_test_tensor = torch.tensor(y_test, dtype=torch.long)

        train_dataset = torch.utils.data.TensorDataset(x_train_tensor, y_train_tensor)
        test_dataset = torch.utils.data.TensorDataset(x_test_tensor, y_test_tensor)

        num_classes = 10  # Digits from 0 to 9
        in_channels = 1  # Grayscale images
        imagesize = (32, 32)  # Original image size
    elif dataset == "colored_emnist":
        # target: color or digit or combined
        
        # Data augmentation
        transform = transforms.Compose([
            transforms.ToPILImage(),  # Convert numpy array to PIL Image
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        if target == 'color':
            x_train = np.load('data/colored_EMNIST/x_train_colored.npy')
            y_train_colors = np.load('data/colored_EMNIST/y_train_colors.npy')
            x_test = np.load('data/colored_EMNIST/x_test_colored.npy')
            y_test_colors = np.load('data/colored_EMNIST/y_test_colors.npy')
            
            # Apply transformation
            x_train_tensor = apply_transform(x_train, transform)
            x_test_tensor = apply_transform(x_test, transform)
            
            y_train_tensor = torch.tensor(y_train_colors, dtype=torch.long)
            y_test_tensor = torch.tensor(y_test_colors, dtype=torch.long)
            
            train_dataset = torch.utils.data.TensorDataset(x_train_tensor, y_train_tensor)
            test_dataset = torch.utils.data.TensorDataset(x_test_tensor, y_test_tensor)
        
        elif target == 'digit':
            x_train = np.load('data/colored_EMNIST/x_train_colored.npy')
            y_train_digits = np.load('data/colored_EMNIST/y_train_digits.npy')
            x_test = np.load('data/colored_EMNIST/x_test_colored.npy')
            y_test_digits = np.load('data/colored_EMNIST/y_test_digits.npy')
            
            # Apply transformation
            x_train_tensor = apply_transform(x_train, transform)
            x_test_tensor = apply_transform(x_test, transform)
            
            y_train_tensor = torch.tensor(y_train_digits, dtype=torch.long)
            y_test_tensor = torch.tensor(y_test_digits, dtype=torch.long)
            
            train_dataset = torch.utils.data.TensorDataset(x_train_tensor, y_train_tensor)
            test_dataset = torch.utils.data.TensorDataset(x_test_tensor, y_test_tensor)
            
        elif target == 'combined':
            x_train = np.load('data/colored_EMNIST/x_train_colored.npy')
            y_train_combined = np.load('data/colored_EMNIST/y_train_combined.npy')
            x_test = np.load('data/colored_EMNIST/x_test_colored.npy')
            y_test_combined = np.load('data/colored_EMNIST/y_test_combined.npy')
            
            # Apply transformation
            x_train_tensor = apply_transform(x_train, transform)
            x_test_tensor = apply_transform(x_test, transform)
            
            y_train_tensor = torch.tensor(y_train_combined, dtype=torch.long)
            y_test_tensor = torch.tensor(y_test_combined, dtype=torch.long)
            
            train_dataset = torch.utils.data.TensorDataset(x_train_tensor, y_train_tensor)
            test_dataset = torch.utils.data.TensorDataset(x_test_tensor, y_test_tensor)
            
        num_classes = 10 if target in ['color', 'digit'] else 100
        in_channels = 3
        imagesize = (32, 32)
    elif dataset == "distribution_colored_emnist":
        # target: color or digit or combined
        seed = args.fix_seed
        variance = args.variance
        correlation = args.correlation
        # Data augmentation
        print(f"loading_dataset_dis_colored_emnist_{variance}")
        transform = transforms.Compose([
            transforms.ToPILImage(),  # Convert numpy array to PIL Image
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        if target == 'color':
            x_train = np.load(f'data/distribution_colored_EMNIST_Seed{seed}_Var{variance}_Corr{correlation}/x_train_colored.npy')
            y_train_colors = np.load(f'data/distribution_colored_EMNIST_Seed{seed}_Var{variance}_Corr{correlation}/y_train_colors.npy')
            x_test = np.load(f'data/distribution_colored_EMNIST_Seed{seed}_Var{variance}_Corr{correlation}/x_test_colored.npy')
            y_test_colors = np.load(f'data/distribution_colored_EMNIST_Seed{seed}_Var{variance}_Corr{correlation}/y_test_colors.npy')
            
            # Apply transformation
            x_train_tensor = apply_transform(x_train, transform)
            x_test_tensor = apply_transform(x_test, transform)
            
            y_train_tensor = torch.tensor(y_train_colors, dtype=torch.long)
            y_test_tensor = torch.tensor(y_test_colors, dtype=torch.long)
            
            train_dataset = torch.utils.data.TensorDataset(x_train_tensor, y_train_tensor)
            test_dataset = torch.utils.data.TensorDataset(x_test_tensor, y_test_tensor)
        
        elif target == 'digit':
            x_train = np.load(f'data/distribution_colored_EMNIST_Seed{seed}_Var{variance}_Corr{correlation}/x_train_colored.npy')
            y_train_digits = np.load(f'data/distribution_colored_EMNIST_Seed{seed}_Var{variance}_Corr{correlation}/y_train_digits.npy')
            x_test = np.load(f'data/distribution_colored_EMNIST_Seed{seed}_Var{variance}_Corr{correlation}/x_test_colored.npy')
            y_test_digits = np.load(f'data/distribution_colored_EMNIST_Seed{seed}_Var{variance}_Corr{correlation}/y_test_digits.npy')
            
            # Apply transformation
            x_train_tensor = apply_transform(x_train, transform)
            x_test_tensor = apply_transform(x_test, transform)
            
            y_train_tensor = torch.tensor(y_train_digits, dtype=torch.long)
            y_test_tensor = torch.tensor(y_test_digits, dtype=torch.long)
            
            train_dataset = torch.utils.data.TensorDataset(x_train_tensor, y_train_tensor)
            test_dataset = torch.utils.data.TensorDataset(x_test_tensor, y_test_tensor)
            
        elif target == 'combined':
            x_train = np.load(f'data/distribution_colored_EMNIST_Seed{seed}_Var{variance}_Corr{correlation}/x_train_colored.npy')
            y_train_combined = np.load(f'data/distribution_colored_EMNIST_Seed{seed}_Var{variance}_Corr{correlation}/y_train_combined.npy')
            x_test = np.load(f'data/distribution_colored_EMNIST_Seed{seed}_Var{variance}_Corr{correlation}/x_test_colored.npy')
            y_test_combined = np.load(f'data/distribution_colored_EMNIST_Seed{seed}_Var{variance}_Corr{correlation}/y_test_combined.npy')
            
            # Apply transformation
            x_train_tensor = apply_transform(x_train, transform)
            x_test_tensor = apply_transform(x_test, transform)
            
            y_train_tensor = torch.tensor(y_train_combined, dtype=torch.long)
            y_test_tensor = torch.tensor(y_test_combined, dtype=torch.long)
            
            train_dataset = torch.utils.data.TensorDataset(x_train_tensor, y_train_tensor)
            test_dataset = torch.utils.data.TensorDataset(x_test_tensor, y_test_tensor)
    
        num_classes = 10 if target in ['color', 'digit'] else 100
        in_channels = 3
        imagesize = (32, 32)
    elif dataset == "cifar10":
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        imagesize = (32, 32)
        num_classes = 10
        in_channels = 3
    elif dataset == "cifar100":
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
        train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
        imagesize = (32, 32)
        num_classes = 100
        in_channels = 3
    elif dataset == "tinyImageNet":
        transform = transforms.Compose([
            transforms.RandomCrop(64, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262))
        ])
        train_dataset = datasets.ImageFolder(root='./data/tiny-imagenet-200/train', transform=transform)
        test_dataset = datasets.ImageFolder(root='./data/tiny-imagenet-200/val', transform=transform)
        imagesize = (64, 64)
        num_classes = 200
        in_channels = 3
    elif dataset == "distribution_to_normal":
        # target: color or digit or combined
        seed = args.fix_seed
        variance = args.variance
        correlation = args.correlation
        # Data augmentation
        transform = transforms.Compose([
            transforms.ToPILImage(),  # Convert numpy array to PIL Image
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        if target == 'combined':
            x_train = np.load(f'data/distribution_colored_EMNIST_Seed{seed}_Var{variance}_Corr{correlation}/x_train_colored.npy')
            y_train_combined = np.load(f'data/distribution_colored_EMNIST_Seed{seed}_Var{variance}_Corr{correlation}/y_train_combined.npy')
            x_test = np.load('data/colored_EMNIST/x_test_colored.npy')
            y_test_combined = np.load('data/colored_EMNIST/y_test_combined.npy')
            
            # Apply transformation
            x_train_tensor = apply_transform(x_train, transform)
            x_test_tensor = apply_transform(x_test, transform)
            
            y_train_tensor = torch.tensor(y_train_combined, dtype=torch.long)
            y_test_tensor = torch.tensor(y_test_combined, dtype=torch.long)
            
            train_dataset = torch.utils.data.TensorDataset(x_train_tensor, y_train_tensor)
            test_dataset = torch.utils.data.TensorDataset(x_test_tensor, y_test_tensor)
    
        num_classes = 10 if target in ['color', 'digit'] else 100
        in_channels = 3
        imagesize = (32, 32)
    else:
        raise ValueError("Invalid dataset name")
        

    if gray_scale:
        transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_dataset.transform = transform
        test_dataset.transform = transform  

    return train_dataset, test_dataset, imagesize, num_classes, in_channels
def plot_examples_pca_reconstructed(x, y_digits, y_colors, dataset_name, n_comp, args):
    # フォルダ作成
    output_dir = "pca"
    os.makedirs(output_dir, exist_ok=True)

    # 画像プロット
    plt.figure(figsize=(15, 15))
    for digit in range(10):
        for color in range(10):
            index = np.where((y_digits == digit) & (y_colors == color))[0][0]
            plt.subplot(10, 10, digit * 10 + color + 1)
            plt.imshow(np.transpose(x[index], (1, 2, 0)))
            plt.axis('off')
            plt.title(f'{digit}-{color}', fontsize=8, pad=3)
    plt.suptitle(f'{dataset_name} Dataset Examples (PCA Reconstructed)', fontsize=16)
    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    plt.show()
    plt.savefig(os.path.join(output_dir, f'pca_reconstructed_{args.dataset}_Var{args.variance}_n{n_comp}.png'))

if __name__ == "__main__":
    # Arguments
    class Args:
        fix_seed = 42
        variance = 0
        correlation = 0.5
        target = "digit"
        dataset = "emnist_digits"
        gray_scale=True
    args = Args()

    # Load dataset
    train_dataset, imagesize, num_classes, in_channels = load_datasets(
        dataset=args.dataset, target=args.target, gray_scale=args.gray_scale, args=args
    )
    x_train = np.array([data[0].numpy().flatten() for data in train_dataset])
    y_train = np.array([data[1].item() for data in train_dataset])

    # Apply PCA
    n_comp = 10
    pca = PCA(n_components=n_comp)
    pca_transformed = pca.fit_transform(x_train)
    flatten_data_reconstructed = pca.inverse_transform(pca_transformed)
    x_train_pca_reconstructed = flatten_data_reconstructed.reshape(-1, 3, 32, 32)

    # Plot examples
    y_train_digits = y_train // 10
    y_train_colors = y_train % 10
    plot_examples_pca_reconstructed(x_train_pca_reconstructed, y_train_digits, y_train_colors, 'Train', n_comp, args)

    # Plot cumulative explained variance
    plot_cumulative_explained_variance(pca, args)

    # Plot 2D PCA projection
    plot_pca_2d(pca_transformed, y_train, n=0, m=1, label_type=args.target, args=args)