import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

def convert_variance(variance):
    if variance == "N/A":
        return variance
    return f"{float(variance) / (255.0 ** 2):.6f}"

def generate_accuracy_image(df, output_path):
    epochs = df['epoch']
    train_combined_acc = df[' train_combined_acc']
    test_combined_acc = df[' test_combined_acc']
    train_digit_acc = df[' train_digit_acc']
    test_digit_acc = df[' test_digit_acc']
    train_color_acc = df[' train_color_acc']
    test_color_acc = df[' test_color_acc']

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.plot(epochs, train_combined_acc, color='blue', label='Train Combined')
    ax.plot(epochs, test_combined_acc, color='blue', linestyle='dashed', label='Test Combined')
    ax.plot(epochs, train_digit_acc, color='green', label='Train Digit')
    ax.plot(epochs, test_digit_acc, color='green', linestyle='dashed', label='Test Digit')
    ax.plot(epochs, train_color_acc, color='red', label='Train Color')
    ax.plot(epochs, test_color_acc, color='red', linestyle='dashed', label='Test Color')
    ax.set_xscale('log')
    ax.set_xlim(1, 1000)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.grid(True)
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def generate_loss_image(df, output_path):
    epochs = df['epoch']
    train_loss = df[' train_loss']
    test_loss = df[' test_loss']

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.plot(epochs, train_loss, color='purple', label='Train Loss')
    ax.plot(epochs, test_loss, color='purple', linestyle='dashed', label='Test Loss')
    ax.set_xscale('log')
    ax.set_xlim(1, 1000)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.grid(True)
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def generate_images(file_path, output_dir, model, lnr, var):
    df = pd.read_csv(file_path)

    if var == "N/A":
        base_filename = f"{model}_lnr{lnr}"
    else:
        base_filename = f"{model}_var{var}_lnr{lnr}"

    acc_output_path = os.path.join(output_dir, f"{base_filename}_acc.png")
    loss_output_path = os.path.join(output_dir, f"{base_filename}_loss.png")

    generate_accuracy_image(df, acc_output_path)
    generate_loss_image(df, loss_output_path)

    print(f"Saved: {acc_output_path}")
    print(f"Saved: {loss_output_path}")

def compile_images(base_dir, models, label_noise_rates, variances):
    for model in models:
        model_dir = os.path.join(base_dir, f"{model}_results")
        os.makedirs(model_dir, exist_ok=True)

        for lnr in label_noise_rates:
            for var in variances:
                if var == "N/A":
                    pattern = f"{base_dir}/{model}_colored_emnist_combined_*_LabelNoiseRate{lnr}*/log.csv"
                else:
                    pattern = f"{base_dir}/{model}_distribution_colored_emnist_Variance{var}_*_LabelNoiseRate{lnr}*/log.csv"
                
                matching_files = glob.glob(pattern)
                if matching_files:
                    generate_images(matching_files[0], model_dir, model, lnr, var)

# スクリプトのメイン部分
base_dir = "/workspace/csv/combine"
models = ["cnn_2layers", "cnn_5layers", "resnet18"]
label_noise_rates = ["0.0", "0.1", "0.2", "0.4", "0.8"]
variances = ["N/A", "100", "1000", "10000", "100000"]  # N/A for non-distribution models

compile_images(base_dir, models, label_noise_rates, variances)