from train import (
    train_model_standard,
    train_model_distr_colored,
    train_model_0_epoch_standard,
    train_model_0_epoch_distr_colored,
    test_model_standard,
    test_model_distr_colored
)

def train_model_dispatch(model, train_loader, optimizer, criterion_noisy, criterion_clean,
                         weight_noisy, weight_clean, args, device,
                         num_colors=None, num_digits=None,
                         epoch_batch=None, experiment_name=None):
    if "distribution_colored_emnist" in args.dataset.lower():
        return train_model_distr_colored(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            criterion_noisy=criterion_noisy,
            criterion_clean=criterion_clean,
            weight_noisy=weight_noisy,
            weight_clean=weight_clean,
            device=device,
            num_colors=num_colors,
            num_digits=num_digits
        )
    else:
        return train_model_standard(
    model=model,
    train_loader=train_loader,
    optimizer=optimizer,
    criterion_noisy=criterion_noisy,
    criterion_clean=criterion_clean,
    weight_noisy=weight_noisy,
    weight_clean=weight_clean,
    device=device,
    epoch_batch=epoch_batch,
    experiment_name=experiment_name,
    args=args
    )

def train_model_0_epoch_dispatch(model, train_loader, optimizer, criterion_noisy, criterion_clean,
                                 weight_noisy, weight_clean, args, device, num_colors=None, num_digits=None):
    if "distribution_colored_emnist" in args.dataset.lower():
        return train_model_0_epoch_distr_colored(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            criterion_noisy=criterion_noisy,
            criterion_clean=criterion_clean,
            weight_noisy=weight_noisy,
            weight_clean=weight_clean,
            device=device,
            num_colors=num_colors,
            num_digits=num_digits
        )
    else:
        return train_model_0_epoch_standard(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            criterion_noisy=criterion_noisy,
            criterion_clean=criterion_clean,
            device=device
        )

def test_model_dispatch(model, test_loader, args, device, num_colors=None, num_digits=None):
    if "distribution_colored_emnist" in args.dataset.lower():
        return test_model_distr_colored(
            model=model,
            test_loader=test_loader,
            device=device,
            num_colors=num_colors,
            num_digits=num_digits
        )
    else:
        return test_model_standard(
            model=model,
            test_loader=test_loader,
            device=device
        )
