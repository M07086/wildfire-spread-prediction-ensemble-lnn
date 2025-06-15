"""
Training script for Part I wildfire prediction models.
"""
from src.part1 import (
    train_file_pattern, val_file_pattern, test_file_pattern,
    prepare_datasets, build_convnet_model, build_multi_kernel_cnn_model,
    build_unet_model, build_single_lnn_model, build_ensemble_lnn_simple,
    train_model, evaluate_model, summarize_and_plot, DEFAULT_BATCH_SIZE
)

def main():
    # Prepare datasets
    conv_train_ds, conv_val_ds, conv_test_ds = prepare_datasets(
        train_file_pattern, val_file_pattern, test_file_pattern,
        batch_size=DEFAULT_BATCH_SIZE, ensemble_flag=False
    )
    lnn_train_ds, lnn_val_ds, lnn_test_ds = prepare_datasets(
        train_file_pattern, val_file_pattern, test_file_pattern,
        batch_size=DEFAULT_BATCH_SIZE, ensemble_flag=False
    )
    ens_train_ds, ens_val_ds, ens_test_ds = prepare_datasets(
        train_file_pattern, val_file_pattern, test_file_pattern,
        batch_size=DEFAULT_BATCH_SIZE, ensemble_flag=True, filter_fire=True
    )

    # Part I models
    models = {
        'MultiKernelCNN': build_multi_kernel_cnn_model(),
        'UNet': build_unet_model(),
        'ConvNet_WBCE': build_convnet_model('wbce'),
        'ConvNet_Focal': build_convnet_model('focal'),
        'SingleLNN_WBCE': build_single_lnn_model('wbce'),
        'SingleLNN_Combined': build_single_lnn_model('combined'),
    }
    results = {}

    for name, model in models.items():
        print(f"=== Training {name} ===")
        param_count = summarize_and_plot(model, name)
        ds_train = conv_train_ds if 'ConvNet' in name or name in ['MultiKernelCNN','UNet'] else lnn_train_ds
        ds_val = conv_val_ds if 'ConvNet' in name or name in ['MultiKernelCNN','UNet'] else lnn_val_ds
        train_model(model, ds_train, ds_val, name)
        results[name] = evaluate_model(model, conv_test_ds if 'ConvNet' in name or name in ['MultiKernelCNN','UNet'] else lnn_test_ds, param_count)

    # Ensemble LNN
    print("=== Training EnsembleLNN ===")
    ens_model = build_ensemble_lnn_simple()
    summarize_and_plot(ens_model, 'EnsembleLNN')
    train_model(ens_model, ens_train_ds, ens_val_ds, 'EnsembleLNN')
    results['EnsembleLNN'] = evaluate_model(ens_model, ens_test_ds)

    print("\nFinal Results:")
    for name, metrics in results.items():
        print(f"{name}: {metrics}")

if __name__ == '__main__':
    main()
