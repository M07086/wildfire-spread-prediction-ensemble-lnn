"""
Training script for Part II hyperparameterized ensemble LNN models.
"""
from src.part2 import (
    train_file_pattern, val_file_pattern, test_file_pattern,
    prepare_datasets, prepare_bootstrap_input,
    generate_fixed_ensemble, generate_formula_based_ensemble,
    build_custom_ensemble_lnn, train_model, evaluate_model,
    INPUT_FEATURES, ENSEMBLE_BATCH_SIZE
)

def main():
    # Prepare ensemble datasets
    train_ds_ens, val_ds_ens, test_ds_ens = prepare_datasets(
        train_file_pattern, val_file_pattern, test_file_pattern,
        batch_size=ENSEMBLE_BATCH_SIZE, ensemble_flag=True
    )

    results = {}
    # Fixed and formula-based ensembles
    for k in [2, 3, 4, 5]:
        for method, generator in [('fixed', generate_fixed_ensemble), ('formula', generate_formula_based_ensemble)]:
            subsets, branch_count = generator(INPUT_FEATURES, k)
            name = f"{method}_k{k}"
            print(f"=== Training {name}: {branch_count} branches of size {k} ===")
            model = build_custom_ensemble_lnn(subsets)

            # Prepare datasets
            train_prep = prepare_bootstrap_input(train_ds_ens, subsets)
            val_prep = prepare_bootstrap_input(val_ds_ens, subsets)
            test_prep = prepare_bootstrap_input(test_ds_ens, subsets)

            # Train and evaluate
            train_model(model, train_prep, val_prep, name)
            results[name] = evaluate_model(model, test_prep)

    # Summary
    print("\nAggregated Results:")
    for name, metrics in results.items():
        print(f"{name}: {metrics}")

if __name__ == '__main__':
    main()
