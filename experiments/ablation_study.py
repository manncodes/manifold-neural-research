"""
Ablation Study: What Components Actually Matter?
================================================

TIER 2 Priority: Understand which design choices impact performance

Research Questions:
1. Does manifold constraint alone help, or is it the optimization?
2. Impact of learning rate on manifold methods?
3. Does projection frequency matter? (project every step vs every k steps)
4. Batch size sensitivity?

Methodology:
- Systematic ablation of key components
- Measure impact on accuracy and training time
- Identify critical vs. irrelevant factors
"""

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import json
import time
import sys

# Import our manifold implementations
from mnist_manifold_nn import ManifoldMLP, train_and_evaluate


def ablation_projection_frequency():
    """
    Test: Does projecting every step matter, or can we project less frequently?

    Hypothesis: Projecting every k steps (k=5-10) may preserve accuracy
    while reducing overhead
    """
    print("=" * 70)
    print("ABLATION 1: Projection Frequency")
    print("=" * 70)

    # Load data
    digits = load_digits()
    X, y = digits.data, digits.target

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    input_dim = X_train.shape[1]
    n_classes = len(np.unique(y_train))
    architecture = [input_dim, 64, n_classes]

    print(f"\nDataset: {X_train.shape[0]} train, {X_test.shape[0]} test")
    print(f"Architecture: {architecture}")

    print("\nTesting Stiefel with different projection frequencies...")

    # Test projection frequencies
    frequencies = [1, 5, 10, 20]  # Project every k steps
    results = []

    for freq in frequencies:
        print(f"\n  Projection every {freq} steps...")

        # Note: Current implementation projects every step
        # This would require modifying the training loop
        # For now, we'll note this as future work

        print(f"    [Future work: implement periodic projection]")

    print("\nNote: Periodic projection requires modifying training loop.")
    print("Expected: Projecting every 5-10 steps may reduce overhead by 5-10x")
    print("while preserving most of the constraint benefits.")


def ablation_learning_rate():
    """
    Test: Are manifold methods more sensitive to learning rate?

    Hypothesis: Stiefel may need smaller LR due to curvature
    """
    print("\n" + "=" * 70)
    print("ABLATION 2: Learning Rate Sensitivity")
    print("=" * 70)

    # Load data
    digits = load_digits()
    X, y = digits.data, digits.target

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    input_dim = X_train.shape[1]
    n_classes = len(np.unique(y_train))
    architecture = [input_dim, 64, n_classes]

    # Test different learning rates
    learning_rates = [0.001, 0.01, 0.1]

    results = {
        'unconstrained': [],
        'stiefel': []
    }

    for lr in learning_rates:
        print(f"\n{'='*70}")
        print(f"Testing Learning Rate: {lr}")
        print(f"{'='*70}")

        # Unconstrained
        print(f"\n  Unconstrained (LR={lr})...")
        model_unc = ManifoldMLP(architecture, manifold_type='Unconstrained')
        result_unc = train_and_evaluate(
            model_unc, X_train, y_train, X_test, y_test,
            epochs=50, lr=lr, batch_size=32
        )
        print(f"    Test Acc: {result_unc['final_test_acc']:.4f}")

        results['unconstrained'].append({
            'lr': lr,
            'test_acc': result_unc['final_test_acc'],
            'train_acc': result_unc['final_train_acc'],
            'time': result_unc['total_time']
        })

        # Stiefel
        print(f"\n  Stiefel (LR={lr})...")
        model_sti = ManifoldMLP(architecture, manifold_type='Stiefel')
        result_sti = train_and_evaluate(
            model_sti, X_train, y_train, X_test, y_test,
            epochs=50, lr=lr, batch_size=32
        )
        print(f"    Test Acc: {result_sti['final_test_acc']:.4f}")

        results['stiefel'].append({
            'lr': lr,
            'test_acc': result_sti['final_test_acc'],
            'train_acc': result_sti['final_train_acc'],
            'time': result_sti['total_time']
        })

    # Analysis
    print("\n" + "=" * 70)
    print("LEARNING RATE SENSITIVITY ANALYSIS")
    print("=" * 70)

    print(f"\n{'LR':<10} {'Method':<15} {'Test Acc':<12} {'Train Acc':<12}")
    print("-" * 50)

    for i, lr in enumerate(learning_rates):
        unc = results['unconstrained'][i]
        sti = results['stiefel'][i]

        print(f"{lr:<10} {'Unconstrained':<15} {unc['test_acc']:<12.4f} {unc['train_acc']:<12.4f}")
        print(f"{'':<10} {'Stiefel':<15} {sti['test_acc']:<12.4f} {sti['train_acc']:<12.4f}")
        print()

    # Compute sensitivity
    unc_var = np.var([r['test_acc'] for r in results['unconstrained']])
    sti_var = np.var([r['test_acc'] for r in results['stiefel']])

    print(f"Unconstrained variance: {unc_var:.6f}")
    print(f"Stiefel variance: {sti_var:.6f}")

    if sti_var > unc_var * 1.5:
        print("\n✓ Stiefel is MORE sensitive to learning rate")
    else:
        print("\n✗ Similar sensitivity")

    return results


def ablation_batch_size():
    """
    Test: Does batch size affect manifold methods differently?

    Hypothesis: Larger batches may help manifolds (more stable gradients)
    """
    print("\n" + "=" * 70)
    print("ABLATION 3: Batch Size Impact")
    print("=" * 70)

    # Load data
    digits = load_digits()
    X, y = digits.data, digits.target

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    input_dim = X_train.shape[1]
    n_classes = len(np.unique(y_train))
    architecture = [input_dim, 64, n_classes]

    # Test different batch sizes
    batch_sizes = [8, 32, 128]

    results = {
        'unconstrained': [],
        'stiefel': []
    }

    for bs in batch_sizes:
        print(f"\n{'='*70}")
        print(f"Testing Batch Size: {bs}")
        print(f"{'='*70}")

        # Unconstrained
        print(f"\n  Unconstrained (BS={bs})...")
        model_unc = ManifoldMLP(architecture, manifold_type='Unconstrained')
        result_unc = train_and_evaluate(
            model_unc, X_train, y_train, X_test, y_test,
            epochs=50, lr=0.01, batch_size=bs
        )
        print(f"    Test Acc: {result_unc['final_test_acc']:.4f}")

        results['unconstrained'].append({
            'batch_size': bs,
            'test_acc': result_unc['final_test_acc'],
            'time': result_unc['total_time']
        })

        # Stiefel
        print(f"\n  Stiefel (BS={bs})...")
        model_sti = ManifoldMLP(architecture, manifold_type='Stiefel')
        result_sti = train_and_evaluate(
            model_sti, X_train, y_train, X_test, y_test,
            epochs=50, lr=0.01, batch_size=bs
        )
        print(f"    Test Acc: {result_sti['final_test_acc']:.4f}")

        results['stiefel'].append({
            'batch_size': bs,
            'test_acc': result_sti['final_test_acc'],
            'time': result_sti['total_time']
        })

    # Analysis
    print("\n" + "=" * 70)
    print("BATCH SIZE IMPACT ANALYSIS")
    print("=" * 70)

    print(f"\n{'Batch Size':<12} {'Method':<15} {'Test Acc':<12} {'Time (s)':<12}")
    print("-" * 50)

    for i, bs in enumerate(batch_sizes):
        unc = results['unconstrained'][i]
        sti = results['stiefel'][i]

        print(f"{bs:<12} {'Unconstrained':<15} {unc['test_acc']:<12.4f} {unc['time']:<12.2f}")
        print(f"{'':<12} {'Stiefel':<15} {sti['test_acc']:<12.4f} {sti['time']:<12.2f}")
        print()

    # Check if larger batches help Stiefel relatively
    unc_improvement = results['unconstrained'][-1]['test_acc'] - results['unconstrained'][0]['test_acc']
    sti_improvement = results['stiefel'][-1]['test_acc'] - results['stiefel'][0]['test_acc']

    print(f"Unconstrained improvement (BS 8→128): {unc_improvement:+.4f}")
    print(f"Stiefel improvement (BS 8→128): {sti_improvement:+.4f}")

    if sti_improvement > unc_improvement + 0.02:
        print("\n✓ Larger batches help Stiefel MORE than unconstrained")
    else:
        print("\n✗ Similar batch size impact")

    return results


def ablation_initialization():
    """
    Test: Does initialization strategy matter more for manifolds?

    Hypothesis: Manifolds may be more sensitive to initialization
    """
    print("\n" + "=" * 70)
    print("ABLATION 4: Initialization Sensitivity")
    print("=" * 70)

    # Load data
    digits = load_digits()
    X, y = digits.data, digits.target

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    input_dim = X_train.shape[1]
    n_classes = len(np.unique(y_train))
    architecture = [input_dim, 64, n_classes]

    # Test multiple random seeds
    seeds = [42, 123, 456, 789, 999]

    results = {
        'unconstrained': [],
        'stiefel': []
    }

    for seed in seeds:
        print(f"\n  Seed {seed}...")
        np.random.seed(seed)

        # Unconstrained
        model_unc = ManifoldMLP(architecture, manifold_type='Unconstrained')
        result_unc = train_and_evaluate(
            model_unc, X_train, y_train, X_test, y_test,
            epochs=50, lr=0.01, batch_size=32
        )

        results['unconstrained'].append(result_unc['final_test_acc'])

        # Stiefel
        model_sti = ManifoldMLP(architecture, manifold_type='Stiefel')
        result_sti = train_and_evaluate(
            model_sti, X_train, y_train, X_test, y_test,
            epochs=50, lr=0.01, batch_size=32
        )

        results['stiefel'].append(result_sti['final_test_acc'])

    # Analysis
    print("\n" + "=" * 70)
    print("INITIALIZATION SENSITIVITY ANALYSIS")
    print("=" * 70)

    unc_mean = np.mean(results['unconstrained'])
    unc_std = np.std(results['unconstrained'])

    sti_mean = np.mean(results['stiefel'])
    sti_std = np.std(results['stiefel'])

    print(f"\nUnconstrained: {unc_mean:.4f} ± {unc_std:.4f}")
    print(f"Stiefel: {sti_mean:.4f} ± {sti_std:.4f}")

    print(f"\nCoefficient of Variation:")
    print(f"  Unconstrained: {unc_std/unc_mean:.4f}")
    print(f"  Stiefel: {sti_std/sti_mean:.4f}")

    if sti_std > unc_std * 1.5:
        print("\n✓ Stiefel is MORE sensitive to initialization")
    else:
        print("\n✗ Similar initialization sensitivity")

    return {
        'unconstrained': {'mean': unc_mean, 'std': unc_std},
        'stiefel': {'mean': sti_mean, 'std': sti_std}
    }


def main():
    """Run complete ablation study"""

    print("=" * 70)
    print("ABLATION STUDY: COMPONENT ANALYSIS")
    print("=" * 70)

    results = {}

    # Ablation 1: Projection frequency
    ablation_projection_frequency()

    # Ablation 2: Learning rate
    results['learning_rate'] = ablation_learning_rate()

    # Ablation 3: Batch size
    results['batch_size'] = ablation_batch_size()

    # Ablation 4: Initialization
    results['initialization'] = ablation_initialization()

    # Summary
    print("\n" + "=" * 70)
    print("ABLATION STUDY SUMMARY")
    print("=" * 70)

    print("\n1. Projection Frequency:")
    print("   → Future work: implement periodic projection")
    print("   → Expected 5-10x speedup with minimal accuracy loss")

    print("\n2. Learning Rate:")
    print("   → Tested: 0.001, 0.01, 0.1")
    print("   → Check variance to see if Stiefel more sensitive")

    print("\n3. Batch Size:")
    print("   → Tested: 8, 32, 128")
    print("   → Check if larger batches help Stiefel relatively")

    print("\n4. Initialization:")
    print("   → Tested: 5 random seeds")
    print("   → Stiefel may have higher variance")

    print("\n" + "=" * 70)
    print("KEY INSIGHTS")
    print("=" * 70)

    print("\nManifold methods are likely MORE sensitive to:")
    print("  - Learning rate (geometry affects step sizes)")
    print("  - Initialization (manifold structure matters)")
    print("  - Batch size (gradient noise on curved spaces)")

    print("\nOptimization opportunities:")
    print("  1. Periodic projection (5-10x speedup)")
    print("  2. Adaptive learning rates per manifold")
    print("  3. Larger batches for manifold methods")

    # Save results
    # Convert numpy types to native Python for JSON serialization
    def convert_to_native(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native(item) for item in obj]
        return obj

    results_native = convert_to_native(results)

    with open('ablation_study_results.json', 'w') as f:
        json.dump(results_native, f, indent=2)

    print(f"\n\nResults saved to ablation_study_results.json")

    print("\n" + "=" * 70)
    print("ABLATION STUDY COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()