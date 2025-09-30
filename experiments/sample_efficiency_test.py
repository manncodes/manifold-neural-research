"""
Sample Efficiency Investigation
================================

Hypothesis: Manifold constraints provide better inductive bias
with limited training data.

Experiment: Train on n_samples ∈ {50, 100, 200, 500, 1000, 2000}
Compare unconstrained vs. Stiefel vs. SpectralNorm
"""

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import json
import time
import sys
sys.path.append('..')

from experiments.mnist_manifold_nn import ManifoldMLP, train_and_evaluate


def test_sample_efficiency():
    """
    Core hypothesis test: Do manifolds help more with less data?
    """

    print("=" * 70)
    print("SAMPLE EFFICIENCY INVESTIGATION")
    print("=" * 70)

    # Load full dataset
    digits = load_digits()
    X, y = digits.data, digits.target

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Test different sample sizes
    sample_sizes = [50, 100, 200, 500, 1000, 1797]

    results = {
        'sample_sizes': sample_sizes,
        'unconstrained': [],
        'stiefel': [],
        'spectral': []
    }

    for n_samples in sample_sizes:
        print(f"\n{'='*70}")
        print(f"Testing with {n_samples} samples")
        print(f"{'='*70}")

        # Subsample
        if n_samples < len(X):
            indices = np.random.choice(len(X), n_samples, replace=False)
            X_sub = X[indices]
            y_sub = y[indices]
        else:
            X_sub, y_sub = X, y

        # Train/test split (80/20)
        X_train, X_test, y_train, y_test = train_test_split(
            X_sub, y_sub, test_size=0.2, random_state=42
        )

        input_dim = X_train.shape[1]
        n_classes = len(np.unique(y_train))
        architecture = [input_dim, 64, n_classes]

        # Test each method
        methods = [
            ('Unconstrained', 'Unconstrained'),
            ('Stiefel', 'Stiefel'),
            ('SpectralNorm', 'SpectralNorm')
        ]

        for method_name, manifold_type in methods:
            print(f"\n  Training {method_name}...")

            model = ManifoldMLP(architecture, manifold_type=manifold_type)

            result = train_and_evaluate(
                model,
                X_train, y_train,
                X_test, y_test,
                epochs=50,  # Fewer epochs for faster experiments
                lr=0.01,
                batch_size=min(16, len(X_train))  # Adjust batch size
            )

            key = method_name.lower()
            results[key].append({
                'n_samples': n_samples,
                'test_acc': result['final_test_acc'],
                'train_acc': result['final_train_acc'],
                'time': result['total_time'],
                'generalization_gap': result['final_train_acc'] - result['final_test_acc']
            })

            print(f"    Test Acc: {result['final_test_acc']:.4f}")

    # Analysis
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    # Compute relative performance at each sample size
    print("\nTest Accuracy by Sample Size:")
    print(f"{'Samples':<10} {'Unconstrained':<15} {'Stiefel':<15} {'SpectralNorm':<15}")
    print("-" * 55)

    for i, n in enumerate(sample_sizes):
        unc_acc = results['unconstrained'][i]['test_acc']
        sti_acc = results['stiefel'][i]['test_acc']
        spe_acc = results['spectral'][i]['test_acc']

        print(f"{n:<10} {unc_acc:<15.4f} {sti_acc:<15.4f} {spe_acc:<15.4f}")

    # Key insight: When does Stiefel beat unconstrained?
    print("\nDifference (Stiefel - Unconstrained):")
    for i, n in enumerate(sample_sizes):
        unc_acc = results['unconstrained'][i]['test_acc']
        sti_acc = results['stiefel'][i]['test_acc']
        diff = sti_acc - unc_acc

        status = "✓ BETTER" if diff > 0 else "✗ WORSE"
        print(f"  n={n:<6}: {diff:+.4f} {status}")

    # Hypothesis test
    print("\n" + "=" * 70)
    print("HYPOTHESIS TEST")
    print("=" * 70)

    # Count how many times Stiefel beats unconstrained at small vs large n
    small_samples = [r for r in results['stiefel'] if r['n_samples'] <= 200]
    large_samples = [r for r in results['stiefel'] if r['n_samples'] > 200]

    small_unc = [r['test_acc'] for r in results['unconstrained'] if r['n_samples'] <= 200]
    large_unc = [r['test_acc'] for r in results['unconstrained'] if r['n_samples'] > 200]

    small_wins = sum(1 for s, u in zip(small_samples, small_unc) if s['test_acc'] > u)
    large_wins = sum(1 for s, u in zip(large_samples, large_unc) if s['test_acc'] > u)

    print(f"Stiefel wins with n ≤ 200: {small_wins}/{len(small_samples)}")
    print(f"Stiefel wins with n > 200: {large_wins}/{len(large_samples)}")

    if small_wins / len(small_samples) > large_wins / len(large_samples):
        print("\n✓ HYPOTHESIS SUPPORTED: Manifolds help more with limited data")
    else:
        print("\n✗ HYPOTHESIS NOT SUPPORTED: No sample efficiency advantage")

    # Generalization gap analysis
    print("\n" + "=" * 70)
    print("GENERALIZATION GAP ANALYSIS")
    print("=" * 70)

    print("\nGeneralization Gap (Train - Test):")
    for i, n in enumerate(sample_sizes):
        unc_gap = results['unconstrained'][i]['generalization_gap']
        sti_gap = results['stiefel'][i]['generalization_gap']

        print(f"  n={n:<6}: Unc={unc_gap:.4f}, Stiefel={sti_gap:.4f}")

        if sti_gap < unc_gap:
            print(f"    → Stiefel has BETTER generalization")

    # Save results
    with open('sample_efficiency_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to sample_efficiency_results.json")

    return results


if __name__ == "__main__":
    results = test_sample_efficiency()

    print("\n" + "=" * 70)
    print("SAMPLE EFFICIENCY TEST COMPLETE")
    print("=" * 70)