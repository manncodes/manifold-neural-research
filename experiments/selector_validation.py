"""
Manifold Selector Cross-Validation
===================================

Critical Question: Does our automatic manifold selector actually work?

Methodology:
1. Generate synthetic datasets with KNOWN optimal manifolds
2. Test if selector recommends the right one
3. Measure: Recommendation accuracy, confidence calibration

This validates the meta-learning framework.
"""

import numpy as np
import sys
sys.path.append('..')
from theory.manifold_selection_theory import ManifoldSelector
from mnist_manifold_nn import ManifoldMLP, train_and_evaluate
from sklearn.model_selection import train_test_split
import json


def generate_low_rank_problem(n_samples=200, n_features=50, rank=5):
    """
    Generate low-rank data where Stiefel SHOULD help

    Ground truth: data lies in rank-k subspace
    Expected: Selector recommends Stiefel or Grassmann
    """
    U = np.random.randn(n_samples, rank)
    V = np.random.randn(rank, n_features)
    X = U @ V

    # Labels from first principal component
    y = (X[:, 0] > np.median(X[:, 0])).astype(int)

    return X, y, "Stiefel"  # Expected recommendation


def generate_high_noise_problem(n_samples=200, n_features=30):
    """
    Generate noisy data where SpectralNorm SHOULD help

    High noise requires regularization
    Expected: Selector recommends SpectralNorm
    """
    X_clean = np.random.randn(n_samples, n_features)
    noise = np.random.randn(n_samples, n_features) * 3.0  # High noise
    X = X_clean + noise

    # Simple linear separator
    y = (X_clean[:, 0] + X_clean[:, 1] > 0).astype(int)

    return X, y, "SpectralNorm"


def generate_simple_problem(n_samples=500, n_features=20):
    """
    Generate simple problem where Unconstrained SHOULD win

    Clean, linearly separable, plenty of data
    Expected: Selector recommends Unconstrained
    """
    X = np.random.randn(n_samples, n_features)

    # Very simple decision boundary
    y = (X[:, 0] > 0).astype(int)

    return X, y, "Unconstrained"


def generate_limited_data_problem(n_samples=50, n_features=100):
    """
    Limited samples, high-dimensional

    Need strong regularization
    Expected: SpectralNorm or Stiefel
    """
    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, 2, n_samples)

    return X, y, "SpectralNorm"


def validate_selector():
    """
    Main validation: Does selector recommend correct manifold?
    """

    print("=" * 70)
    print("MANIFOLD SELECTOR CROSS-VALIDATION")
    print("=" * 70)

    selector = ManifoldSelector()

    # Define test scenarios
    scenarios = [
        ("Low-Rank Data", generate_low_rank_problem),
        ("High Noise", generate_high_noise_problem),
        ("Simple Problem", generate_simple_problem),
        ("Limited Data", generate_limited_data_problem),
    ]

    results = {
        'scenarios': [],
        'correct_recommendations': 0,
        'total_scenarios': len(scenarios)
    }

    for scenario_name, generator in scenarios:
        print(f"\n{'='*70}")
        print(f"SCENARIO: {scenario_name}")
        print(f"{'='*70}")

        # Generate data
        X, y, expected_manifold = generator()

        print(f"Expected Optimal: {expected_manifold}")

        # Get recommendation
        recommendation = selector.recommend_manifold(X, y)

        print(f"\nSelector Recommendation:")
        selector.explain_recommendation(recommendation)

        # Check if correct
        is_correct = recommendation.manifold_type == expected_manifold

        if is_correct:
            results['correct_recommendations'] += 1
            print(f"\n✓ CORRECT RECOMMENDATION")
        else:
            print(f"\n✗ INCORRECT: Expected {expected_manifold}, got {recommendation.manifold_type}")

        # Store results
        results['scenarios'].append({
            'name': scenario_name,
            'expected': expected_manifold,
            'recommended': recommendation.manifold_type,
            'confidence': recommendation.confidence,
            'correct': is_correct
        })

    # Summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)

    accuracy = results['correct_recommendations'] / results['total_scenarios']
    print(f"\nRecommendation Accuracy: {accuracy:.2%}")
    print(f"Correct: {results['correct_recommendations']}/{results['total_scenarios']}")

    print("\nPer-Scenario Results:")
    for s in results['scenarios']:
        status = "✓" if s['correct'] else "✗"
        print(f"  {status} {s['name']}: {s['recommended']} (conf={s['confidence']:.2f})")

    # Confidence calibration
    print("\n" + "=" * 70)
    print("CONFIDENCE CALIBRATION")
    print("=" * 70)

    correct_confidences = [s['confidence'] for s in results['scenarios'] if s['correct']]
    incorrect_confidences = [s['confidence'] for s in results['scenarios'] if not s['correct']]

    if correct_confidences:
        print(f"Mean confidence (correct): {np.mean(correct_confidences):.3f}")
    if incorrect_confidences:
        print(f"Mean confidence (incorrect): {np.mean(incorrect_confidences):.3f}")

    # Analysis
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    if accuracy >= 0.75:
        print("✓ Selector performs WELL (≥75% accuracy)")
    elif accuracy >= 0.5:
        print("~ Selector performs MODERATELY (50-75% accuracy)")
    else:
        print("✗ Selector performs POORLY (<50% accuracy)")

    if correct_confidences and incorrect_confidences:
        if np.mean(correct_confidences) > np.mean(incorrect_confidences):
            print("✓ Confidence is CALIBRATED (higher for correct)")
        else:
            print("✗ Confidence is NOT calibrated")

    # Save results
    with open('selector_validation_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n\nResults saved to selector_validation_results.json")

    return results


def empirical_validation():
    """
    STRONGER TEST: Actually train networks and measure performance

    Does selector's recommendation lead to better accuracy?
    """

    print("\n" + "=" * 70)
    print("EMPIRICAL VALIDATION: Train Networks with Recommendations")
    print("=" * 70)

    selector = ManifoldSelector()

    # Test on low-rank problem
    print("\nGenerating low-rank test problem...")
    X, y = generate_low_rank_problem()[:2]

    # Normalize
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Get recommendation
    recommendation = selector.recommend_manifold(X_train, y_train)
    selector.explain_recommendation(recommendation)

    # Train with recommended manifold
    print(f"\nTraining with RECOMMENDED manifold ({recommendation.manifold_type})...")

    input_dim = X_train.shape[1]
    hidden_dim = 32
    n_classes = len(np.unique(y_train))
    architecture = [input_dim, hidden_dim, n_classes]

    model_rec = ManifoldMLP(architecture, manifold_type=recommendation.manifold_type)
    result_rec = train_and_evaluate(
        model_rec, X_train, y_train, X_test, y_test,
        epochs=50, lr=0.01, batch_size=16
    )

    print(f"Recommended ({recommendation.manifold_type}) Test Acc: {result_rec['final_test_acc']:.4f}")

    # Train with alternative (for comparison)
    alternatives = ['Unconstrained', 'Stiefel', 'SpectralNorm']
    alternatives = [m for m in alternatives if m != recommendation.manifold_type]

    best_alternative_acc = 0
    best_alternative = None

    for alt in alternatives[:2]:  # Test top 2 alternatives
        print(f"\nTraining with ALTERNATIVE ({alt})...")
        model_alt = ManifoldMLP(architecture, manifold_type=alt)
        result_alt = train_and_evaluate(
            model_alt, X_train, y_train, X_test, y_test,
            epochs=50, lr=0.01, batch_size=16
        )
        print(f"Alternative ({alt}) Test Acc: {result_alt['final_test_acc']:.4f}")

        if result_alt['final_test_acc'] > best_alternative_acc:
            best_alternative_acc = result_alt['final_test_acc']
            best_alternative = alt

    # Compare
    print("\n" + "=" * 70)
    print("EMPIRICAL COMPARISON")
    print("=" * 70)

    improvement = result_rec['final_test_acc'] - best_alternative_acc

    print(f"\nRecommended ({recommendation.manifold_type}): {result_rec['final_test_acc']:.4f}")
    print(f"Best Alternative ({best_alternative}): {best_alternative_acc:.4f}")
    print(f"Improvement: {improvement:+.4f}")

    if improvement > 0:
        print("\n✓ Selector's recommendation OUTPERFORMED alternatives")
    elif abs(improvement) < 0.01:
        print("\n~ Selector's recommendation COMPARABLE to alternatives")
    else:
        print("\n✗ Selector's recommendation UNDERPERFORMED alternatives")

    return {
        'recommended_manifold': recommendation.manifold_type,
        'recommended_acc': result_rec['final_test_acc'],
        'best_alternative': best_alternative,
        'best_alternative_acc': best_alternative_acc,
        'improvement': improvement
    }


if __name__ == "__main__":
    # Stage 1: Logical validation
    results_logical = validate_selector()

    # Stage 2: Empirical validation
    results_empirical = empirical_validation()

    print("\n" + "=" * 70)
    print("SELECTOR VALIDATION COMPLETE")
    print("=" * 70)

    print(f"\nLogical Accuracy: {results_logical['correct_recommendations']}/{results_logical['total_scenarios']}")
    print(f"Empirical Improvement: {results_empirical['improvement']:+.4f}")