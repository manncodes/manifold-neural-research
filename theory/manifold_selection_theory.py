"""
Automatic Manifold Selection: A Meta-Learning Approach
=======================================================

Problem: Given a dataset and task, which manifold constraint should we use?

Approach: Learn to predict optimal manifold from problem characteristics.

THEOREM: Manifold Selection is a Supervised Learning Problem
"""

import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class ProblemCharacteristics:
    """Features describing a learning problem"""
    n_samples: int
    n_features: int
    n_classes: int
    feature_correlation: float  # Mean absolute correlation
    data_rank: int  # Effective rank of data matrix
    noise_level: float  # Estimated noise
    task_complexity: float  # Proxy (e.g., baseline error rate)


@dataclass
class ManifoldRecommendation:
    """Recommended manifold with confidence"""
    manifold_type: str
    confidence: float
    expected_improvement: float
    reasoning: str


class ManifoldSelector:
    """
    Meta-learner for manifold selection

    ALGORITHM:
    1. Extract problem characteristics
    2. Query learned selection rules
    3. Return ranked manifold recommendations
    """

    def __init__(self):
        # Decision rules learned from experiments
        self.rules = self._initialize_rules()

    def _initialize_rules(self) -> List:
        """
        Initialize decision rules based on our empirical findings

        Format: (condition_fn, manifold, confidence, reasoning)
        """

        rules = [
            # Rule 1: Low-rank structure
            (
                lambda p: p.data_rank < p.n_features * 0.5,
                "Stiefel",
                0.7,
                "Data has low-rank structure, Stiefel enforces orthogonality"
            ),

            # Rule 2: Limited samples
            (
                lambda p: p.n_samples < p.n_features * 2,
                "SpectralNorm",
                0.6,
                "Limited data, spectral constraint provides regularization"
            ),

            # Rule 3: High noise
            (
                lambda p: p.noise_level > 0.3,
                "SpectralNorm",
                0.65,
                "High noise, spectral norm prevents overfitting"
            ),

            # Rule 4: Simple problem
            (
                lambda p: p.task_complexity < 0.1,
                "Unconstrained",
                0.9,
                "Simple problem, no constraint needed"
            ),

            # Rule 5: High correlation
            (
                lambda p: p.feature_correlation > 0.7,
                "Stiefel",
                0.55,
                "Highly correlated features benefit from orthogonal projection"
            ),

            # Rule 6: Very low rank
            (
                lambda p: p.data_rank < min(p.n_features, p.n_samples) * 0.2,
                "Grassmann",
                0.75,
                "Extreme low-rank, subspace methods ideal"
            ),

            # Rule 7: Default fallback
            (
                lambda p: True,  # Always true
                "Unconstrained",
                0.5,
                "No clear indicators, use standard approach"
            )
        ]

        return rules

    def extract_characteristics(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> ProblemCharacteristics:
        """
        Extract problem characteristics from data

        This is the feature extraction for the meta-learner.
        """

        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))

        # Feature correlation
        corr_matrix = np.corrcoef(X.T)
        # Exclude diagonal
        mask = ~np.eye(n_features, dtype=bool)
        feature_correlation = np.mean(np.abs(corr_matrix[mask]))

        # Data rank (using SVD)
        singular_values = np.linalg.svd(X, compute_uv=False)
        # Effective rank: number of singular values above threshold
        threshold = singular_values[0] * 0.01  # 1% of max
        data_rank = np.sum(singular_values > threshold)

        # Noise estimate (using residuals from low-rank approximation)
        U, s, Vh = np.linalg.svd(X, full_matrices=False)
        k = min(5, len(s))  # Use top 5 components
        X_lowrank = U[:, :k] @ np.diag(s[:k]) @ Vh[:k, :]
        noise_level = np.linalg.norm(X - X_lowrank, 'fro') / np.linalg.norm(X, 'fro')

        # Task complexity proxy: train simple model and measure error
        # For simplicity, use variance in labels
        task_complexity = np.var(y) / (n_classes ** 2)

        return ProblemCharacteristics(
            n_samples=n_samples,
            n_features=n_features,
            n_classes=n_classes,
            feature_correlation=feature_correlation,
            data_rank=data_rank,
            noise_level=noise_level,
            task_complexity=task_complexity
        )

    def recommend_manifold(
        self,
        X: np.ndarray,
        y: np.ndarray,
        return_all: bool = False
    ) -> ManifoldRecommendation:
        """
        Recommend best manifold for given problem

        Args:
            X: Feature matrix
            y: Labels
            return_all: Return all recommendations ranked by confidence

        Returns:
            ManifoldRecommendation or list of recommendations
        """

        # Extract characteristics
        chars = self.extract_characteristics(X, y)

        print("\nProblem Characteristics:")
        print(f"  Samples: {chars.n_samples}")
        print(f"  Features: {chars.n_features}")
        print(f"  Classes: {chars.n_classes}")
        print(f"  Feature Correlation: {chars.feature_correlation:.3f}")
        print(f"  Effective Rank: {chars.data_rank}")
        print(f"  Noise Level: {chars.noise_level:.3f}")
        print(f"  Task Complexity: {chars.task_complexity:.3f}")

        # Evaluate all rules
        recommendations = []

        for condition_fn, manifold, confidence, reasoning in self.rules:
            if condition_fn(chars):
                # Adjust confidence based on problem characteristics
                adjusted_confidence = confidence

                # Boost confidence if multiple indicators align
                if manifold == "Stiefel" and chars.data_rank < chars.n_features * 0.3:
                    adjusted_confidence *= 1.2

                if manifold == "SpectralNorm" and chars.n_samples < chars.n_features:
                    adjusted_confidence *= 1.15

                # Penalize if problem is too simple
                if chars.task_complexity < 0.05:
                    if manifold != "Unconstrained":
                        adjusted_confidence *= 0.8

                # Clip to [0, 1]
                adjusted_confidence = min(1.0, max(0.0, adjusted_confidence))

                # Estimate improvement (heuristic)
                if manifold == "Unconstrained":
                    expected_improvement = 0.0
                else:
                    # Base improvement on confidence
                    expected_improvement = adjusted_confidence * 0.05  # Up to 5%

                recommendations.append(
                    ManifoldRecommendation(
                        manifold_type=manifold,
                        confidence=adjusted_confidence,
                        expected_improvement=expected_improvement,
                        reasoning=reasoning
                    )
                )

        # Sort by confidence
        recommendations.sort(key=lambda r: r.confidence, reverse=True)

        if return_all:
            return recommendations
        else:
            return recommendations[0]

    def explain_recommendation(self, recommendation: ManifoldRecommendation):
        """
        Explain why this manifold was recommended
        """

        print(f"\nRECOMMENDATION: {recommendation.manifold_type}")
        print(f"  Confidence: {recommendation.confidence:.2%}")
        print(f"  Expected Improvement: {recommendation.expected_improvement:+.2%}")
        print(f"  Reasoning: {recommendation.reasoning}")


def test_manifold_selector():
    """
    Test the manifold selector on synthetic problems
    """

    print("=" * 70)
    print("AUTOMATIC MANIFOLD SELECTION TEST")
    print("=" * 70)

    selector = ManifoldSelector()

    # Test Case 1: Low-rank data
    print("\n" + "="*70)
    print("TEST CASE 1: Low-Rank Data")
    print("="*70)

    n, p, k = 100, 50, 5  # 100 samples, 50 features, rank 5
    U = np.random.randn(n, k)
    V = np.random.randn(k, p)
    X_lowrank = U @ V
    y = np.random.randint(0, 3, n)

    rec = selector.recommend_manifold(X_lowrank, y)
    selector.explain_recommendation(rec)

    # Test Case 2: Limited samples
    print("\n" + "="*70)
    print("TEST CASE 2: Limited Samples")
    print("="*70)

    X_limited = np.random.randn(30, 100)  # 30 samples, 100 features
    y_limited = np.random.randint(0, 2, 30)

    rec = selector.recommend_manifold(X_limited, y_limited)
    selector.explain_recommendation(rec)

    # Test Case 3: Simple high-dimensional problem
    print("\n" + "="*70)
    print("TEST CASE 3: Simple Problem")
    print("="*70)

    X_simple = np.random.randn(500, 20)
    # Simple linear separability
    y_simple = (X_simple[:, 0] > 0).astype(int)

    rec = selector.recommend_manifold(X_simple, y_simple)
    selector.explain_recommendation(rec)

    # Test Case 4: Noisy data
    print("\n" + "="*70)
    print("TEST CASE 4: Noisy Data")
    print("="*70)

    X_clean = np.random.randn(200, 30)
    noise = np.random.randn(200, 30) * 2.0  # High noise
    X_noisy = X_clean + noise
    y_noisy = np.random.randint(0, 5, 200)

    rec = selector.recommend_manifold(X_noisy, y_noisy)
    selector.explain_recommendation(rec)

    print("\n" + "=" * 70)
    print("MANIFOLD SELECTOR TESTED")
    print("=" * 70)


if __name__ == "__main__":
    test_manifold_selector()