"""
Grassmann and Product Manifold Neural Networks
===============================================

Extending our manifold coverage beyond Stiefel and SpectralNorm.

GRASSMANN MANIFOLD:
- Grassmann(n,p): space of p-dimensional subspaces in R^n
- Related to Stiefel but quotient by rotations: Gr(n,p) = St(n,p) / O(p)
- Better for subspace learning tasks

PRODUCT MANIFOLD:
- M1 × M2: combine different manifolds for different layers
- Example: Stiefel × SpectralNorm for hybrid constraints
- Flexibility: match constraint to layer role

Research Questions:
1. Do Grassmann manifolds outperform Stiefel for subspace tasks?
2. Can product manifolds get best of both worlds?
3. How does computational cost scale?
"""

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import json
import time


class GrassmannLayer:
    """
    Grassmann manifold layer via Stiefel quotient

    Instead of constraining W to St(n,p), we only care about
    the subspace span(W), not the specific basis.

    This is implemented by:
    1. Projecting gradient to Stiefel tangent space
    2. Using QR retraction (which preserves subspace)
    """

    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Initialize on Stiefel (but interpret as Grassmann)
        W = np.random.randn(input_dim, output_dim)
        self.W, _ = np.linalg.qr(W)  # Orthonormal columns

    def forward(self, X):
        """Forward pass: y = W^T x"""
        return X @ self.W

    def backward(self, X, grad_output):
        """
        Backward on Grassmann:
        - Euclidean gradient: W @ grad_output.T @ X
        - Project to horizontal space (quotient by O(p))
        - This removes "rotation" components
        """
        # Euclidean gradient
        grad_eucl = X.T @ grad_output

        # Grassmann projection: remove symmetric part of W^T grad
        # This is horizontal lift for Grassmann
        WtG = self.W.T @ grad_eucl
        grad_grass = grad_eucl - self.W @ (WtG + WtG.T) / 2

        return grad_grass

    def retract(self, grad, lr):
        """QR retraction (natural for Grassmann)"""
        W_new = self.W - lr * grad
        self.W, _ = np.linalg.qr(W_new)

    def get_params(self):
        return self.W.copy()


class ProductManifoldLayer:
    """
    Product Manifold: St(n1,p1) × Spec(n2,p2)

    Two weight matrices with DIFFERENT constraints:
    - W1: Stiefel (orthogonality)
    - W2: SpectralNorm (bounded spectral radius)

    This allows layer-specific geometric priors.
    """

    def __init__(self, input_dim, hidden_dim, output_dim):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # W1: Stiefel constraint (input -> hidden)
        W1 = np.random.randn(input_dim, hidden_dim)
        self.W1, _ = np.linalg.qr(W1)

        # W2: SpectralNorm constraint (hidden -> output)
        self.W2 = np.random.randn(hidden_dim, output_dim) * 0.1
        self._project_spectral()

    def _project_spectral(self, max_sv=1.0):
        """Project W2 to spectral norm ball"""
        U, s, Vh = np.linalg.svd(self.W2, full_matrices=False)
        s_clipped = np.minimum(s, max_sv)
        self.W2 = U @ np.diag(s_clipped) @ Vh

    def forward(self, X):
        """Two-layer forward: X -> W1 -> ReLU -> W2"""
        self.X = X
        self.h = X @ self.W1
        self.h_act = np.maximum(0, self.h)  # ReLU
        return self.h_act @ self.W2

    def backward(self, grad_output, lr):
        """
        Backward through product manifold:
        - W2 gradient with spectral projection
        - W1 gradient with Stiefel projection
        """
        # Gradient w.r.t W2 (spectral constraint)
        grad_W2 = self.h_act.T @ grad_output

        # Gradient w.r.t h (backprop through ReLU)
        grad_h = (grad_output @ self.W2.T) * (self.h > 0)

        # Gradient w.r.t W1 (Stiefel constraint)
        grad_eucl_W1 = self.X.T @ grad_h

        # Stiefel projection for W1
        WtG = self.W1.T @ grad_eucl_W1
        grad_W1 = grad_eucl_W1 - self.W1 @ (WtG + WtG.T) / 2

        # Update W1 (Stiefel)
        W1_new = self.W1 - lr * grad_W1
        self.W1, _ = np.linalg.qr(W1_new)

        # Update W2 (SpectralNorm)
        self.W2 = self.W2 - lr * grad_W2
        self._project_spectral()

    def get_params(self):
        return {'W1': self.W1.copy(), 'W2': self.W2.copy()}


class ManifoldMLPExtended:
    """
    Extended MLP supporting:
    - Grassmann layers
    - Product manifolds
    - Comparison with baseline
    """

    def __init__(self, architecture, manifold_type='Grassmann'):
        """
        Args:
            architecture: [input_dim, hidden_dim, output_dim]
            manifold_type: 'Grassmann', 'Product', 'Unconstrained'
        """
        self.architecture = architecture
        self.manifold_type = manifold_type

        input_dim, hidden_dim, output_dim = architecture

        if manifold_type == 'Grassmann':
            self.layer1 = GrassmannLayer(input_dim, hidden_dim)
            self.W2 = np.random.randn(hidden_dim, output_dim) * 0.1

        elif manifold_type == 'Product':
            # Single product layer handles both transformations
            self.product_layer = ProductManifoldLayer(
                input_dim, hidden_dim, output_dim
            )

        elif manifold_type == 'Unconstrained':
            self.W1 = np.random.randn(input_dim, hidden_dim) * 0.1
            self.W2 = np.random.randn(hidden_dim, output_dim) * 0.1

    def forward(self, X):
        """Forward pass"""
        if self.manifold_type == 'Product':
            return self.product_layer.forward(X)

        elif self.manifold_type == 'Grassmann':
            h = self.layer1.forward(X)
            h = np.maximum(0, h)  # ReLU
            return h @ self.W2

        else:  # Unconstrained
            h = X @ self.W1
            h = np.maximum(0, h)
            return h @ self.W2

    def backward_and_update(self, X, y, lr):
        """Backward pass with manifold-aware updates"""
        batch_size = X.shape[0]

        # Forward
        if self.manifold_type == 'Product':
            logits = self.product_layer.forward(X)
        elif self.manifold_type == 'Grassmann':
            h = self.layer1.forward(X)
            h_act = np.maximum(0, h)
            logits = h_act @ self.W2
        else:  # Unconstrained
            h = X @ self.W1
            h_act = np.maximum(0, h)
            logits = h_act @ self.W2

        # Softmax
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        # Cross-entropy gradient
        grad_output = probs.copy()
        grad_output[np.arange(batch_size), y] -= 1
        grad_output /= batch_size

        # Backward
        if self.manifold_type == 'Product':
            self.product_layer.backward(grad_output, lr)

        elif self.manifold_type == 'Grassmann':
            # Update W2 (unconstrained)
            grad_W2 = h_act.T @ grad_output
            self.W2 -= lr * grad_W2

            # Backprop to layer1
            grad_h = (grad_output @ self.W2.T) * (h > 0)
            grad_W1 = self.layer1.backward(X, grad_h)
            self.layer1.retract(grad_W1, lr)

        else:  # Unconstrained
            grad_W2 = h_act.T @ grad_output
            self.W2 -= lr * grad_W2

            grad_h = (grad_output @ self.W2.T) * (h > 0)
            grad_W1 = X.T @ grad_h
            self.W1 -= lr * grad_W1


def train_and_evaluate_extended(
    model,
    X_train, y_train,
    X_test, y_test,
    epochs=100,
    lr=0.01,
    batch_size=32
):
    """Training loop for extended manifolds"""

    n_train = X_train.shape[0]
    start_time = time.time()

    train_accs = []
    test_accs = []

    for epoch in range(epochs):
        # Shuffle
        indices = np.random.permutation(n_train)
        X_shuffled = X_train[indices]
        y_shuffled = y_train[indices]

        # Mini-batch training
        for i in range(0, n_train, batch_size):
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]

            model.backward_and_update(X_batch, y_batch, lr)

        # Evaluate every 10 epochs
        if epoch % 10 == 0 or epoch == epochs - 1:
            train_logits = model.forward(X_train)
            train_preds = np.argmax(train_logits, axis=1)
            train_acc = np.mean(train_preds == y_train)

            test_logits = model.forward(X_test)
            test_preds = np.argmax(test_logits, axis=1)
            test_acc = np.mean(test_preds == y_test)

            train_accs.append(train_acc)
            test_accs.append(test_acc)

            print(f"  Epoch {epoch:3d}: Train={train_acc:.4f}, Test={test_acc:.4f}")

    total_time = time.time() - start_time

    return {
        'final_train_acc': train_accs[-1],
        'final_test_acc': test_accs[-1],
        'total_time': total_time,
        'train_history': train_accs,
        'test_history': test_accs
    }


def main():
    """
    Experiment: Compare Grassmann and Product manifolds to baseline

    Hypothesis:
    1. Grassmann should perform similar to Stiefel (quotient structure)
    2. Product manifolds may combine benefits of multiple constraints
    """

    print("=" * 70)
    print("GRASSMANN AND PRODUCT MANIFOLD EXPERIMENTS")
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
    hidden_dim = 64
    n_classes = len(np.unique(y_train))
    architecture = [input_dim, hidden_dim, n_classes]

    print(f"\nDataset: {X_train.shape[0]} train, {X_test.shape[0]} test")
    print(f"Architecture: {architecture}")

    results = {}

    # Test each manifold type
    manifold_types = ['Unconstrained', 'Grassmann', 'Product']

    for manifold_type in manifold_types:
        print(f"\n{'='*70}")
        print(f"Training: {manifold_type}")
        print(f"{'='*70}")

        model = ManifoldMLPExtended(architecture, manifold_type=manifold_type)

        result = train_and_evaluate_extended(
            model,
            X_train, y_train,
            X_test, y_test,
            epochs=100,
            lr=0.01,
            batch_size=32
        )

        results[manifold_type] = result

        print(f"\nFinal Results:")
        print(f"  Test Accuracy: {result['final_test_acc']:.4f}")
        print(f"  Train Accuracy: {result['final_train_acc']:.4f}")
        print(f"  Training Time: {result['total_time']:.2f}s")
        print(f"  Generalization Gap: {result['final_train_acc'] - result['final_test_acc']:.4f}")

    # Comparative analysis
    print("\n" + "=" * 70)
    print("COMPARATIVE ANALYSIS")
    print("=" * 70)

    print(f"\n{'Method':<20} {'Test Acc':<12} {'Time (s)':<12} {'Gap':<12}")
    print("-" * 56)

    for method in manifold_types:
        r = results[method]
        gap = r['final_train_acc'] - r['final_test_acc']
        print(f"{method:<20} {r['final_test_acc']:<12.4f} {r['total_time']:<12.2f} {gap:<12.4f}")

    # Key comparisons
    print("\n" + "=" * 70)
    print("KEY COMPARISONS")
    print("=" * 70)

    unc_acc = results['Unconstrained']['final_test_acc']
    grass_acc = results['Grassmann']['final_test_acc']
    prod_acc = results['Product']['final_test_acc']

    print(f"\nGrassmann vs Unconstrained:")
    print(f"  Δ Accuracy: {grass_acc - unc_acc:+.4f}")
    print(f"  {'BETTER' if grass_acc > unc_acc else 'WORSE'}")

    print(f"\nProduct vs Unconstrained:")
    print(f"  Δ Accuracy: {prod_acc - unc_acc:+.4f}")
    print(f"  {'BETTER' if prod_acc > unc_acc else 'WORSE'}")

    print(f"\nProduct vs Grassmann:")
    print(f"  Δ Accuracy: {prod_acc - grass_acc:+.4f}")
    print(f"  {'BETTER' if prod_acc > grass_acc else 'WORSE'}")

    # Computational efficiency
    print("\n" + "=" * 70)
    print("COMPUTATIONAL EFFICIENCY")
    print("=" * 70)

    unc_time = results['Unconstrained']['total_time']

    for method in ['Grassmann', 'Product']:
        overhead = results[method]['total_time'] / unc_time
        print(f"\n{method}:")
        print(f"  Time overhead: {overhead:.2f}x")
        print(f"  Absolute time: {results[method]['total_time']:.2f}s")

    # Save results
    with open('grassmann_product_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n\nResults saved to grassmann_product_results.json")

    print("\n" + "=" * 70)
    print("GRASSMANN & PRODUCT MANIFOLD EXPERIMENTS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()