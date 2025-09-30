"""
Core Manifold Implementations
==============================
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, Dict, Any
import math


class ManifoldOptimizer:
    """Base class for manifold-constrained optimization"""

    def __init__(self, manifold_type: str):
        self.manifold_type = manifold_type
        self.metrics = {}

    def project(self, X: torch.Tensor) -> torch.Tensor:
        """Project onto the manifold"""
        raise NotImplementedError

    def retract(self, X: torch.Tensor, V: torch.Tensor, t: float = 1.0) -> torch.Tensor:
        """Retraction mapping from tangent space to manifold"""
        raise NotImplementedError

    def log(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """Logarithmic map (inverse of exponential map)"""
        raise NotImplementedError

    def geodesic_distance(self, X: torch.Tensor, Y: torch.Tensor) -> float:
        """Compute geodesic distance between two points"""
        raise NotImplementedError

    def tangent_projection(self, X: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        """Project vector V onto tangent space at X"""
        raise NotImplementedError


class StiefelManifold(ManifoldOptimizer):
    """
    Stiefel manifold St(n,p): set of n×p matrices with orthonormal columns
    """

    def __init__(self, n: int, p: int):
        super().__init__("Stiefel")
        self.n = n
        self.p = p

    def project(self, X: torch.Tensor) -> torch.Tensor:
        """Project via SVD"""
        U, _, Vh = torch.linalg.svd(X, full_matrices=False)
        return U @ Vh

    def retract(self, X: torch.Tensor, V: torch.Tensor, t: float = 1.0) -> torch.Tensor:
        """Cayley retraction"""
        n, p = X.shape
        A = V @ X.T - X @ V.T
        I = torch.eye(n, device=X.device, dtype=X.dtype)

        # Cayley transform
        Y = torch.linalg.solve(I + t/2 * A, I - t/2 * A) @ X
        return Y

    def tangent_projection(self, X: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        """Project onto tangent space of Stiefel at X"""
        XtV = X.T @ V
        return V - X @ (XtV + XtV.T) / 2

    def geodesic_distance(self, X: torch.Tensor, Y: torch.Tensor) -> float:
        """Geodesic distance on Stiefel manifold"""
        XtY = X.T @ Y
        # Compute principal angles
        _, s, _ = torch.linalg.svd(XtY)
        # Clamp for numerical stability
        s = torch.clamp(s, -1, 1)
        theta = torch.acos(s)
        return torch.norm(theta).item()


class GrassmannManifold(ManifoldOptimizer):
    """
    Grassmann manifold Gr(n,p): set of p-dimensional subspaces in R^n
    """

    def __init__(self, n: int, p: int):
        super().__init__("Grassmann")
        self.n = n
        self.p = p

    def project(self, X: torch.Tensor) -> torch.Tensor:
        """Project via QR decomposition"""
        Q, _ = torch.linalg.qr(X)
        return Q

    def retract(self, X: torch.Tensor, V: torch.Tensor, t: float = 1.0) -> torch.Tensor:
        """QR-based retraction"""
        Y = X + t * V
        Q, _ = torch.linalg.qr(Y)
        return Q

    def tangent_projection(self, X: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        """Project onto tangent space of Grassmann at X"""
        return V - X @ (X.T @ V)

    def geodesic_distance(self, X: torch.Tensor, Y: torch.Tensor) -> float:
        """Geodesic distance via principal angles"""
        XtY = X.T @ Y
        _, s, _ = torch.linalg.svd(XtY)
        s = torch.clamp(s, -1, 1)
        theta = torch.acos(s)
        return torch.norm(theta).item()

    def principal_angles(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """Compute principal angles between subspaces"""
        XtY = X.T @ Y
        _, s, _ = torch.linalg.svd(XtY)
        s = torch.clamp(s, -1, 1)
        return torch.acos(s)


class HyperbolicManifold(ManifoldOptimizer):
    """
    Hyperbolic manifold (Poincaré ball model)
    """

    def __init__(self, dim: int, c: float = 1.0):
        super().__init__("Hyperbolic")
        self.dim = dim
        self.c = c  # Curvature

    def project(self, X: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
        """Project onto Poincaré ball"""
        norm = torch.norm(X, dim=-1, keepdim=True)
        scale = torch.clamp(norm, max=(1 - eps) / np.sqrt(self.c))
        return scale * X / (norm + eps)

    def mobius_add(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Möbius addition in Poincaré ball"""
        xy = torch.sum(x * y, dim=-1, keepdim=True)
        x2 = torch.sum(x * x, dim=-1, keepdim=True)
        y2 = torch.sum(y * y, dim=-1, keepdim=True)

        num = (1 + 2*self.c*xy + self.c*y2) * x + (1 - self.c*x2) * y
        denom = 1 + 2*self.c*xy + self.c**2 * x2 * y2
        return num / (denom + 1e-8)

    def exp_map(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Exponential map from tangent space to manifold"""
        v_norm = torch.norm(v, dim=-1, keepdim=True)
        v_norm = torch.clamp(v_norm, min=1e-10)

        lambda_x = 2 / (1 - self.c * torch.sum(x*x, dim=-1, keepdim=True))

        # Compute exponential map
        arg = lambda_x * v_norm * np.sqrt(self.c) / 2
        exp_base = torch.tanh(arg) * v / (v_norm * np.sqrt(self.c))

        return self.mobius_add(x, exp_base)

    def log_map(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Logarithmic map from manifold to tangent space"""
        diff = self.mobius_add(-x, y)
        diff_norm = torch.norm(diff, dim=-1, keepdim=True)
        diff_norm = torch.clamp(diff_norm, min=1e-10)

        lambda_x = 2 / (1 - self.c * torch.sum(x*x, dim=-1, keepdim=True))

        arg = diff_norm * np.sqrt(self.c)
        log_base = torch.atanh(torch.clamp(arg, max=1-1e-7)) * diff / (arg + 1e-8)

        return 2 * log_base / (lambda_x * np.sqrt(self.c))

    def geodesic_distance(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """Hyperbolic distance in Poincaré ball"""
        diff = self.mobius_add(-x, y)
        diff_norm = torch.norm(diff, dim=-1)
        dist = 2 * torch.atanh(torch.clamp(diff_norm * np.sqrt(self.c), max=1-1e-7)) / np.sqrt(self.c)
        return dist.mean().item()


class SymplecticManifold(ManifoldOptimizer):
    """
    Symplectic manifold Sp(2n): matrices preserving symplectic form
    """

    def __init__(self, n: int):
        super().__init__("Symplectic")
        self.n = n
        self.J = self._canonical_symplectic_form(n)

    def _canonical_symplectic_form(self, n: int) -> torch.Tensor:
        """Create canonical symplectic form matrix J"""
        J = torch.zeros(2*n, 2*n)
        J[:n, n:] = torch.eye(n)
        J[n:, :n] = -torch.eye(n)
        return J

    def project(self, M: torch.Tensor) -> torch.Tensor:
        """Project matrix onto symplectic group via Cayley transform"""
        n = self.n
        J = self.J.to(M.device)

        # Ensure M is skew-symmetric with respect to J
        A = (M - J @ M.T @ J) / 2

        # Cayley transform
        I = torch.eye(2*n, device=M.device)
        return torch.linalg.solve(I - A, I + A)

    def check_symplectic(self, M: torch.Tensor, tol: float = 1e-6) -> bool:
        """Check if matrix is symplectic: M^T J M = J"""
        J = self.J.to(M.device)
        error = torch.norm(M.T @ J @ M - J)
        return error.item() < tol

    def tangent_projection(self, M: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        """Project onto tangent space (Lie algebra sp(2n))"""
        J = self.J.to(M.device)
        # Tangent vectors satisfy: V^T J + J^T V = 0
        return (V - J @ V.T @ J) / 2


class FlagManifold(ManifoldOptimizer):
    """
    Flag manifold: nested sequence of subspaces
    V_1 ⊂ V_2 ⊂ ... ⊂ V_k ⊂ R^n
    """

    def __init__(self, n: int, dims: list):
        super().__init__("Flag")
        self.n = n
        self.dims = dims  # Dimensions of nested subspaces
        assert all(d1 < d2 for d1, d2 in zip(dims[:-1], dims[1:]))
        assert dims[-1] <= n

    def project(self, X: torch.Tensor) -> list:
        """Project onto flag manifold as list of orthonormal bases"""
        flags = []
        Q, _ = torch.linalg.qr(X)

        for d in self.dims:
            flags.append(Q[:, :d])

        return flags

    def flag_distance(self, F1: list, F2: list) -> float:
        """Distance between two flags"""
        total_dist = 0.0

        for V1, V2 in zip(F1, F2):
            # Use Grassmann distance for each subspace
            gr = GrassmannManifold(self.n, V1.shape[1])
            total_dist += gr.geodesic_distance(V1, V2)**2

        return np.sqrt(total_dist)


class ProductManifold(ManifoldOptimizer):
    """
    Product of multiple manifolds M = M1 × M2 × ... × Mk
    """

    def __init__(self, manifolds: list):
        super().__init__("Product")
        self.manifolds = manifolds

    def project(self, X: list) -> list:
        """Project each component onto its manifold"""
        return [m.project(x) for m, x in zip(self.manifolds, X)]

    def retract(self, X: list, V: list, t: float = 1.0) -> list:
        """Retract each component"""
        return [m.retract(x, v, t) for m, x, v in zip(self.manifolds, X, V)]

    def geodesic_distance(self, X: list, Y: list) -> float:
        """Product metric: sqrt(sum of squared distances)"""
        dists = [m.geodesic_distance(x, y)**2
                 for m, x, y in zip(self.manifolds, X, Y)]
        return np.sqrt(sum(dists))


class LieGroupManifold(ManifoldOptimizer):
    """
    Lie group manifolds (SO(n), SU(n), etc.)
    """

    def __init__(self, group_type: str, n: int):
        super().__init__(f"LieGroup_{group_type}")
        self.group_type = group_type
        self.n = n

    def project_SO(self, M: torch.Tensor) -> torch.Tensor:
        """Project onto SO(n) (special orthogonal group)"""
        U, _, Vh = torch.linalg.svd(M)
        R = U @ Vh
        # Ensure det(R) = 1
        if torch.det(R) < 0:
            U[:, -1] *= -1
            R = U @ Vh
        return R

    def project_SU(self, M: torch.Tensor) -> torch.Tensor:
        """Project onto SU(n) (special unitary group)"""
        # First make unitary
        U, _, Vh = torch.linalg.svd(M)
        U_proj = U @ Vh

        # Ensure det = 1 by adjusting phase
        det = torch.det(U_proj)
        phase = det / torch.abs(det)
        U_proj = U_proj / (phase ** (1/self.n))

        return U_proj

    def lie_algebra_projection(self, M: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        """Project onto Lie algebra (tangent space at identity)"""
        if self.group_type == "SO":
            # so(n): skew-symmetric matrices
            return (V - V.T) / 2
        elif self.group_type == "SU":
            # su(n): traceless skew-Hermitian matrices
            V_skew = (V - V.T.conj()) / 2
            return V_skew - torch.trace(V_skew) * torch.eye(self.n, device=V.device) / self.n


def measure_manifold_properties(manifold: ManifoldOptimizer, X: torch.Tensor) -> Dict[str, float]:
    """Measure various properties of a point on a manifold"""
    metrics = {}

    # Generate random tangent vector
    V = torch.randn_like(X)

    if hasattr(manifold, 'tangent_projection'):
        V_tan = manifold.tangent_projection(X, V)
        metrics['tangent_norm'] = torch.norm(V_tan).item()

    # Measure local curvature (if applicable)
    if isinstance(manifold, HyperbolicManifold):
        metrics['curvature'] = manifold.c

    # Measure projection error
    X_proj = manifold.project(X)
    metrics['projection_error'] = torch.norm(X - X_proj).item()

    return metrics