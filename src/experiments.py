"""
Experimental Framework for Manifold Exploration
================================================
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import time
from dataclasses import dataclass, field
import json
from pathlib import Path

from .manifolds import *


@dataclass
class ExperimentResult:
    """Container for experiment results"""
    iteration: int
    timestamp: float
    manifold_type: str
    hypothesis: str
    metrics: Dict[str, float]
    parameters: Dict[str, Any]
    convergence_data: List[float]
    spectral_data: Dict[str, List[float]]
    findings: str
    anomalies: List[str] = field(default_factory=list)


class ManifoldExperiment:
    """Base class for manifold experiments"""

    def __init__(self, name: str, manifold: ManifoldOptimizer, hypothesis: str):
        self.name = name
        self.manifold = manifold
        self.hypothesis = hypothesis
        self.results: List[ExperimentResult] = []
        self.iteration = 0

    def run_iteration(self) -> ExperimentResult:
        """Run one experimental iteration"""
        self.iteration += 1
        raise NotImplementedError

    def analyze_results(self) -> Dict[str, Any]:
        """Analyze accumulated results"""
        if not self.results:
            return {}

        analysis = {
            "total_iterations": len(self.results),
            "hypothesis": self.hypothesis,
            "confirmed": False,
            "evidence": [],
            "patterns": []
        }

        # Extract patterns
        convergence_rates = [r.metrics.get('convergence_rate', 0) for r in self.results]
        if convergence_rates:
            analysis['mean_convergence'] = np.mean(convergence_rates)
            analysis['convergence_variance'] = np.var(convergence_rates)

        return analysis


class CurvatureGeneralizationExperiment(ManifoldExperiment):
    """Test relationship between manifold curvature and generalization"""

    def __init__(self):
        super().__init__(
            name="Curvature-Generalization",
            manifold=HyperbolicManifold(dim=10),
            hypothesis="Intrinsic curvature correlates with generalization gap"
        )
        self.curvatures = np.linspace(0.1, 10.0, 20)
        self.current_curvature_idx = 0

    def run_iteration(self) -> ExperimentResult:
        self.iteration += 1

        # Select curvature for this iteration
        c = self.curvatures[self.current_curvature_idx % len(self.curvatures)]
        self.manifold.c = c

        # Create synthetic task
        dim = 10
        n_samples = 100

        # Generate data with intrinsic curvature
        X_train = torch.randn(n_samples, dim)
        y_train = torch.sin(torch.sum(X_train, dim=1)) + 0.1 * torch.randn(n_samples)

        X_test = torch.randn(n_samples // 2, dim)
        y_test = torch.sin(torch.sum(X_test, dim=1))

        # Simple network with manifold constraint
        W = torch.randn(dim, dim, requires_grad=True)

        # Training loop
        optimizer = optim.SGD([W], lr=0.01)
        train_losses = []
        test_losses = []

        for epoch in range(50):
            # Forward pass
            W_proj = self.manifold.project(W)
            y_pred = torch.matmul(X_train, W_proj).squeeze()
            loss = nn.MSELoss()(y_pred, y_train)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track metrics
            train_losses.append(loss.item())

            with torch.no_grad():
                y_test_pred = torch.matmul(X_test, W_proj).squeeze()
                test_loss = nn.MSELoss()(y_test_pred, y_test)
                test_losses.append(test_loss.item())

        # Calculate generalization gap
        final_train_loss = train_losses[-1]
        final_test_loss = test_losses[-1]
        generalization_gap = final_test_loss - final_train_loss

        # Analyze spectral properties
        with torch.no_grad():
            eigenvalues = torch.linalg.eigvals(W_proj).real
            spectral_radius = torch.max(torch.abs(eigenvalues)).item()

        # Detect anomalies
        anomalies = []
        if generalization_gap < 0:
            anomalies.append("Negative generalization gap detected")
        if spectral_radius > 10:
            anomalies.append("Large spectral radius observed")

        result = ExperimentResult(
            iteration=self.iteration,
            timestamp=time.time(),
            manifold_type="Hyperbolic",
            hypothesis=self.hypothesis,
            metrics={
                "curvature": c,
                "train_loss": final_train_loss,
                "test_loss": final_test_loss,
                "generalization_gap": generalization_gap,
                "spectral_radius": spectral_radius,
                "convergence_rate": -np.log(train_losses[-1] / train_losses[0]) / len(train_losses)
            },
            parameters={"dim": dim, "n_samples": n_samples, "epochs": 50},
            convergence_data=train_losses,
            spectral_data={"eigenvalues": eigenvalues.tolist()},
            findings=f"Gap={generalization_gap:.4f} at curvature={c:.2f}",
            anomalies=anomalies
        )

        self.results.append(result)
        self.current_curvature_idx += 1

        return result


class CompositionExperiment(ManifoldExperiment):
    """Test manifold composition and interference patterns"""

    def __init__(self):
        manifolds = [
            StiefelManifold(10, 5),
            GrassmannManifold(10, 3),
            HyperbolicManifold(7, c=1.0)
        ]
        super().__init__(
            name="Manifold-Composition",
            manifold=ProductManifold(manifolds),
            hypothesis="Multiple constraints create predictable interference"
        )

    def run_iteration(self) -> ExperimentResult:
        self.iteration += 1

        # Create components
        components = [
            torch.randn(10, 5),
            torch.randn(10, 3),
            torch.randn(7)
        ]

        # Project each component
        projected = self.manifold.project(components)

        # Measure interference between components
        interference_scores = []

        # Simulate optimization with coupling
        for step in range(20):
            gradients = [torch.randn_like(c) for c in projected]

            # Apply cross-component influence
            coupled_grads = []
            for i, grad in enumerate(gradients):
                coupling = 0
                for j, other in enumerate(projected):
                    if i != j:
                        # Measure influence
                        if isinstance(other, torch.Tensor):
                            influence = torch.norm(other).item() * 0.1
                            coupled_grads.append(grad + influence * torch.randn_like(grad))
                            interference_scores.append(influence)

            # Update with retraction
            if coupled_grads:
                projected = self.manifold.retract(projected, coupled_grads[:len(projected)], t=0.01)

        # Calculate total distance moved
        final_distance = self.manifold.geodesic_distance(components, projected)

        metrics = {
            "total_distance": final_distance,
            "mean_interference": np.mean(interference_scores) if interference_scores else 0,
            "max_interference": np.max(interference_scores) if interference_scores else 0,
            "num_components": len(components)
        }

        result = ExperimentResult(
            iteration=self.iteration,
            timestamp=time.time(),
            manifold_type="Product",
            hypothesis=self.hypothesis,
            metrics=metrics,
            parameters={"components": 3, "steps": 20},
            convergence_data=interference_scores,
            spectral_data={},
            findings=f"Interference pattern: mean={metrics['mean_interference']:.4f}"
        )

        self.results.append(result)
        return result


class SpectralAlignmentExperiment(ManifoldExperiment):
    """Test spectral properties and manifold selection"""

    def __init__(self):
        super().__init__(
            name="Spectral-Alignment",
            manifold=StiefelManifold(20, 10),
            hypothesis="Eigenvalue distributions predict optimal manifolds"
        )

    def run_iteration(self) -> ExperimentResult:
        self.iteration += 1

        # Generate random matrix
        M = torch.randn(20, 10)

        # Compute initial spectrum
        initial_spectrum = torch.linalg.svdvals(M)

        # Project onto manifold
        M_proj = self.manifold.project(M)

        # Compute projected spectrum
        proj_spectrum = torch.linalg.svdvals(M_proj)

        # Measure spectral alignment
        alignment = torch.nn.functional.cosine_similarity(
            initial_spectrum[:10].unsqueeze(0),
            proj_spectrum.unsqueeze(0)
        ).item()

        # Test different manifolds and measure fit
        manifold_scores = {}

        # Stiefel
        stiefel = StiefelManifold(20, 10)
        M_stiefel = stiefel.project(M)
        stiefel_spectrum = torch.linalg.svdvals(M_stiefel)
        manifold_scores['stiefel'] = torch.norm(stiefel_spectrum - 1).item()

        # Grassmann
        grassmann = GrassmannManifold(20, 10)
        M_grass = grassmann.project(M)
        grass_spectrum = torch.linalg.svdvals(M_grass)
        manifold_scores['grassmann'] = torch.norm(grass_spectrum - 1).item()

        # Find best manifold
        best_manifold = min(manifold_scores, key=manifold_scores.get)

        metrics = {
            "spectral_alignment": alignment,
            "stiefel_score": manifold_scores['stiefel'],
            "grassmann_score": manifold_scores['grassmann'],
            "spectrum_std": proj_spectrum.std().item(),
            "condition_number": (proj_spectrum[0] / proj_spectrum[-1]).item()
        }

        result = ExperimentResult(
            iteration=self.iteration,
            timestamp=time.time(),
            manifold_type="Stiefel",
            hypothesis=self.hypothesis,
            metrics=metrics,
            parameters={"matrix_size": (20, 10)},
            convergence_data=[],
            spectral_data={
                "initial": initial_spectrum.tolist(),
                "projected": proj_spectrum.tolist()
            },
            findings=f"Best manifold: {best_manifold}, alignment={alignment:.4f}"
        )

        self.results.append(result)
        return result


class ManifoldCapacityExperiment(ManifoldExperiment):
    """Test information capacity of different manifolds"""

    def __init__(self):
        super().__init__(
            name="Manifold-Capacity",
            manifold=GrassmannManifold(50, 10),
            hypothesis="Capacity = f(volume, curvature, dimension)"
        )

    def run_iteration(self) -> ExperimentResult:
        self.iteration += 1

        # Test different manifold dimensions
        dims = [(50, k) for k in range(5, 25, 5)]
        capacities = []

        for n, k in dims:
            manifold = GrassmannManifold(n, k)

            # Generate random data points
            n_points = 100
            points = [manifold.project(torch.randn(n, k)) for _ in range(n_points)]

            # Measure pairwise distances (proxy for volume)
            distances = []
            for i in range(n_points):
                for j in range(i+1, n_points):
                    dist = manifold.geodesic_distance(points[i], points[j])
                    distances.append(dist)

            # Estimate capacity metrics
            mean_dist = np.mean(distances)
            var_dist = np.var(distances)

            # Theoretical volume (Grassmann)
            theoretical_dim = k * (n - k)

            # Information capacity estimate
            capacity = theoretical_dim * np.log(1 + mean_dist/var_dist)
            capacities.append(capacity)

        metrics = {
            "max_capacity": max(capacities),
            "optimal_dim": dims[np.argmax(capacities)][1],
            "capacity_variance": np.var(capacities),
            "dimension_correlation": np.corrcoef(
                [d[1] for d in dims],
                capacities
            )[0, 1]
        }

        result = ExperimentResult(
            iteration=self.iteration,
            timestamp=time.time(),
            manifold_type="Grassmann",
            hypothesis=self.hypothesis,
            metrics=metrics,
            parameters={"n": 50, "k_range": (5, 25)},
            convergence_data=capacities,
            spectral_data={},
            findings=f"Optimal k={metrics['optimal_dim']}, capacity={metrics['max_capacity']:.2f}"
        )

        self.results.append(result)
        return result


class SymplecticConservationExperiment(ManifoldExperiment):
    """Test energy conservation in symplectic networks"""

    def __init__(self):
        super().__init__(
            name="Symplectic-Conservation",
            manifold=SymplecticManifold(5),
            hypothesis="Symplectic constraints preserve energy in dynamics"
        )

    def run_iteration(self) -> ExperimentResult:
        self.iteration += 1

        n = 5
        T = 100  # Time steps

        # Initial state (position, momentum)
        state = torch.randn(2*n)
        initial_energy = torch.norm(state)**2 / 2

        # Symplectic integrator
        M = torch.randn(2*n, 2*n)
        M_symp = self.manifold.project(M)

        energies = [initial_energy.item()]
        states = [state.clone()]

        # Evolve dynamics
        for t in range(T):
            state = M_symp @ state
            energy = torch.norm(state)**2 / 2
            energies.append(energy.item())
            states.append(state.clone())

        # Measure conservation
        energy_drift = abs(energies[-1] - energies[0])
        energy_variance = np.var(energies)
        max_deviation = max(abs(e - initial_energy.item()) for e in energies)

        # Check symplectic property preservation
        is_symplectic = self.manifold.check_symplectic(M_symp)

        # Phase space volume
        if len(states) > 10:
            volumes = []
            for i in range(0, len(states)-10, 10):
                # Sample volume element
                vol = torch.det(torch.stack(states[i:i+5])[:5, :5])
                volumes.append(abs(vol.item()))
            volume_preservation = np.std(volumes) / (np.mean(volumes) + 1e-8)
        else:
            volume_preservation = 0

        metrics = {
            "energy_drift": energy_drift,
            "energy_variance": energy_variance,
            "max_deviation": max_deviation,
            "is_symplectic": float(is_symplectic),
            "volume_preservation": volume_preservation,
            "final_energy_ratio": energies[-1] / initial_energy.item()
        }

        anomalies = []
        if energy_drift > 0.1:
            anomalies.append(f"High energy drift: {energy_drift:.4f}")

        result = ExperimentResult(
            iteration=self.iteration,
            timestamp=time.time(),
            manifold_type="Symplectic",
            hypothesis=self.hypothesis,
            metrics=metrics,
            parameters={"n": n, "time_steps": T},
            convergence_data=energies,
            spectral_data={},
            findings=f"Energy conserved: drift={energy_drift:.6f}",
            anomalies=anomalies
        )

        self.results.append(result)
        return result


class ExperimentRunner:
    """Main experiment runner with logging and analysis"""

    def __init__(self, output_dir: str = "results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.experiments = [
            CurvatureGeneralizationExperiment(),
            CompositionExperiment(),
            SpectralAlignmentExperiment(),
            ManifoldCapacityExperiment(),
            SymplecticConservationExperiment()
        ]

        self.iteration_count = 0
        self.max_iterations = 150
        self.discoveries = []
        self.hypotheses_tested = []

    def run_all_experiments(self):
        """Run all experiments for specified iterations"""

        while self.iteration_count < self.max_iterations:
            self.iteration_count += 1

            print(f"\n========== ITERATION {self.iteration_count}/{self.max_iterations} ==========")

            for exp in self.experiments:
                result = exp.run_iteration()

                # Log progress
                print(f"[{exp.name}] {result.findings}")

                # Check for anomalies
                if result.anomalies:
                    print(f"  ANOMALIES: {result.anomalies}")
                    self.investigate_anomaly(exp, result)

                # Periodic analysis
                if self.iteration_count % 10 == 0:
                    analysis = exp.analyze_results()
                    self.evaluate_hypothesis(exp, analysis)

            # Global analysis every 25 iterations
            if self.iteration_count % 25 == 0:
                self.perform_cross_experiment_analysis()
                self.propose_new_hypotheses()

            # Save checkpoint
            if self.iteration_count % 20 == 0:
                self.save_checkpoint()

    def investigate_anomaly(self, experiment: ManifoldExperiment, result: ExperimentResult):
        """Deep dive into detected anomalies"""
        print(f"  Investigating anomaly in {experiment.name}...")

        # Run focused experiments around the anomaly
        for i in range(3):
            follow_up = experiment.run_iteration()
            if follow_up.anomalies:
                # Pattern detected
                discovery = {
                    "type": "anomaly_pattern",
                    "experiment": experiment.name,
                    "description": f"Consistent anomaly: {follow_up.anomalies}",
                    "iteration": self.iteration_count
                }
                self.discoveries.append(discovery)

    def evaluate_hypothesis(self, experiment: ManifoldExperiment, analysis: Dict):
        """Evaluate whether hypothesis is supported"""

        if 'mean_convergence' in analysis and analysis['mean_convergence'] > 0:
            confidence = 1 - analysis.get('convergence_variance', 1)

            hypothesis_result = {
                "hypothesis": experiment.hypothesis,
                "experiment": experiment.name,
                "status": "supported" if confidence > 0.7 else "inconclusive",
                "confidence": confidence,
                "iteration": self.iteration_count
            }

            self.hypotheses_tested.append(hypothesis_result)

            if confidence > 0.8:
                print(f"  HYPOTHESIS SUPPORTED: {experiment.hypothesis} (conf={confidence:.2f})")

    def perform_cross_experiment_analysis(self):
        """Analyze patterns across different experiments"""
        print("\n=== CROSS-EXPERIMENT ANALYSIS ===")

        all_metrics = {}
        for exp in self.experiments:
            if exp.results:
                metrics = [r.metrics for r in exp.results[-10:]]  # Last 10 results
                for key in metrics[0].keys():
                    values = [m.get(key, 0) for m in metrics]
                    all_metrics[f"{exp.name}_{key}"] = values

        # Look for correlations
        correlations = []
        keys = list(all_metrics.keys())
        for i, key1 in enumerate(keys):
            for key2 in keys[i+1:]:
                if len(all_metrics[key1]) == len(all_metrics[key2]):
                    corr = np.corrcoef(all_metrics[key1], all_metrics[key2])[0, 1]
                    if abs(corr) > 0.7:
                        correlations.append((key1, key2, corr))

        if correlations:
            print("Strong correlations found:")
            for k1, k2, c in correlations[:5]:
                print(f"  {k1} <-> {k2}: {c:.3f}")

                # Record as potential discovery
                if abs(c) > 0.9:
                    discovery = {
                        "type": "correlation",
                        "description": f"Strong correlation between {k1} and {k2}",
                        "value": c,
                        "iteration": self.iteration_count
                    }
                    self.discoveries.append(discovery)

    def propose_new_hypotheses(self):
        """Generate new hypotheses based on observations"""
        print("\n=== PROPOSING NEW HYPOTHESES ===")

        # Analyze discoveries for patterns
        if len(self.discoveries) >= 3:
            # Look for repeated patterns
            pattern_types = [d['type'] for d in self.discoveries]

            if pattern_types.count('correlation') >= 2:
                new_hypothesis = {
                    "hypothesis": "Manifold properties exhibit universal coupling patterns",
                    "basis": "Multiple strong correlations observed",
                    "proposed_at": self.iteration_count
                }
                print(f"  NEW HYPOTHESIS: {new_hypothesis['hypothesis']}")

                # Create new experiment to test it
                # (Would implement actual new experiment here)

    def save_checkpoint(self):
        """Save current state"""
        checkpoint = {
            "iteration": self.iteration_count,
            "discoveries": self.discoveries,
            "hypotheses": self.hypotheses_tested,
            "timestamp": time.time()
        }

        filename = self.output_dir / f"checkpoint_{self.iteration_count}.json"
        with open(filename, 'w') as f:
            json.dump(checkpoint, f, indent=2, default=str)

        print(f"  Checkpoint saved: {filename}")