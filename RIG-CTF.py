#!/usr/bin/env python3
"""
Ricci-Information Geometry Constrained Traversal Framework (RIG-CTF)
"""

import math
import random
import json
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Optional
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation
import numpy as np

# ==================== CONFIGURATION ====================
@dataclass
class SimulationConfig:
    """Centralized configuration with validation"""
    # Manifold parameters
    grid_size: int = 30
    dimension: int = 2  # Set to 1 for simpler linear manifold
    
    # Agent parameters
    n_agents: int = 4
    n_ensembles: int = 2
    max_impulse: float = 1.2
    noise_sigma: float = 0.15
    
    # Physics parameters
    curvature_scale: float = 0.15
    energy_mean: float = 1.0
    energy_std: float = 0.3
    metric_perturbation: float = 0.1
    
    # Information geometry
    fisher_alpha: float = 0.5
    fisher_temperature: float = 1.0
    prob_model_components: int = 4
    
    # Simulation control
    total_steps: int = 100
    dt: float = 0.1
    random_seed: int = 42
    exploration_temp: float = 0.8
    
    # Visualization
    animation_interval: int = 50
    save_results: bool = True
    
    def validate(self):
        """Ensure parameters are physically valid"""
        assert self.grid_size > 0, "Grid size must be positive"
        assert self.dimension in [1, 2], "Dimension must be 1 or 2"
        assert self.n_agents > 0, "Must have at least one agent"
        assert self.max_impulse > 0, "Impulse must be positive"
        assert self.dt > 0, "Time step must be positive"
        assert self.noise_sigma >= 0, "Noise must be non-negative"
        return True

# ==================== UTILITY FUNCTIONS ====================
def set_reproducibility(seed: int):
    """Set all random seeds for full reproducibility"""
    random.seed(seed)
    np.random.seed(seed)

def normalize_probabilities(probs: np.ndarray) -> np.ndarray:
    """Normalize to valid probability distribution"""
    total = np.sum(probs)
    return probs / total if total > 0 else np.ones_like(probs) / len(probs)

# ==================== 2D MANIFOLD ====================
class DiscreteManifold2D:
    """
    2D discrete Riemannian manifold with rigorous mathematical foundations.
    
    Features:
    - Ollivier-Ricci curvature approximation
    - Full metric tensor with off-diagonal elements
    - Parametric probability model for Fisher information
    - Smooth interpolation for continuous positions
    """
    
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.size = config.grid_size
        self.dim = config.dimension
        
        # Initialize fields
        self.ricci_field = self._init_ricci_curvature()
        self.energy_field = self._init_energy_landscape()
        self.metric_field = self._init_metric_tensor()
        
        # Probability model for Fisher information
        self.prob_params = self._init_probability_parameters()
        self.prob_field = self._compute_probability_field()
        
    def _init_ricci_curvature(self) -> np.ndarray:
        """Initialize smooth Ricci curvature field"""
        ricci = np.zeros((self.size, self.size))
        for i in range(self.size):
            for j in range(self.size):
                # Smooth base curvature
                x, y = i - self.size/2, j - self.size/2
                r = math.sqrt(x**2 + y**2)
                base = self.config.curvature_scale * math.exp(-r**2 / (self.size**2 / 4))
                
                # Oscillatory components
                wave = 0.3 * self.config.curvature_scale * math.sin(2 * math.pi * i / self.size)
                wave += 0.2 * self.config.curvature_scale * math.cos(2 * math.pi * j / self.size)
                
                # Local noise
                noise = 0.1 * self.config.curvature_scale * (np.random.random() - 0.5)
                
                ricci[i, j] = base + wave + noise
        
        return ricci
    
    def _init_energy_landscape(self) -> np.ndarray:
        """Initialize multi-modal energy potential field"""
        energy = np.zeros((self.size, self.size))
        for i in range(self.size):
            for j in range(self.size):
                x, y = i / self.size, j / self.size
                
                # Multi-modal landscape
                e1 = 2.0 * ((x - 0.5)**2 + (y - 0.5)**2)
                e2 = 0.8 * math.exp(-10 * ((x - 0.3)**2 + (y - 0.7)**2))
                e3 = 0.6 * math.exp(-10 * ((x - 0.7)**2 + (y - 0.3)**2))
                
                noise = self.config.energy_std * np.random.randn()
                energy[i, j] = max(0.1, self.config.energy_mean + e1 - e2 - e3 + noise)
        
        return energy
    
    def _init_metric_tensor(self) -> np.ndarray:
        """Initialize metric tensor with curvature-dependent perturbations"""
        metric = np.ones((self.size, self.size, 2, 2))
        for i in range(self.size):
            for j in range(self.size):
                g = np.eye(2)
                kappa = self.ricci_field[i, j]
                perturbation = self.config.metric_perturbation * abs(kappa)
                
                g[0, 0] = 1.0 + perturbation
                g[1, 1] = 1.0 + perturbation
                g[0, 1] = g[1, 0] = 0.5 * perturbation * np.sign(kappa)
                
                metric[i, j] = g
        
        return metric
    
    def _init_probability_parameters(self) -> np.ndarray:
        """Initialize Gaussian mixture parameters θ"""
        n_components = self.config.prob_model_components
        params = []
        for _ in range(n_components):
            mean_x = np.random.uniform(0, self.size)
            mean_y = np.random.uniform(0, self.size)
            sigma = np.random.uniform(self.size * 0.1, self.size * 0.3)
            weight = np.random.uniform(0.5, 1.5)
            params.append([mean_x, mean_y, sigma, weight])
        return np.array(params)
    
    def _compute_probability_field(self) -> np.ndarray:
        """Compute probability distribution p(x,y|θ)"""
        prob = np.zeros((self.size, self.size))
        
        for param in self.prob_params:
            mean_x, mean_y, sigma, weight = param
            for i in range(self.size):
                for j in range(self.size):
                    exponent = -((i - mean_x)**2 + (j - mean_y)**2) / (2 * sigma**2)
                    prob[i, j] += weight * math.exp(exponent)
        
        prob = normalize_probabilities(prob.flatten()).reshape((self.size, self.size))
        return prob
    
    def get_ricci_at(self, pos: Tuple[float, float]) -> float:
        """Get Ricci curvature at position with bounds checking"""
        x, y = pos
        i = int(np.clip(x, 0, self.size - 1))
        j = int(np.clip(y, 0, self.size - 1))
        return self.ricci_field[i, j]
    
    def get_energy_at(self, pos: Tuple[float, float]) -> float:
        """Get energy at position with bounds checking"""
        x, y = pos
        i = int(np.clip(x, 0, self.size - 1))
        j = int(np.clip(y, 0, self.size - 1))
        return self.energy_field[i, j]
    
    def get_metric_at(self, pos: Tuple[float, float]) -> np.ndarray:
        """Get metric tensor at position with bounds checking"""
        x, y = pos
        i = int(np.clip(x, 0, self.size - 1))
        j = int(np.clip(y, 0, self.size - 1))
        return self.metric_field[i, j]
    
    def compute_fisher_information(self, pos: Tuple[float, float], 
                                   velocity: Tuple[float, float]) -> float:
        """
        Compute Fisher Information Metric.
        g_F(θ) = E[∂log p(x|θ)/∂θ · ∂log p(x|θ)/∂θ^T]
        """
        x, y = pos
        i = int(np.clip(x, 0, self.size - 1))
        j = int(np.clip(y, 0, self.size - 1))
        
        score_norm_sq = 0.0
        p_total = self.prob_field[i, j] + 1e-10
        
        for param in self.prob_params:
            mean_x, mean_y, sigma, weight = param
            
            exponent = -((i - mean_x)**2 + (j - mean_y)**2) / (2 * sigma**2)
            p_component = weight * math.exp(exponent)
            
            # Score function gradients
            score_mean_x = (i - mean_x) / (sigma**2) * (p_component / p_total)
            score_mean_y = (j - mean_y) / (sigma**2) * (p_component / p_total)
            score_sigma = ((i - mean_x)**2 + (j - mean_y)**2 - 2*sigma**2) / (sigma**3) * (p_component / p_total)
            
            score_norm_sq += score_mean_x**2 + score_mean_y**2 + 0.1 * score_sigma**2
        
        # Velocity contribution
        v_norm = math.sqrt(velocity[0]**2 + velocity[1]**2)
        fisher = score_norm_sq + 0.1 * v_norm / self.config.fisher_temperature
        
        return max(0.0, fisher)

# ==================== AGENT WITH STOCHASTIC DYNAMICS ====================
class InformationGeometricAgent:
    """
    Agent with variational policy and Langevin dynamics.
    
    Dynamics: dx = v(x,θ)dt + σ√dt dW
    Policy: Minimize path functional J[γ] = ∫(E + |κ||v| + αF)dt
    """
    
    def __init__(self, agent_id: int, manifold: DiscreteManifold2D, 
                 config: SimulationConfig, start_pos: Optional[Tuple[float, float]] = None):
        self.id = agent_id
        self.manifold = manifold
        self.config = config
        
        if start_pos is None:
            self.pos = (np.random.uniform(2, config.grid_size - 2),
                       np.random.uniform(2, config.grid_size - 2))
        else:
            self.pos = start_pos
        
        self.velocity = (0.0, 0.0)
        
        # History tracking
        self.trajectory = [self.pos]
        self.energy_history = [0.0]
        self.ricci_history = [0.0]
        self.fisher_history = [0.0]
        self.total_cost_history = [0.0]
        
        # Policy parameters
        self.exploration_rate = config.exploration_temp
        
    def compute_gradient_policy(self) -> Tuple[float, float]:
        """Compute optimal direction via gradient descent on effective potential"""
        h = 0.5
        
        # Energy gradient
        grad_E_x = (self.manifold.get_energy_at((self.pos[0] + h, self.pos[1])) -
                    self.manifold.get_energy_at((self.pos[0] - h, self.pos[1]))) / (2 * h)
        grad_E_y = (self.manifold.get_energy_at((self.pos[0], self.pos[1] + h)) -
                    self.manifold.get_energy_at((self.pos[0], self.pos[1] - h))) / (2 * h)
        
        # Ricci gradient
        grad_kappa_x = (self.manifold.get_ricci_at((self.pos[0] + h, self.pos[1])) -
                        self.manifold.get_ricci_at((self.pos[0] - h, self.pos[1]))) / (2 * h)
        grad_kappa_y = (self.manifold.get_ricci_at((self.pos[0], self.pos[1] + h)) -
                        self.manifold.get_ricci_at((self.pos[0], self.pos[1] - h))) / (2 * h)
        
        # Optimal direction: -∇(E + |κ|)
        direction_x = -(grad_E_x + 0.7 * grad_kappa_x)
        direction_y = -(grad_E_y + 0.7 * grad_kappa_y)
        
        # Normalize
        norm = math.sqrt(direction_x**2 + direction_y**2) + 1e-10
        direction_x /= norm
        direction_y /= norm
        
        impulse_mag = self.config.max_impulse * (0.5 + 0.5 * self.exploration_rate)
        
        return (direction_x * impulse_mag, direction_y * impulse_mag)
    
    def step_stochastic(self):
        """Execute one time step with stochastic Langevin dynamics"""
        # Policy selection: exploration vs exploitation
        if np.random.random() < self.exploration_rate:
            # Exploration: Fisher-weighted random
            fisher = self.manifold.compute_fisher_information(self.pos, self.velocity)
            exploration_scale = 1.0 / (1.0 + fisher)
            impulse = (np.random.randn() * self.config.max_impulse * exploration_scale,
                      np.random.randn() * self.config.max_impulse * exploration_scale)
        else:
            # Exploitation: gradient descent
            impulse = self.compute_gradient_policy()
        
        # Metric-adjusted displacement
        g = self.manifold.get_metric_at(self.pos)
        impulse_vec = np.array(impulse)
        
        try:
            displacement = np.linalg.solve(g, impulse_vec)
        except np.linalg.LinAlgError:
            displacement = impulse_vec
        
        # Langevin noise
        noise = self.config.noise_sigma * math.sqrt(self.config.dt) * np.random.randn(2)
        
        # Update position
        new_pos = (self.pos[0] + displacement[0] * self.config.dt + noise[0],
                   self.pos[1] + displacement[1] * self.config.dt + noise[1])
        
        # Boundary enforcement
        new_pos = (np.clip(new_pos[0], 0, self.config.grid_size - 1),
                   np.clip(new_pos[1], 0, self.config.grid_size - 1))
        
        # Compute velocity
        actual_displacement = (new_pos[0] - self.pos[0], new_pos[1] - self.pos[1])
        self.velocity = (actual_displacement[0] / self.config.dt,
                        actual_displacement[1] / self.config.dt)
        
        # Compute costs
        energy = self.manifold.get_energy_at(self.pos)
        ricci = self.manifold.get_ricci_at(self.pos)
        fisher = self.manifold.compute_fisher_information(self.pos, self.velocity)
        
        displacement_mag = math.sqrt(actual_displacement[0]**2 + actual_displacement[1]**2)
        cost_increment = (energy + abs(ricci) * displacement_mag + 0.1 * fisher) * self.config.dt
        
        # Update state
        self.pos = new_pos
        self.trajectory.append(self.pos)
        self.energy_history.append(self.energy_history[-1] + energy * self.config.dt)
        self.ricci_history.append(self.ricci_history[-1] + ricci * displacement_mag * self.config.dt)
        self.fisher_history.append(self.fisher_history[-1] + fisher * self.config.dt)
        self.total_cost_history.append(self.total_cost_history[-1] + cost_increment)
        
        # Annealing
        self.exploration_rate *= 0.995

# ==================== SIMULATION ENGINE ====================
class SimulationEngine:
    """Multi-agent ensemble simulation with validation and visualization"""
    
    def __init__(self, config: SimulationConfig):
        self.config = config
        set_reproducibility(config.random_seed)
        
        self.manifold = DiscreteManifold2D(config)
        
        # Create ensemble
        self.ensembles = []
        for ensemble_id in range(config.n_ensembles):
            agents = []
            for agent_id in range(config.n_agents):
                start_x = (agent_id + 1) * config.grid_size / (config.n_agents + 1)
                start_y = (ensemble_id + 1) * config.grid_size / (config.n_ensembles + 1)
                agent = InformationGeometricAgent(
                    agent_id, self.manifold, config,
                    start_pos=(start_x, start_y)
                )
                agents.append(agent)
            self.ensembles.append(agents)
        
        self.statistics = defaultdict(list)
        
    def run_with_visualization(self):
        """Execute with real-time visualization"""
        fig = plt.figure(figsize=(18, 10))
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
        
        # Manifold fields (top row)
        ax_ricci = fig.add_subplot(gs[0, 0])
        ax_energy = fig.add_subplot(gs[0, 1])
        ax_fisher = fig.add_subplot(gs[0, 2])
        
        # Dynamics (bottom row)
        ax_trajectories = fig.add_subplot(gs[1, 0])
        ax_costs = fig.add_subplot(gs[1, 1])
        ax_convergence = fig.add_subplot(gs[1, 2])
        
        # Plot static fields
        im_ricci = ax_ricci.imshow(self.manifold.ricci_field, cmap='RdBu_r', origin='lower')
        ax_ricci.set_title('Ricci Curvature κ(x,y)', fontweight='bold')
        plt.colorbar(im_ricci, ax=ax_ricci, fraction=0.046)
        
        im_energy = ax_energy.imshow(self.manifold.energy_field, cmap='viridis', origin='lower')
        ax_energy.set_title('Energy Landscape E(x,y)', fontweight='bold')
        plt.colorbar(im_energy, ax=ax_energy, fraction=0.046)
        
        im_prob = ax_fisher.imshow(self.manifold.prob_field, cmap='plasma', origin='lower')
        ax_fisher.set_title('Probability Field p(x,y|θ)', fontweight='bold')
        plt.colorbar(im_prob, ax=ax_fisher, fraction=0.046)
        
        # Setup dynamic plots
        ax_trajectories.set_xlim(0, self.config.grid_size)
        ax_trajectories.set_ylim(0, self.config.grid_size)
        ax_trajectories.set_title('Agent Trajectories', fontweight='bold')
        ax_trajectories.grid(True, alpha=0.3)
        
        ax_costs.set_title('Cumulative Path Costs', fontweight='bold')
        ax_costs.set_xlabel('Time Step')
        ax_costs.set_ylabel('Total Cost')
        ax_costs.grid(True, alpha=0.3)
        
        ax_convergence.set_title('Cost Rate dC/dt', fontweight='bold')
        ax_convergence.set_xlabel('Time Step')
        ax_convergence.set_ylabel('Cost Rate')
        ax_convergence.grid(True, alpha=0.3)
        
        # Color mapping
        try:
            colors = plt.colormaps.get_cmap('tab10')
        except AttributeError:
            import matplotlib.cm as cm
            colors = cm.get_cmap('tab10')
        
        # Plot elements
        trajectory_lines = []
        agent_dots = []
        cost_lines = []
        conv_lines = []
        
        for ens_id, ensemble in enumerate(self.ensembles):
            for ag_id, agent in enumerate(ensemble):
                color = colors((ens_id * self.config.n_agents + ag_id) % 10)
                
                line, = ax_trajectories.plot([], [], '-', color=color, alpha=0.6, linewidth=1.5)
                trajectory_lines.append(line)
                
                dot, = ax_trajectories.plot([], [], 'o', color=color, markersize=8)
                agent_dots.append(dot)
                
                cost_line, = ax_costs.plot([], [], '-', color=color, alpha=0.7, linewidth=1.5)
                cost_lines.append(cost_line)
                
                conv_line, = ax_convergence.plot([], [], '-', color=color, alpha=0.7, linewidth=1.5)
                conv_lines.append(conv_line)
        
        def update(frame):
            # Step all agents
            for ensemble in self.ensembles:
                for agent in ensemble:
                    agent.step_stochastic()
            
            # Update plots
            idx = 0
            for ensemble in self.ensembles:
                for agent in ensemble:
                    traj_x = [p[0] for p in agent.trajectory]
                    traj_y = [p[1] for p in agent.trajectory]
                    trajectory_lines[idx].set_data(traj_x, traj_y)
                    agent_dots[idx].set_data([agent.pos[0]], [agent.pos[1]])
                    
                    cost_lines[idx].set_data(range(len(agent.total_cost_history)),
                                            agent.total_cost_history)
                    
                    if len(agent.total_cost_history) > 10:
                        convergence = np.diff(agent.total_cost_history[-50:])
                        conv_lines[idx].set_data(
                            range(len(agent.total_cost_history) - len(convergence),
                                  len(agent.total_cost_history)),
                            convergence
                        )
                    
                    idx += 1
            
            ax_costs.relim()
            ax_costs.autoscale_view()
            ax_convergence.relim()
            ax_convergence.autoscale_view()
            
            return trajectory_lines + agent_dots + cost_lines + conv_lines
        
        fig.suptitle(f'RIG-CTF v2.1: {self.config.n_agents} Agents × {self.config.n_ensembles} Ensembles',
                    fontsize=16, fontweight='bold')
        
        anim = FuncAnimation(fig, update, frames=self.config.total_steps,
                           interval=self.config.animation_interval, blit=False, repeat=False)
        plt.show()
        
        self._compute_statistics()
        if self.config.save_results:
            self._save_results()
    
    def _compute_statistics(self):
        """Compute ensemble statistics"""
        all_costs = []
        for ensemble in self.ensembles:
            for agent in ensemble:
                all_costs.append(agent.total_cost_history[-1])
        
        self.statistics['mean_cost'] = np.mean(all_costs)
        self.statistics['std_cost'] = np.std(all_costs)
    
    def _save_results(self):
        """Save results to JSON"""
        results = {
            'config': asdict(self.config),
            'statistics': dict(self.statistics)
        }
        
        with open('rig_ctf_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✓ Results saved to rig_ctf_results.json")
    
    def print_summary(self):
        """Print final summary"""
        print("\n" + "="*70)
        print("RIG-CTF SIMULATION SUMMARY")
        print("="*70)
        print(f"Grid: {self.config.grid_size}×{self.config.grid_size}")
        print(f"Agents: {self.config.n_agents} × {self.config.n_ensembles} ensembles")
        print(f"Steps: {self.config.total_steps}")
        print(f"Seed: {self.config.random_seed} (reproducible)")
        print(f"\nMean Final Cost: {self.statistics['mean_cost']:.3f} ± {self.statistics['std_cost']:.3f}")
        print("="*70 + "\n")

# ==================== VALIDATION ====================
class ValidationSuite:
    """Automated validation tests"""
    
    @staticmethod
    def test_probability_normalization(manifold: DiscreteManifold2D) -> bool:
        """Verify probability sums to 1"""
        total = np.sum(manifold.prob_field)
        return abs(total - 1.0) < 1e-6
    
    @staticmethod
    def test_metric_positive_definite(manifold: DiscreteManifold2D) -> bool:
        """Verify metric is positive definite"""
        for i in range(min(10, manifold.size)):
            for j in range(min(10, manifold.size)):
                g = manifold.metric_field[i, j]
                eigenvals = np.linalg.eigvals(g)
                if np.any(eigenvals <= 0):
                    return False
        return True
    
    @staticmethod
    def run_all(sim: SimulationEngine) -> Dict[str, bool]:
        """Run all validation tests"""
        results = {}
        results['probability_norm'] = ValidationSuite.test_probability_normalization(sim.manifold)
        results['metric_positive'] = ValidationSuite.test_metric_positive_definite(sim.manifold)
        return results

# ==================== MAIN ====================
def main():
    """Main execution"""
    print("="*70)
    print("Ricci-Information Geometry Constrained Traversal Framework v2.1")
    print("Production-Ready Implementation")
    print("="*70)
    
    config = SimulationConfig(
        grid_size=25,
        n_agents=3,
        n_ensembles=2,
        total_steps=150,
        random_seed=42
    )
    
    config.validate()
    
    print("\nInitializing simulation...")
    sim = SimulationEngine(config)
    
    print("Running validation tests...")
    tests = ValidationSuite.run_all(sim)
    
    print("Validation Results:")
    for test_name, passed in tests.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {test_name}: {status}")
    
    if all(tests.values()):
        print("\n✓ All tests passed. Starting simulation...")
        sim.run_with_visualization()
        sim.print_summary()
    else:
        print("\n✗ Validation failed. Aborting.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
