#!/usr/bin/env python3
"""
RIG-CTF v3.1: Geodesic Deviation & Curvature Analysis
A vectorized engine measuring the divergence between Euclidean and Riemannian trajectories.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from dataclasses import dataclass

@dataclass
class RIGConfig:
    grid_size: int = 60
    n_agents: int = 15
    total_steps: int = 200
    dt: float = 0.15
    curvature_scale: float = 0.8
    noise_floor: float = 0.04
    metric_warp: float = 0.5
    seed: int = 42

class MetricsEngine:
    def __init__(self, cfg: RIGConfig):
        self.cfg = cfg
        np.random.seed(cfg.seed)
        
        # 1. Manifold Setup
        self.size = cfg.grid_size
        y, x = np.mgrid[0:self.size, 0:self.size]
        
        # Ricci and Metric Tensor
        r_sq = (x - self.size/2)**2 + (y - self.size/2)**2
        self.ricci = cfg.curvature_scale * np.exp(-r_sq / (self.size**2 / 4))
        self.g = np.zeros((self.size, self.size, 2, 2))
        self.g[:] = np.eye(2)
        self.g[..., 0, 0] += cfg.metric_warp * np.abs(self.ricci)
        self.g[..., 1, 1] += cfg.metric_warp * np.abs(self.ricci)
        
        # Pre-compute Energy and Gradient
        self.energy = 0.5 * (((x - self.size/2)/self.size)**2 + ((y - self.size/2)/self.size)**2)
        grad_y, grad_x = np.gradient(self.energy)
        self.grad_e = np.stack([grad_x, grad_y], axis=-1)

        # 2. Agent State
        self.pos = np.random.uniform(self.size*0.3, self.size*0.7, (cfg.n_agents, 2))
        self.euclidean_pos = self.pos.copy() # Parallel shadow-agent in flat space
        self.vel = np.zeros((cfg.n_agents, 2))
        self.euc_vel = np.zeros((cfg.n_agents, 2))
        
        # 3. Metric Tracking
        self.deviation_history = []
        self.time_axis = []

    def step(self):
        idx = self.pos.astype(int).clip(0, self.size - 1)
        
        # Riemannian Update (The Warped Path)
        g_batch = self.g[idx[:, 0], idx[:, 1]]
        grad_batch = self.grad_e[idx[:, 0], idx[:, 1]]
        accel = np.linalg.solve(g_batch, -grad_batch[..., np.newaxis]).squeeze(-1)
        
        # Euclidean Update (The "Straight" Path)
        # In flat space, g is identity, so accel is just -grad_e
        accel_euc = -grad_batch 
        
        noise = self.cfg.noise_floor * np.random.randn(self.cfg.n_agents, 2)
        
        # Update Riemannian state
        self.vel = 0.9 * self.vel + accel * self.cfg.dt + noise
        self.pos = np.clip(self.pos + self.vel, 0, self.size - 1)
        
        # Update Euclidean state
        self.euc_vel = 0.9 * self.euc_vel + accel_euc * self.cfg.dt + noise
        self.euclidean_pos = np.clip(self.euclidean_pos + self.euc_vel, 0, self.size - 1)
        
        # Calculate Deviation: Euclidean distance between Riemannian and Flat paths
        deviation = np.linalg.norm(self.pos - self.euclidean_pos, axis=1).mean()
        return deviation

def run_simulation():
    cfg = RIGConfig()
    engine = MetricsEngine(cfg)
    
    fig = plt.figure(figsize=(15, 8), facecolor='#0a0a0a')
    gs = fig.add_gridspec(2, 2)
    
    # 1. Main Trajectory View
    ax_main = fig.add_subplot(gs[:, 0])
    ax_main.imshow(engine.ricci.T, origin='lower', cmap='magma', alpha=0.6)
    dots_riemann = ax_main.scatter([], [], c='cyan', s=30, label='Riemannian (Actual)')
    dots_euclidean = ax_main.scatter([], [], c='white', s=10, alpha=0.5, label='Euclidean (Reference)')
    ax_main.set_title("Manifold Traversal Comparison", color='white')
    ax_main.legend()
    
    # 2. Geodesic Deviation Plot
    ax_dev = fig.add_subplot(gs[0, 1])
    dev_line, = ax_dev.plot([], [], c='cyan', lw=2)
    ax_dev.set_facecolor('#111111')
    ax_dev.set_title("Mean Geodesic Deviation (L² Norm)", color='white')
    ax_dev.set_ylabel("Distance Divergence", color='white')
    
    # 3. Local Metric Tension (Determinant)
    ax_det = fig.add_subplot(gs[1, 1])
    det_field = np.linalg.det(engine.g)
    im_det = ax_det.imshow(det_field.T, origin='lower', cmap='plasma')
    ax_det.set_title("Local Metric Determinant |g|", color='white')
    plt.colorbar(im_det, ax=ax_det)

    for ax in [ax_main, ax_dev, ax_det]:
        ax.tick_params(colors='white')

    def update(f):
        dev = engine.step()
        engine.deviation_history.append(dev)
        engine.time_axis.append(f)
        
        # Update Dots
        dots_riemann.set_offsets(engine.pos)
        dots_euclidean.set_offsets(engine.euclidean_pos)
        
        # Update Deviation Graph
        dev_line.set_data(engine.time_axis, engine.deviation_history)
        ax_dev.set_xlim(0, max(20, f))
        ax_dev.set_ylim(0, max(0.5, max(engine.deviation_history) * 1.1))
        
        return dots_riemann, dots_euclidean, dev_line

    ani = FuncAnimation(fig, update, frames=cfg.total_steps, interval=30, blit=False)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_simulation()