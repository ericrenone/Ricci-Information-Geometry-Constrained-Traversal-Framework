#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from dataclasses import dataclass

@dataclass
class RIGConfig:
    grid_size: int = 60
    n_agents: int = 25
    dt: float = 0.15
    total_steps: int = 300
    curvature_intensity: float = 0.5
    noise_floor: float = 0.05
    seed: int = 42

class VectorizedManifold:
    def __init__(self, cfg: RIGConfig):
        self.cfg = cfg
        self.size = cfg.grid_size
        y, x = np.mgrid[0:self.size, 0:self.size]
        
        # 1. Ricci Curvature (Smooth interference pattern)
        r_sq = (x - self.size/2)**2 + (y - self.size/2)**2
        self.ricci = cfg.curvature_intensity * np.exp(-r_sq / (self.size**2 / 4))
        self.ricci += 0.1 * np.sin(0.15 * x) * np.cos(0.15 * y)

        # 2. Energy Landscape & Pre-computed Gradient
        # A simple central potential well
        self.energy = 0.5 * (((x - self.size/2)/self.size)**2 + ((y - self.size/2)/self.size)**2)
        grad_y, grad_x = np.gradient(self.energy)
        self.grad_e = np.stack([grad_x, grad_y], axis=-1)

        # 3. Metric Tensor Field g_ij (Shape: size, size, 2, 2)
        # We ensure it's a valid Riemannian metric (symmetric, positive definite)
        self.g = np.zeros((self.size, self.size, 2, 2))
        eye = np.eye(2)
        # Base Euclidean metric + curvature warping
        self.g[:] = eye 
        deformation = np.abs(self.ricci)
        self.g[..., 0, 0] += deformation
        self.g[..., 1, 1] += deformation
        self.g[..., 0, 1] = self.g[..., 1, 0] = 0.2 * self.ricci

def execute():
    cfg = RIGConfig()
    np.random.seed(cfg.seed)
    m = VectorizedManifold(cfg)

    # Agent state: (n_agents, 2)
    pos = np.random.uniform(m.size*0.2, m.size*0.8, (cfg.n_agents, 2))
    vel = np.zeros((cfg.n_agents, 2))
    # Path history for drawing lines (Time, Agents, XY)
    history = np.full((cfg.total_steps, cfg.n_agents, 2), np.nan)

    fig, ax = plt.subplots(figsize=(10, 8), facecolor='#0f0f0f')
    ax.imshow(m.energy.T, origin='lower', cmap='magma', extent=[0, m.size, 0, m.size], alpha=0.7)
    
    lines = [ax.plot([], [], lw=1.2, alpha=0.5, c='cyan')[0] for _ in range(cfg.n_agents)]
    dots = ax.scatter(pos[:, 0], pos[:, 1], s=30, c='white', edgecolors='cyan', zorder=3)
    
    ax.set_title("Manifold Optimization: Vectorized Langevin Dynamics", color='white', pad=20)
    ax.set_xlim(0, m.size); ax.set_ylim(0, m.size)
    ax.axis('off')

    def update(frame):
        nonlocal pos, vel
        
        # 1. Map positions to grid indices
        idx = pos.astype(int).clip(0, m.size - 1)
        
        # 2. Pull Metric and Gradient for all agents
        g_batch = m.g[idx[:, 0], idx[:, 1]]          # (N, 2, 2)
        grad_batch = m.grad_e[idx[:, 0], idx[:, 1]]  # (N, 2)
        
        # 3. Solve for Acceleration: g * accel = -grad
        # Adding a trailing dimension to grad_batch for the solver: (N, 2, 1)
        accel = np.linalg.solve(g_batch, -grad_batch[..., np.newaxis])
        accel = accel.squeeze(-1) # Back to (N, 2)

        # 4. Langevin Update (Velocity Verlet style)
        noise = cfg.noise_floor * np.random.randn(cfg.n_agents, 2)
        vel = 0.9 * vel + accel * cfg.dt + noise
        pos = np.clip(pos + vel, 0, m.size - 1)
        
        history[frame] = pos
        
        # 5. Visual Update
        dots.set_offsets(pos)
        for i in range(cfg.n_agents):
            lines[i].set_data(history[:frame+1, i, 0], history[:frame+1, i, 1])
        
        return lines + [dots]

    ani = FuncAnimation(fig, update, frames=cfg.total_steps, blit=True, interval=25)
    plt.show()

if __name__ == "__main__":
    execute()
