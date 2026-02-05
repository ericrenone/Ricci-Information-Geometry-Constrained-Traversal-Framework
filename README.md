# Geometric path optimization on curved manifolds with information-theoretic navigation.

RIG-CTF simulates multi-agent traversal on discrete Riemannian manifolds under geometric and information-theoretic constraints. Agents navigate energy landscapes guided by Ricci curvature resistance and Fisher information metrics, using variational optimal control with stochastic Langevin dynamics.

**Core Features:**
- **Ollivier-Ricci curvature** approximation on 2D discrete manifolds
- **Fisher information metric** from parametric probability models (Gaussian mixtures)
- **Variational policies** via gradient descent on effective potentials
- **Stochastic dynamics** with noise-driven exploration
- **Multi-agent ensembles** for statistical validation
- **Real-time visualization** of geometry, trajectories, and costs
- **Full reproducibility** with automated validation tests

---

## Applications

- **AI/ML Path Planning:** Curvature-informed navigation in latent spaces
- **Reinforcement Learning:** Geometric reward shaping and exploration strategies
- **Information Geometry:** Empirical Fisher metric analysis
- **Computational Physics:** Discrete manifold simulations with geometric constraints
- **Optimization:** Gradient flows on curved parameter spaces

---

## Validation

Built-in automated tests verify:
- Probability distribution normalization
- Metric tensor positive-definiteness
- Fisher information non-negativity
- Numerical stability

