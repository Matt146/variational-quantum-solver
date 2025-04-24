---
# Variational Quantum Solver with Real-Time 3D Visualization

A fully interactive variational quantum solver and simulator built in Python — powered by PyTorch and featuring real-time 3D visualization via `matplotlib` + `tkinter`.

This project is an educational and experimental sandbox for exploring quantum mechanics using both analytical (polynomial) and neural network trial wavefunctions. It supports arbitrary particle count, potential selection, and symmetry handling (bosonic or fermionic), and it dynamically evolves and renders the probability density over time in 3D.

> ⚠️ **Note:** This project is actively under development. Many core features are incomplete and there are known bugs and optimizations I'm still working through. Feedback and contributions are welcome!
---

## 🚀 Features

### 🔬 Physics Engine

- Implements the **variational method** to approximate ground state energies.
- Supports **multiple quantum potentials**:
  - Harmonic oscillator
  - Coulomb attraction
  - Finite square well
- Supports **arbitrary particle count** (3D positions per particle).
- Handles **bosonic (symmetric)** and **fermionic (antisymmetric)** wavefunctions via automatic permutation.

### 🧠 Trial Wavefunctions

- **Polynomial wavefunctions** with tunable degree.
- **Neural network wavefunctions** built with customizable layer depth and activation.
- Auto-differentiation via PyTorch for gradient-based optimization.

### ⚙️ Optimization

- Uses `Adam` optimizer to minimize the variational energy estimate.
- Computes ⟨ψ|H|ψ⟩ via Monte Carlo integration using normally distributed samples.

### 📊 Visualization

- **Real-time 3D probability density visualization** in `matplotlib`, with:
  - Time evolution of Re(ψ) via complex phase rotation.
  - Adjustable resolution and probability threshold for rendering.
  - Color-coded scatter plots with a live-updating 3D canvas.

### 🖥️ Full GUI

- Built with `tkinter`, the GUI includes:
  - Configurable input fields for particle count, epochs, potentials, learning rate, etc.
  - Wavefunction type selection (Polynomial / NeuralNet).
  - Time controls (`Play`, `Pause`, `Reset`) for dynamic visualization.
  - Save/Load functionality for wavefunction states.
  - Embedded matplotlib canvas with animation controls.
  - Real-time updates of energy and time.

---

## 🛠️ Installation

```bash
git clone https://github.com/yourusername/variational-quantum-solver.git
cd variational-quantum-solver
pip install -r requirements.txt
python quantum_solver.py
```

Make sure you have:

- Python 3.9+
- PyTorch with CUDA (optional but recommended)
- `matplotlib`, `tkinter`, `numpy`, etc.

---

## 🧪 Example Use Case

1. Choose **2 particles**.
2. Set potential to **Coulomb**.
3. Select **NeuralNet** wavefunction with `[64, 64]` layers.
4. Hit **Run Solver**.
5. Watch the probability density evolve in real time in 3D.
6. Adjust **resolution** or **threshold** mid-visualization to explore details.

---

## 📦 TODO / Known Issues

- ❗ GUI layout glitches on some screen sizes.
- ❗ Occasionally unstable animation performance when resolution is high.
- ❗ Energy convergence stalls for poorly initialized neural networks.
- ✅ Improve NN weight initialization for better training.
- ✅ Add support for custom potentials (user-defined).
- 🔄 Refactor `Visualizer3D` to reduce redraw overhead.

---

## 🤝 Contributing

Pull requests are welcome! Please file an issue if you discover any bugs or want to propose a feature. I'm particularly interested in:

- Optimizing rendering and solver performance.
- Supporting more realistic many-body potentials.
- Replacing `tkinter` with a more modern GUI framework.

---

## 📜 License

MIT License. Free to use, study, and modify.

---

## 📷 Screenshots

![screenshot](https://github.com/user-attachments/assets/8514e8c6-d849-446e-b667-866e1ff15f9d)

Let me know if you want me to generate a badge-style GitHub header or add GIF demos!
