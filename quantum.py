import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pickle
from itertools import permutations
import tkinter as tk
from tkinter import ttk, filedialog
import threading
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.animation import FuncAnimation


# Enable interactive mode for Matplotlib
#plt.ion()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hbar = 1.0
m = 1.0

class Potential:
    def __call__(self, positions: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

class HarmonicOscillatorPotential(Potential):
    def __init__(self, omega: float = 1.0):
        self.omega = omega

    def __call__(self, positions: torch.Tensor) -> torch.Tensor:
        return 0.5 * self.omega**2 * torch.sum(positions**2, dim=[1,2])

class CoulombPotential(Potential):
    def __call__(self, positions: torch.Tensor) -> torch.Tensor:
        r = torch.norm(positions, dim=-1)
        return torch.sum(-1.0 / (r + 1e-6), dim=1)

class SquareWellPotential(Potential):
    def __call__(self, positions: torch.Tensor) -> torch.Tensor:
        r = torch.norm(positions, dim=-1)
        vals = torch.where(r < 2.0, torch.tensor(-5.0, device=device), torch.tensor(0.0, device=device))
        return torch.sum(vals, dim=1)

class NeuralNetWavefunction(nn.Module):
    def __init__(self, input_dim: int, layers: list[int] = [64,64]):
        super().__init__()
        net = []
        last = input_dim
        for width in layers:
            net.append(nn.Linear(last, width))
            net.append(nn.Tanh())
            last = width
        net.append(nn.Linear(last, 1))
        self.model = nn.Sequential(*net)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x).squeeze(-1)

class TrialWavefunction:
    def __init__(
        self,
        degree: int | None = None,
        symmetrize: bool = False,
        antisymmetrize: bool = False,
        use_nn: bool = False,
        n_particles: int = 2,
        nn_layers: list[int] = [64,64]
    ):
        self.symmetrize = symmetrize
        self.antisymmetrize = antisymmetrize
        self.use_nn = use_nn
        self.n_particles = n_particles
        if use_nn:
            self.net = NeuralNetWavefunction(input_dim=n_particles*3, layers=nn_layers).to(device)
        else:
            assert degree is not None
            self.coeffs = nn.Parameter(torch.randn(degree+1, device=device))

    def polynomial(self, positions: torch.Tensor) -> torch.Tensor:
        r = torch.norm(positions, dim=-1)
        r_tot = torch.sum(r, dim=1)
        poly = sum(self.coeffs[i] * r_tot**i for i in range(len(self.coeffs)))
        return poly * torch.exp(-r_tot**2 / 2)

    def evaluate(self, positions: torch.Tensor) -> torch.Tensor:
        batch = positions.shape[0]
        if self.use_nn:
            flat = positions.view(batch, -1)
            return self.net(flat)

        if self.symmetrize or self.antisymmetrize:
            perms = torch.stack([positions[:, perm, :] for perm in permutations(range(self.n_particles))], dim=1)
            vals = self.polynomial(perms.reshape(-1, self.n_particles, 3))
            vals = vals.view(batch, -1)
            if self.symmetrize:
                return vals.sum(dim=1) / torch.sqrt(torch.tensor(vals.shape[1], device=device, dtype=vals.dtype))
            else:
                signs = torch.tensor([(-1)**i for i in range(vals.shape[1])], device=device, dtype=vals.dtype)
                return (vals * signs).sum(dim=1) / torch.sqrt(torch.tensor(vals.shape[1], device=device, dtype=vals.dtype))
        else:
            return self.polynomial(positions)

    def probability_density(self, positions: torch.Tensor) -> torch.Tensor:
        psi = self.evaluate(positions)
        return psi**2

    def parameters(self):
        return self.net.parameters() if self.use_nn else [self.coeffs]

    def save(self, path: str):
        with open(path, 'wb') as f:
            if self.use_nn:
                torch.save(self.net.state_dict(), f)
            else:
                pickle.dump({'coeffs': self.coeffs.detach().cpu().numpy()}, f)

    def load(self, path: str):
        with open(path, 'rb') as f:
            if self.use_nn:
                self.net.load_state_dict(torch.load(f, map_location=device))
            else:
                data = pickle.load(f)
                self.coeffs.data.copy_(torch.tensor(data['coeffs'], device=device))

class VariationalSolver:
    def __init__(self, potential: Potential, wavefunction: TrialWavefunction, n_particles: int):
        self.V = potential
        self.psi = wavefunction
        self.n_particles = n_particles

    def hamiltonian_expectation(self, samples: int = 500) -> torch.Tensor:
        pos = torch.randn((samples, self.n_particles, 3), device=device, requires_grad=True)
        psi = self.psi.evaluate(pos)
        grad = torch.autograd.grad(psi.sum(), pos, create_graph=True)[0]
        kinetic = - (hbar**2 / (2*m)) * torch.sum(grad**2, dim=[1,2])
        potential = self.V(pos)
        numer = torch.mean((kinetic + potential) * psi**2)
        denom = torch.mean(psi**2)
        return numer / denom

    def solve(self, lr: float = 0.01, epochs: int = 200) -> float:
        opt = optim.Adam(self.psi.parameters(), lr=lr)
        energy = torch.tensor(0.0, device=device)
        for epoch in range(1, epochs+1):
            opt.zero_grad()
            energy = self.hamiltonian_expectation()
            energy.backward()
            opt.step()
            print(f"Epoch {epoch}/{epochs} - Energy = {energy.item():.6f}")
        return energy.item()

class Visualizer3D:
    def __init__(self, wavefunction, n_particles,
                 xrange=(-2,2), yrange=(-2,2), zrange=(-2,2),
                 threshold=0.05, resolution=20, energy=0.0):
        self.psi = wavefunction
        self.n_particles = n_particles
        self.xrange, self.yrange, self.zrange = xrange, yrange, zrange
        self.threshold = threshold
        self.resolution = resolution
        self.energy = energy
        self.t = 0.0
        self.dt = 0.01

        self.cached_grid = None
        self.cached_psi0  = None

        # precompute grid & frozen psi₀
        xs = torch.linspace(*xrange, resolution)
        ys = torch.linspace(*yrange, resolution)
        zs = torch.linspace(*zrange, resolution)
        grid_pts = torch.stack(torch.meshgrid(xs, ys, zs, indexing='ij'), dim=-1)
        grid = grid_pts.reshape(-1,1,3).repeat(1, n_particles, 1).to(device)
        psi0 = self.psi.evaluate(grid).detach()
        pd = (psi0**2).view(-1)
        mask = pd > (threshold * pd.max().item())
        self.pts = grid[mask,0].cpu().numpy()     # shape (M,3)
        self.psi0 = psi0.cpu().numpy()[mask]     # shape (M,)

        # set up figure & scatter once
        self.fig = Figure(figsize=(5,4))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_xlim(*xrange)
        self.ax.set_ylim(*yrange)
        self.ax.set_zlim(*zrange)
        self.scatter = self.ax.scatter(
            self.pts[:,0], self.pts[:,1], self.pts[:,2],
            c=self.psi0.real, cmap='plasma', s=10
        )
        self.colorbar = self.fig.colorbar(self.scatter, ax=self.ax, shrink=0.6)
    
    def update(self, _frame):
        self.t += self.dt
        phase = np.exp(-1j * self.energy * self.t / hbar)
        vals = (self.psi0 * phase).real

        # Update scatter colors & title
        self.scatter.set_array(vals)
        self.ax.set_title(f"t = {self.t:.3f}")

        # *** Tell the TkAgg canvas to repaint ***
        self.canvas.draw_idle()

        return (self.scatter,)

    def plot(self):
        with torch.no_grad():
            # 1) Clear the entire figure (axes + colorbars + text)
            self.fig.clf()

            # 2) Re-create the 3D axes
            self.ax = self.fig.add_subplot(111, projection='3d')

            # 3) Re-set fixed limits and labels
            self.ax.set_xlim(*self.xrange)
            self.ax.set_ylim(*self.yrange)
            self.ax.set_zlim(*self.zrange)
            self.ax.set_title(f"t = {self.t:.3f}")
            self.ax.set_xlabel("x")
            self.ax.set_ylabel("y")
            self.ax.set_zlabel("z")

            # 4) Build/carry-over your cached grid & psi₀ exactly as before
            if self.cached_grid is None:
                xs = torch.linspace(*self.xrange, self.resolution)
                ys = torch.linspace(*self.yrange, self.resolution)
                zs = torch.linspace(*self.zrange, self.resolution)
                grid_pts = torch.stack(
                    torch.meshgrid(xs, ys, zs, indexing='ij'), dim=-1
                )
                self.cached_grid = grid_pts.reshape(-1, 1, 3).repeat(1, self.n_particles, 1).to(device)

            if self.cached_psi0 is None:
                self.cached_psi0 = self.psi.evaluate(self.cached_grid).detach()

            # 5) Time-evolve
            psi_t = self.cached_psi0 * torch.exp(
                -1j * torch.tensor(self.energy * self.t / hbar, dtype=torch.cfloat, device=device)
            )

            # 6) Compute density mask & points
            pd = (self.cached_psi0**2).view(-1)
            max_pd = pd.max().item()
            mask = pd > (self.threshold * max_pd)
            pts = self.cached_grid[mask, 0].cpu().numpy()
            vals = psi_t.real.cpu().numpy()[mask]

            # 7) Scatter & add colorbar
            if pts.size:
                scatter = self.ax.scatter(
                    pts[:, 0], pts[:, 1], pts[:, 2],
                    c=vals, cmap='plasma', s=10
                )
                self.colorbar = self.fig.colorbar(
                    scatter, ax=self.ax, shrink=0.6, label="Re ψ(t)"
                )

            # 8) Draw to the Tk canvas
            self.canvas.draw()
            self.canvas.flush_events()

class GUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Variational Quantum Solver")
        frame = ttk.Frame(self.root, padding=10)
        frame.pack()

        ttk.Label(frame, text="Number of Particles:").grid(row=0, column=0, sticky='w')
        self.n_particles_var = tk.IntVar(value=2)
        ttk.Entry(frame, textvariable=self.n_particles_var, width=10).grid(row=0, column=1)

        ttk.Label(frame, text="Epochs:").grid(row=1, column=0, sticky='w')
        self.epochs_var = tk.IntVar(value=200)
        ttk.Entry(frame, textvariable=self.epochs_var, width=10).grid(row=1, column=1)

        ttk.Label(frame, text="Learning Rate:").grid(row=2, column=0, sticky='w')
        self.lr_var = tk.DoubleVar(value=0.01)
        ttk.Entry(frame, textvariable=self.lr_var, width=10).grid(row=2, column=1)

        ttk.Label(frame, text="x Range (min,max):").grid(row=3, column=0, sticky='w')
        self.xrange_var = tk.StringVar(value="-2,2")
        ttk.Entry(frame, textvariable=self.xrange_var, width=15).grid(row=3, column=1)
        ttk.Label(frame, text="y Range (min,max):").grid(row=4, column=0, sticky='w')
        self.yrange_var = tk.StringVar(value="-2,2")
        ttk.Entry(frame, textvariable=self.yrange_var, width=15).grid(row=4, column=1)
        ttk.Label(frame, text="z Range (min,max):").grid(row=5, column=0, sticky='w')
        self.zrange_var = tk.StringVar(value="-2,2")
        ttk.Entry(frame, textvariable=self.zrange_var, width=15).grid(row=5, column=1)

        ttk.Label(frame, text="Symmetry Type:").grid(row=6, column=0, sticky='w')
        self.symmetry = ttk.Combobox(frame, values=["None","Bosons","Fermions"], width=12)
        self.symmetry.current(0)
        self.symmetry.grid(row=6, column=1)

        ttk.Label(frame, text="Wavefunction Type:").grid(row=7, column=0, sticky='w')
        self.wf_type = ttk.Combobox(frame, values=["Polynomial","NeuralNet"], width=12)
        self.wf_type.current(0)
        self.wf_type.grid(row=7, column=1)
        self.wf_type.bind("<<ComboboxSelected>>", self.toggle_wf_config)

        self.poly_frame = ttk.Frame(frame)
        ttk.Label(self.poly_frame, text="Polynomial Degree:").grid(row=0, column=0)
        self.poly_degree_var = tk.IntVar(value=4)
        ttk.Entry(self.poly_frame, textvariable=self.poly_degree_var, width=10).grid(row=0, column=1)
        self.poly_frame.grid(row=8, column=0, columnspan=2)

        self.nn_frame = ttk.Frame(frame)
        ttk.Label(self.nn_frame, text="NN Layers (comma sep):").grid(row=0, column=0)
        self.nn_layers_var = tk.StringVar(value="64,64")
        ttk.Entry(self.nn_frame, textvariable=self.nn_layers_var, width=10).grid(row=0, column=1)

        ttk.Label(frame, text="Potential Type:").grid(row=9, column=0, sticky='w')
        self.potential_type = ttk.Combobox(frame, values=["Harmonic","Coulomb","SquareWell"], width=12)
        self.potential_type.current(0)
        self.potential_type.grid(row=9, column=1)

        ttk.Button(frame, text="Run Solver", command=self.run_solver_threaded).grid(row=10, column=0, pady=5)
        ttk.Button(frame, text="Save Wavefunction", command=self.save_wavefunction).grid(row=10, column=1)
        ttk.Button(frame, text="Load Wavefunction", command=self.load_wavefunction).grid(row=11, column=1)

        viz_frame = ttk.LabelFrame(frame, text="Visualization Controls", padding=10)
        viz_frame.grid(row=12, column=0, columnspan=2, pady=5, sticky='ew')
        ttk.Label(viz_frame, text="Threshold (%)").grid(row=0, column=0, sticky='w')
        self.threshold_var = tk.DoubleVar(value=5.0)
        ttk.Scale(viz_frame, from_=0, to=100, variable=self.threshold_var, orient='horizontal', command=self.update_visualization).grid(row=0, column=1, sticky='ew')
        ttk.Label(viz_frame, text="Resolution").grid(row=1, column=0, sticky='w')
        self.resolution_var = tk.IntVar(value=20)
        ttk.Scale(viz_frame, from_=5, to=50, variable=self.resolution_var, orient='horizontal', command=self.update_visualization).grid(row=1, column=1, sticky='ew')

        self.output = tk.Text(frame, height=5, width=40)
        self.output.grid(row=13, column=0, columnspan=2, pady=5)

        ttk.Label(frame, text="dt (s):").grid(row=14, column=0, sticky='w')
        self.dt_var = tk.DoubleVar(value=0.01)
        ttk.Entry(frame, textvariable=self.dt_var, width=6).grid(row=14, column=1, sticky='w')

        self.time_label = ttk.Label(frame, text="t = 0.000")
        self.time_label.grid(row=15, column=0, columnspan=2)

        btn_frame = ttk.Frame(frame)
        btn_frame.grid(row=16, column=0, columnspan=2, pady=5)
        ttk.Button(btn_frame, text="Start", command=self.start_time).grid(row=0, column=0)
        ttk.Button(btn_frame, text="Pause", command=self.pause_time).grid(row=0, column=1)
        ttk.Button(btn_frame, text="Reset", command=self.reset_time).grid(row=0, column=2)

        self.running = False
        self.viz = None
        self.toggle_wf_config()
        self.wavefunction = None
        self.solver = None
        self.root.mainloop()

    def toggle_wf_config(self, event=None):
        if self.wf_type.get() == "NeuralNet":
            self.poly_frame.grid_remove()
            self.nn_frame.grid(row=8, column=0, columnspan=2)
        else:
            self.nn_frame.grid_remove()
            self.poly_frame.grid(row=8, column=0, columnspan=2)

    def run_solver_threaded(self):
        threading.Thread(target=self.run_solver, daemon=True).start()

    def run_solver(self):
        n = self.n_particles_var.get()
        epochs = self.epochs_var.get()
        lr = self.lr_var.get()
        xr = tuple(map(float, self.xrange_var.get().split(',')))
        yr = tuple(map(float, self.yrange_var.get().split(',')))
        zr = tuple(map(float, self.zrange_var.get().split(',')))
        sym = self.symmetry.get()
        use_nn = self.wf_type.get() == "NeuralNet"
        layers = [int(x) for x in self.nn_layers_var.get().split(',')] if use_nn else []
        degree = self.poly_degree_var.get() if not use_nn else None
        pot_map = {"Harmonic": HarmonicOscillatorPotential, "Coulomb": CoulombPotential, "SquareWell": SquareWellPotential}

        wf = TrialWavefunction(degree=degree, symmetrize=(sym=="Bosons"), antisymmetrize=(sym=="Fermions"), use_nn=use_nn, n_particles=n, nn_layers=layers)
        solver = VariationalSolver(pot_map[self.potential_type.get()](), wf, n)

        energy = solver.solve(lr=lr, epochs=epochs)
        self.wavefunction = wf
        self.solver = solver
        self.energy = energy
        self.output.insert('end', f"Final energy: {energy:.6f}\n")
        self.output.see('end')

        self.viz = Visualizer3D(wf, n, xrange=xr, yrange=yr, zrange=zr, threshold=self.threshold_var.get()/100, resolution=self.resolution_var.get(), energy=energy)
        # Remove old canvas if it exists
        if hasattr(self, 'viz_canvas_widget'):
            self.viz_canvas_widget.destroy()

        if hasattr(self, 'viz_canvas_widget'):
            self.viz_canvas_widget.destroy()

        # Create one new Visualizer3D
        self.viz = Visualizer3D(
            wf, n,
            xrange=xr, yrange=yr, zrange=zr,
            threshold=self.threshold_var.get()/100,
            resolution=self.resolution_var.get(),
            energy=energy
        )

        # Hook up the new canvas
        canvas = FigureCanvasTkAgg(self.viz.fig, master=self.root)
        canvas.draw()
        self.viz_canvas_widget = canvas.get_tk_widget()
        self.viz_canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        self.viz.canvas = canvas
        self.viz.update(0)
        self.viz.canvas.draw()

        # Initialize time parameters and draw the first frame
        self.viz.t = 0.0
        self.viz.dt = self.dt_var.get()

    def update_visualization(self, event=None):
        if not self.viz:
            return
        self.viz.threshold = self.threshold_var.get() / 100.0
        self.viz.resolution = int(self.resolution_var.get())
        

    def start_time(self):
        if not self.viz:
            return

        # grab the latest dt
        self.viz.dt = self.dt_var.get()

        if not hasattr(self, 'anim'):
            # first time: create the FuncAnimation
            self.anim = FuncAnimation(
                self.viz.fig,
                self.viz.update,
                interval=self.viz.dt * 1000,
                blit=False,
                cache_frame_data=False
            )
        else:
            # restart an existing, paused animation
            self.anim.event_source.start()

        # make sure the canvas shows up
        self.viz.canvas.draw()

    def pause_time(self):
        if hasattr(self, 'anim'):
            self.anim.event_source.stop()

    def reset_time(self):
        if hasattr(self, 'anim'):
            # definitely stop, so Start can restart later
            self.anim.event_source.stop()

        # reset time counter
        self.viz.t = 0.0
        self.time_label.config(text=f"t = {self.viz.t:.3f}")

        # draw a single frame at t=0
        self.viz.update(0)
        self.viz.canvas.draw()

    def _tick(self):
        if not self.running:
            return

        # advance time
        self.viz.t += self.viz.dt
        self.time_label.config(text=f"t = {self.viz.t:.3f}")

        # redraw the figure
        

        # Force Tk to process any pending redraws right now
        self.root.update_idletasks()
        self.root.update()

        # schedule next
        self.root.after(int(self.viz.dt * 1000), self._tick)

    def save_wavefunction(self):
        if not self.wavefunction:
            return
        path = filedialog.asksaveasfilename(defaultextension='.pkl')
        if path:
            self.wavefunction.save(path)
            self.output.insert('end', f"Saved wavefunction to {path}\n")

    def load_wavefunction(self):
        path = filedialog.askopenfilename(filetypes=[('Pickle','*.pkl')])
        if not path:
            return
        sym = self.symmetry.get()
        use_nn = self.wf_type.get() == "NeuralNet"
        layers = [int(x) for x in self.nn_layers_var.get().split(',')] if use_nn else []
        degree = self.poly_degree_var.get() if not use_nn else None
        wf = TrialWavefunction(degree=degree, symmetrize=(sym=="Bosons"), antisymmetrize=(sym=="Fermions"), use_nn=use_nn, n_particles=self.n_particles_var.get(), nn_layers=layers)
        wf.load(path)
        self.wavefunction = wf
        self.output.insert('end', f"Loaded wavefunction from {path}\n")
        self.viz = Visualizer3D(wf, self.n_particles_var.get(), threshold=self.threshold_var.get()/100, resolution=self.resolution_var.get())
        

if __name__ == '__main__':
    GUI()
