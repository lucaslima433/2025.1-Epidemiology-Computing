import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap
from dataclasses import dataclass
from typing import Tuple, List

SUSC = 0
INF  = 1
REC  = 2

cmap = ListedColormap(["#2ecc71", "#e74c3c", "#000000"])

@dataclass
class Params:
    size: int = 100
    beta: float = 0.35
    gamma: float = 0.1
    p_seed: float = 0.002
    neighborhood: str = "moore"
    periodic: bool = True
    steps: int = 200
    record_every: int = 1


def init_grid(p: Params) -> np.ndarray:
    grid = np.zeros((p.size, p.size), dtype=np.uint8)
    seeds = np.random.rand(p.size, p.size) < p.p_seed
    grid[seeds] = INF
    return grid


def neighbor_indices(i: int, j: int, p: Params) -> List[Tuple[int, int]]:
    if p.neighborhood == "vonneumann":
        neigh = [(-1,0),(1,0),(0,-1),(0,1)]
    else:
        neigh = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]

    idxs = []
    n = p.size
    for di, dj in neigh:
        ni, nj = i + di, j + dj
        if p.periodic:
            ni %= n
            nj %= n
            idxs.append((ni, nj))
        else:
            if 0 <= ni < n and 0 <= nj < n:
                idxs.append((ni, nj))
    return idxs


def step(grid: np.ndarray, p: Params) -> Tuple[np.ndarray, int, int]:
    n = p.size
    new_grid = grid.copy()

    infected = (grid == INF)

    recov_mask = infected & (np.random.rand(n, n) < p.gamma)
    new_grid[recov_mask] = REC
    new_recoveries = int(recov_mask.sum())

    susc = (grid == SUSC)
    will_infect = np.zeros((n, n), dtype=bool)

    inf_pos = np.argwhere(infected)
    for i, j in inf_pos:
        for ni, nj in neighbor_indices(i, j, p):
            if grid[ni, nj] == SUSC:
                if np.random.rand() < p.beta:
                    will_infect[ni, nj] = True

    new_grid[will_infect] = INF
    new_infections = int(will_infect.sum())

    return new_grid, new_infections, new_recoveries

def box_counting_dimension(mask: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    n = mask.shape[0]
    sizes = []
    s = n
    while s >= 2:
        if n % s == 0:
            sizes.append(s)
        s //= 2
    if 1 not in sizes:
        sizes.append(1)

    counts = []
    for s in sizes:
        n_blocks = n // s
        view = mask.reshape(n_blocks, s, n_blocks, s).swapaxes(1,2)
        occupied = view.max(axis=(2,3))
        counts.append(occupied.sum())

    eps = np.array([s / n for s in sizes], dtype=float)
    log_inv_eps = np.log(1.0 / eps)
    log_N = np.log(np.array(counts, dtype=float) + 1e-12)

    A = np.vstack([log_inv_eps, np.ones_like(log_inv_eps)]).T
    slope, intercept = np.linalg.lstsq(A, log_N, rcond=None)[0]
    return float(slope), log_inv_eps, log_N

def run_simulation(p: Params, animate: bool = True):
    grid = init_grid(p)

    dims = []
    infections = []
    recoveries = []

    fig, ax = plt.subplots()

    def draw(g, title_extra=""):
        ax.clear()
        ax.imshow(g, cmap=cmap, vmin=0, vmax=2)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"SIR CA — S: verde, I: vermelho, R: preto {title_extra}")

    if animate:
        def update(_):
            nonlocal grid
            grid, new_inf, new_rec = step(grid, p)

            mask = (grid == INF)
            if np.any(mask):
                D, _, _ = box_counting_dimension(mask)
            else:
                D = 0.0
            dims.append(D)
            infections.append(new_inf)
            recoveries.append(new_rec)

            draw(grid, title_extra=f" | D≈{D:.2f}")

        ani = animation.FuncAnimation(fig, update, frames=p.steps, interval=80)
        plt.show()
        
    else:
        for t in range(p.steps):
            grid, new_inf, new_rec = step(grid, p)
            mask = (grid == INF)
            D = box_counting_dimension(mask)[0] if np.any(mask) else 0.0
            dims.append(D)
            infections.append(new_inf)
            recoveries.append(new_rec)
        draw(grid, title_extra=f" | D≈{dims[-1]:.2f}")
        plt.show()

        plt.figure()
        plt.plot(dims)
        plt.xlabel("Passo")
        plt.ylabel("Dimensão (box-counting) do cluster infectado")
        plt.title("Evolução da dimensão de similaridade")
        plt.show()

    return dims, infections, recoveries


if __name__ == "__main__":
    p = Params(
        size=128,
        beta=0.28,
        gamma=0.12,
        p_seed=0.003,
        neighborhood="moore",
        periodic=True,
        steps=400,
        record_every=1,
    )
    run_simulation(p, animate=True)