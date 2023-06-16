"""Atividade 7: Teoria dos Jogos."""
import matplotlib.pyplot as plt
import numpy as np
from numba import jit

# |%%--%%| <GDwRoXIWIR|svGBvqDvFD>


@jit(nopython=True)
def play(player: np.ndarray, b: float):
    """Rode uma partida de um jogador com seus vizinhos, com ganho `b`."""
    neighbors = [(1, 0), (0, 1), (-1, 0), (0, -1), (0, 0)]
    current = players[player[0]][player[1]]
    gains = 0
    for k in neighbors:
        other = players[(player[0] + k[0]) % N][(player[1] + k[1]) % N]
        if current == 1 and other == 1:
            gains += 1.0
        elif current == 0 and other == 1:
            gains += b
    return gains


# |%%--%%| <svGBvqDvFD|LOS3FXNhrm>


@jit(nopython=True)
def simulate(players: np.ndarray, passos: int, b: float):
    """Simula uma execução do dilema do prisioneiro."""
    # É necessário fazer a cópia para não interferir em execuções posteriores
    copy = players.copy()
    cooperadores = np.zeros(passos)
    for i in range(passos):
        idx = np.random.randint(0, N, size=2)

        neighbor = np.random.randint(0, 4)
        idx_neighbor = np.array([(idx[0] + neighbor) % N, (idx[1] + neighbor) % N])

        cooperadores[i] = np.count_nonzero(copy)

        gains = play(idx, b)
        gains_neighbor = play(idx_neighbor, b)

        w = 1 / (1 + np.exp((gains - gains_neighbor) / K))

        if np.random.rand() < w:
            copy[idx[0]][idx[1]] = copy[idx_neighbor[0]][idx_neighbor[1]]
    return cooperadores


# |%%--%%| <LOS3FXNhrm|DRzVrniwMg>

K = 0.5
N = 200
players = np.random.choice(a=[0, 1], size=(N, N))

# |%%--%%| <DRzVrniwMg|dBRhV89wgc>

for i in np.linspace(1, 2, 5):
    cooperadores = simulate(players, 4000000, i)
    plt.plot(cooperadores)

# |%%--%%| <dBRhV89wgc|p5bJk82J0d>
