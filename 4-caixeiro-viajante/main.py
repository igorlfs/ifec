"""Atividade 4: Caixeiro Viajante."""
import time
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
from numba import jit
from numpy.typing import NDArray

# |%%--%%| <MrHJR5Y2F4|E2C7DjUfE8>

start_time = time.time()

# |%%--%%| <E2C7DjUfE8|5bIZbFwebX>

# Define as posições aleatórias das cidades
rng = np.random.default_rng(seed=42)

# |%%--%%| <5bIZbFwebX|hpWZ5w2W65>


@jit(nopython=True, parallel=True)
def calculate_distance(
    num: np.int64, x: NDArray[np.float64], y: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Define a distancia entre duas cidades quaisquer."""
    distance = np.zeros((num, num), dtype=np.float64)
    for i in range(num):
        for j in range(num):
            distance[i, j] = np.sqrt(
                np.power((x[i] - x[j]), 2) + np.power((y[i] - y[j]), 2)
            )
    return distance


# |%%--%%| <hpWZ5w2W65|Vv5OsnyhvZ>


@jit(nopython=True, parallel=True)
def calculate_cost(
    num: np.int64, path: NDArray[np.int64], dist: NDArray[np.float64]
) -> float:
    """Calcula a distancia total percorrida pela caminhada."""
    cost = np.zeros(1, dtype=np.float64)
    for i in range(num):
        cost += dist[path[i % num], path[(i + 1) % num]]
    return cost[0]


# |%%--%%| <Vv5OsnyhvZ|CR1bHWdpUs>


@jit(nopython=True, parallel=True)
def calculate_new_path(
    num: np.int64, path: NDArray[np.int64]
) -> tuple[NDArray[np.int64], int, int]:
    """Calcula o novo caminho, trocando dois índices."""
    # define uma nova caminhada
    newpath = np.zeros(num, dtype=np.int64)

    indexes: NDArray[np.int64] = np.random.choice(np.arange(num), 2, replace=False)
    # o índice incial deve ser o menor, então ordenamos
    indexes.sort()
    ini: int = indexes[0]
    fin: int = indexes[1]

    # inverte o sentido em que percorre o caminho entre os indices escolhidos
    for k in range(num):
        if k >= ini and k <= fin:
            newpath[k] = path[fin - k + ini]
        else:
            newpath[k] = path[k]

    return newpath, ini, fin


# |%%--%%| <CR1bHWdpUs|rSMST6bRwb>


@jit(nopython=True, parallel=True)
def mc_step(
    num: np.int64,
    beta: np.float64,
    energy,
    path: NDArray[np.int64],
    best_e: float,
    best_p: NDArray[np.int64],
    dist: NDArray[np.float64],
):
    """Rode um passo de Monte Carlo."""
    # realiza um passo de Monte Carlo
    new_path = np.zeros(num, dtype=np.int64)

    new_path, ini, fin = calculate_new_path(num, path)  # propoe um novo caminho

    # determina a diferença de energia
    left = ini - 1  # cidade anterior a inicial
    if left < 0:
        left = num - 1  # condicao de contorno
    right = fin + 1  # cidade apos a final
    if right > num - 1:
        right = 0  # condicao de contorno
    delta = (
        -dist[path[left], path[ini]]
        - dist[path[right], path[fin]]
        + dist[new_path[left], new_path[ini]]
        + dist[new_path[right], new_path[fin]]
    )

    if delta < 0:  # aplica o criterio de Metropolis
        energy += delta
        path = new_path
        if energy < best_e:  # guarda o melhor caminho gerado até o momento
            best_e = energy
            best_p = path
    # aplica o criterio de Metropolis
    elif np.random.random() < np.exp(-beta * delta):
        energy += delta
        path = new_path

    return energy, path, best_e, best_p


# |%%--%%| <rSMST6bRwb|M8kUzJF4dp>


def generate_graph(num: np.int64) -> tuple[NDArray, NDArray, NDArray]:
    """Gere um novo grafo com `num` arestas."""
    x = rng.random(num)
    y = rng.random(num)

    # define o caminho que liga as cidades (inicialmente a sequencia como foi criada)
    path = np.zeros(num, dtype=np.int64)
    for i in range(num):
        path[i] = i

    return x, y, path


# |%%--%%| <M8kUzJF4dp|PmDkupE1CQ>


def plot_path(x: NDArray, y: NDArray, path: NDArray, cost: float, num_cities: int):
    """Imprima o caminho encontrado."""
    plt.title(f"Comprimento: {cost}")
    for i in range(num_cities):
        plt.plot(
            [x[path[i]], x[path[(i + 1) % num_cities]]],
            [y[path[i]], y[path[(i + 1) % num_cities]]],
            color="blue",
        )
    plt.show()


# |%%--%%| <PmDkupE1CQ|EMBIcvI8Kk>


def plot_metrics(temperatures: NDArray, path_cost: NDArray, num_cities: int):
    """Imprima a evolução das temperaturas e do custo do caminho."""
    _, ax = plt.subplots()
    ax = cast(plt.Axes, ax)
    plt.title(f"N: {num_cities} $T_0$: {T_INIT} base: {MULTIPLIER}")

    ax2 = ax.twinx()
    ax.plot(path_cost, color="blue")
    ax2.plot(temperatures, color="red")

    ax.set_xlabel("Passos de Monte Carlo")
    ax.set_ylabel("Custo", color="blue")
    ax2.set_ylabel("Temperaturas", color="red")


# |%%--%%| <EMBIcvI8Kk|6Y9DnasT0B>
r"""°°°
## Análise

O método de simulated annealing é razoável, basta fazer o tuning correto dos parâmetros, o que pode exigir bastante tempo. O tempo gasto é bom, para a aproximação que é entregue.
°°°"""
# |%%--%%| <6Y9DnasT0B|zl6jNTwcRe>

file = np.loadtxt("4-caixeiro-viajante/posições.dat")
x, y = file[:, 0], file[:, 1]
path = np.zeros(len(x), dtype=np.int64)
for i in range(len(x)):
    path[i] = i

# |%%--%%| <zl6jNTwcRe|wLMKZyeeLE>

NUM_OF_CITIES = len(x)
MC_STEPS = 30000
T_INIT = 2.5 * NUM_OF_CITIES
MULTIPLIER = 0.999999

# |%%--%%| <wLMKZyeeLE|vd5zMy4ZaN>


@jit(nopython=True, parallel=True)
def main(path: NDArray[np.int64]):
    """Função com o loop principal de otimização."""
    path_cost = np.zeros(MC_STEPS, dtype=np.float64)
    temperatures = np.zeros(MC_STEPS, dtype=np.float64)
    beta = np.float64(T_INIT)
    dist = calculate_distance(NUM_OF_CITIES, x, y)
    cost = calculate_cost(NUM_OF_CITIES, path, dist)
    best_e = cost
    best_p = path

    for i in range(MC_STEPS):
        cost, path, best_e, best_p = mc_step(
            NUM_OF_CITIES, beta, cost, path, best_e, best_p, dist
        )
        path_cost[i] = best_e
        temperatures[i] = beta
        beta = beta * MULTIPLIER
    return best_p, best_e, path_cost, temperatures


# |%%--%%| <vd5zMy4ZaN|CcDL0TMWst>

best_p, best_e, path_cost, temperatures = main(path)
plot_path(x, y, best_p, best_e, NUM_OF_CITIES)
plot_metrics(temperatures, path_cost, NUM_OF_CITIES)

# |%%--%%| <CcDL0TMWst|1lBY7RYBvy>

print("--- %s seconds ---" % (time.time() - start_time))

# |%%--%%| <1lBY7RYBvy|Ld0P5EEuQ1>
