# pyright: reportUnusedExpression=false

# |%%--%%| <wB6g7BwnFN|qGq4OeeWD1>

import numpy as np
from numba import jit
import matplotlib.pyplot as plt

# |%%--%%| <qGq4OeeWD1|hcdGp3jU6f>

from numpy.typing import NDArray

# |%%--%%| <hcdGp3jU6f|19Z6oUBWhn>

import matplotlib.gridspec as gridspec

# |%%--%%| <19Z6oUBWhn|ql26NSQkHF>


@jit(nopython=True)
def vizinhos(N: int):
    # Define a tabela de vizinhos
    L = int(np.sqrt(N))
    viz = np.zeros((N, 4), dtype=np.int16)
    for k in range(N):
        viz[k, 0] = k + 1
        if (k + 1) % L == 0:
            viz[k, 0] = k + 1 - L
        viz[k, 1] = k + L
        if k > (N - L - 1):
            viz[k, 1] = k + L - N
        viz[k, 2] = k - 1
        if k % L == 0:
            viz[k, 2] = k + L - 1
        viz[k, 3] = k - L
        if k < L:
            viz[k, 3] = k + N - L
    return viz


# |%%--%%| <ql26NSQkHF|POJv20oWHb>


@jit(nopython=True)
def algoritmo_de_metropolis(L: int, T: float, passos: int):
    energia: np.ndarray = np.zeros(passos, dtype=np.int32)
    magnetização: np.ndarray = np.zeros(passos, dtype=np.int32)

    spins: np.ndarray = np.array([-1, 1], dtype=np.int8)

    variações_de_energia = np.array([8.0, 4.0, 0.0, -4.0, -8.0], dtype=np.float64)
    expoentes = np.exp(variações_de_energia / T)

    N = L * L
    S = np.random.choice(spins, N)

    viz = vizinhos(N)

    for i in range(passos):
        for k in np.arange(N):
            índice = int(S[k] * np.sum(S[viz[k]]) * 0.5 + 2)
            if np.random.rand() < expoentes[índice]:
                S[k] = -1 * S[k]
        energia[i] = -1 * np.sum(S * (S[viz[:, 0]] + S[viz[:, 1]]))
        magnetização[i] = np.sum(S)

    return energia, magnetização


# |%%--%%| <POJv20oWHb|ZNeMMSFNjy>


@jit(nopython=True)
def calor_específico(T: float, N: int, E: NDArray) -> NDArray:
    return (np.average(np.power(E, 2)) - np.power(np.average(E), 2)) / (
        N * np.power(T, 2)
    )


# |%%--%%| <ZNeMMSFNjy|S5wXJyZc6z>


@jit(nopython=True)
def suscetibilidade_magnética(T: float, N: int, M: NDArray) -> NDArray:
    return (np.average(np.power(M, 2)) - np.power(np.average(M), 2)) / (N * T)


# |%%--%%| <S5wXJyZc6z|gmVPUrJcBr>


@jit(nopython=True)
def calcula_erro(arr: NDArray) -> float:
    return np.sqrt(
        np.sum(np.power(arr - np.average(arr), 2)) / (arr.size * (arr.size - 1))
    )


# |%%--%%| <gmVPUrJcBr|el040SsWYu>


@jit(nopython=True)
def calcula_métricas(
    passo: int,
    seguro: int,
    passosDeMC: int,
    temperatura: float,
    numSpins: int,
    energias: NDArray[np.int32],
    magnetizações: NDArray[np.int32],
):
    magnetizações_mod = np.abs(magnetizações)
    energias_útil = energias[seguro:]
    magnetos_útil = magnetizações_mod[seguro:]
    batches = int((passosDeMC - seguro) / passo)
    matriz = np.zeros((batches, 4), dtype=float)

    for j in range(0, passosDeMC - seguro, passo):
        cal = calor_específico(temperatura, numSpins, energias_útil[j : j + passo])
        sus = suscetibilidade_magnética(
            temperatura, numSpins, magnetos_útil[j : j + passo]
        )
        ene = np.average(energias_útil[j : j + passo]) / numSpins
        mag = np.average(magnetos_útil[j : j + passo]) / numSpins
        matriz[int(j / passo)] = np.array([cal, sus, ene, mag])

    erros = np.zeros(4)
    for j in range(4):
        erros[j] = calcula_erro(matriz[:, j])

    novas = np.average(matriz, axis=0)

    return novas, erros


# |%%--%%| <el040SsWYu|bRqeuJnU2k>


@jit(nopython=True)
def variação_por_comprimento(comprimento: int):
    métricas = np.zeros((TEMPERATURAS.size, 4))
    erros = np.zeros((TEMPERATURAS.size, 4))
    NÚMERO_DE_SPINS = np.power(comprimento, 2)
    for i, t in enumerate(TEMPERATURAS):
        energias = np.zeros(PASSOS_DE_MONTECARLO)
        magnetizações = np.zeros(PASSOS_DE_MONTECARLO)
        energias, magnetizações = algoritmo_de_metropolis(
            comprimento, t, PASSOS_DE_MONTECARLO
        )
        métricas_e_erros = calcula_métricas(
            PARTIÇÃO,
            NÚMERO_DE_SEGURANÇA,
            PASSOS_DE_MONTECARLO,
            t,
            NÚMERO_DE_SPINS,
            energias,
            magnetizações,
        )
        métricas[i], erros[i] = métricas_e_erros
    return métricas, erros


# |%%--%%| <bRqeuJnU2k|SWmXA6pcz0>

# Número de passos em uma "batch"
PARTIÇÃO = 1000
# Variações de Comprimento
COMPRIMENTOS = np.linspace(32, 100, 5, dtype=int)
# Passos de Monte Carlo ignorados (termalização)
NÚMERO_DE_SEGURANÇA = 5000
# Total de Passos de Monte Carlo
PASSOS_DE_MONTECARLO = 15000
# Variações de Temperatura
TEMPERATURAS = np.linspace(0.4, 3, 20)

# |%%--%%| <SWmXA6pcz0|O451up2cZ4>


def plot_por_comprimento(métricas: NDArray, erros: NDArray, comprimento: int):
    titulos = [
        "Calor Específico",
        "Suscetibilidade Magnética",
        "Energias",
        "Magnetizações",
    ]
    fig = plt.figure(figsize=(10, 20))
    gs = gridspec.GridSpec(4, 1, figure=fig)
    for i in range(4):
        ax = fig.add_subplot(gs[i])
        ax.set_title(f"{titulos[i]} {comprimento}")
        ax.set_ylabel(titulos[i])
        ax.set_xlabel("Temperatura")
        ax.errorbar(
            x=TEMPERATURAS,
            y=métricas[:, i],
            yerr=erros[:, i],
            fmt="o",
            markersize=1,
            ecolor="red",
            color="blue",
        )

    fig.subplots_adjust(wspace=0.4, hspace=0.4)
    plt.savefig(f"{comprimento}-métricas.png")
    plt.show()


# |%%--%%| <O451up2cZ4|hONwWbcE6X>

for comprimento in COMPRIMENTOS:
    métricas, erros = variação_por_comprimento(comprimento)
    plot_por_comprimento(métricas, erros, comprimento)
