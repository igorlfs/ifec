"""Atividade 6: Autômatos Celulares."""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# |%%--%%| <UqvyFoZuFk|vb2ryy0ksp>

SAUDÁVEL = 0
INFECTADO = 1
RECUPERADO = 2

PROB_INFECTAR = 0.4
PROB_CURAR = 0.1

# |%%--%%| <vb2ryy0ksp|BhXrWgcenQ>


def simula_infectados(tamanho: int, passos: int, num_infectados: int):
    """Roda uma simulação com tamanho, passos e num_infectados."""
    vizinhos = [(0, 1), (1, 0), (-1, 0), (0, -1)]

    pessoas = np.zeros((tamanho, tamanho), dtype=np.int8)
    infectados = np.random.choice(tamanho * tamanho, num_infectados, replace=False)
    pessoas.ravel()[infectados] = 1

    cmap = mpl.colors.ListedColormap(["blue", "red", "green"])

    for _ in range(passos):
        novas_pessoas = pessoas.copy()
        plt.imshow(pessoas, cmap=cmap, vmin=0, vmax=2)
        plt.autoscale(False)
        plt.colorbar()
        plt.show()
        for i in range(tamanho):
            for j in range(tamanho):
                if pessoas[i][j] == INFECTADO and np.random.rand() < PROB_CURAR:
                    novas_pessoas[i][j] = RECUPERADO
                if pessoas[i][j] == SAUDÁVEL:
                    for k in vizinhos:
                        if (
                            i + k[0] < tamanho
                            and i + k[0] >= 0
                            and j + k[1] < tamanho
                            and j + k[1] >= 0
                            and pessoas[i + k[0]][j + k[1]] == INFECTADO
                            and np.random.rand() < PROB_INFECTAR
                        ):
                            novas_pessoas[i][j] = 1
        pessoas = novas_pessoas


# |%%--%%| <BhXrWgcenQ|Ovc5vhwOpB>

simula_infectados(10, 30, 5)

# |%%--%%| <Ovc5vhwOpB|qX63SYpvGh>


import plotly.graph_objects as go


def simula_infectados_plotly(tamanho: int, passos: int, num_infectados: int):
    vizinhos = [(0, 1), (1, 0), (-1, 0), (0, -1)]

    pessoas = np.zeros((tamanho, tamanho), dtype=np.int8)
    infectados = np.random.choice(tamanho * tamanho, num_infectados, replace=False)
    pessoas.ravel()[infectados] = 1

    frames = []

    for _ in range(passos):
        novas_pessoas = pessoas.copy()

        frames.append(
            go.Frame(data=go.Heatmap(z=pessoas, colorscale=["blue", "red", "green"]))
        )

        for i in range(tamanho):
            for j in range(tamanho):
                if pessoas[i][j] == 1 and np.random.rand() < PROB_CURAR:
                    novas_pessoas[i][j] = 2
                if pessoas[i][j] == 0:
                    for k in vizinhos:
                        if (
                            i + k[0] < tamanho
                            and i + k[0] >= 0
                            and j + k[1] < tamanho
                            and j + k[1] >= 0
                            and pessoas[i + k[0]][j + k[1]] == 1
                            and np.random.rand() < PROB_INFECTAR
                        ):
                            novas_pessoas[i][j] = 1
        pessoas = novas_pessoas

    fig = go.Figure(data=go.Heatmap(z=pessoas, zmin=0, zmax=2), frames=frames)
    fig.update_layout(title="Grid Evolution")
    fig.update_layout(
        updatemenus=[
            {
                "type": "buttons",
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "frame": {"duration": 500, "redraw": True},
                                "fromcurrent": True,
                                "transition": {"duration": 0},
                            },
                        ],
                    }
                ],
            }
        ]
    )
    fig.show()
    return fig


# |%%--%%| <qX63SYpvGh|Dc0zLbhZhe>

simula_infectados_plotly(10, 50, 5)

# |%%--%%| <Dc0zLbhZhe|OJHe2UGsBF>
