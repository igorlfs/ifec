"""Atividade 6: Autômatos Celulares."""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# |%%--%%| <UqvyFoZuFk|vb2ryy0ksp>

SAUDÁVEL = 0
INFECTADO = 1
RECUPERADO = 2

PROB_INFECTAR = 0.5
PROB_CURAR = 0.1

# |%%--%%| <vb2ryy0ksp|BhXrWgcenQ>


def simula_infectados(tamanho: int, passos: int, num_infectados: int):
    """Roda uma simulação com tamanho, passos e num_infectados."""
    frames = []
    vizinhos = [(0, 1), (1, 0), (-1, 0), (0, -1)]

    pessoas = np.zeros((tamanho, tamanho), dtype=np.int8)
    infectados = np.random.choice(tamanho * tamanho, num_infectados, replace=False)
    pessoas.ravel()[infectados] = 1

    cmap = mpl.colors.ListedColormap(["blue", "red", "green"])

    for z in range(passos):
        novas_pessoas = pessoas.copy()
        plt.imshow(pessoas, cmap=cmap, vmin=0, vmax=2)
        plt.autoscale(False)
        plt.colorbar()
        plt.savefig(f"6-autômatos-celulares/img/frame_{z+1}.png")
        plt.clf()
        img = Image.open(f"6-autômatos-celulares/img/frame_{z+1}.png")
        frames.append(img)
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
    return frames


# |%%--%%| <BhXrWgcenQ|Ovc5vhwOpB>

frames = simula_infectados(100, 50, 7)
frames[0].save(
    "6-autômatos-celulares/animation.gif",
    format="GIF",
    append_images=frames[1:],
    save_all=True,
    duration=300,
    loop=0,
)

# |%%--%%| <Ovc5vhwOpB|OJHe2UGsBF>
