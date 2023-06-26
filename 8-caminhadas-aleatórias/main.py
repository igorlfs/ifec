"""Atividade 8: Caminhadas Aleatórias."""
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm

# |%%--%%| <rJkcb21zTy|i6ykH7ZEy8>


def random_walk(steps: int, dimensions: int):
    current_pos = np.zeros(shape=(steps, dimensions))
    for i in range(steps):
        direction = np.random.uniform(low=-0.5, high=0.5, size=dimensions)
        current_pos[i] = direction + current_pos[i - 1]
    return current_pos


# |%%--%%| <i6ykH7ZEy8|Kl5T2Evw6j>

positions = random_walk(10_000, 1)
plt.title("Distância em função do número de passos")
plt.plot(positions)
plt.axhline(y=0, linestyle="dashed")

# |%%--%%| <Kl5T2Evw6j|Lfo4ReWyGc>

positions = random_walk(10_000, 2)
plt.title("Dispersão dos passos no plano")
plt.plot(positions[:, 0], positions[:, 1])
plt.axhline(y=0, linestyle="dashed")
plt.axvline(x=0, linestyle="dashed")

# |%%--%%| <Lfo4ReWyGc|A4aJZkkrpJ>

for j in [100_000, 1_000, 10]:
    z = int(round(j, 0))
    positions = random_walk(z, 2)
    plt.plot(positions[:, 0], positions[:, 1], label=z)
plt.axhline(y=0, linestyle="dashed")
plt.axvline(x=0, linestyle="dashed")
plt.xlim([-150, 150])
plt.ylim([-150, 150])
plt.legend()
plt.title("Múltiplas dispersões para tamanhos diferentes")

# |%%--%%| <A4aJZkkrpJ|Yc0W3OpWeU>
r"""°°°
### B

Não, a distância final não aumenta por cerca de 10 vezes. Parece não existir um padrão mutio claro.
°°°"""
# |%%--%%| <Yc0W3OpWeU|Vg8JIjJ8aG>


def final_walk(w: int):
    for i in [10, 1]:
        last = np.zeros(shape=(w, 2))
        for j in range(w):
            last_step = random_walk(i, 2)
            last[j] = np.array([last_step[-1, 0], last_step[-1, 1]])
        plt.scatter(x=last[:, 0], y=last[:, 1], s=0.2)


# |%%--%%| <Vg8JIjJ8aG|nR0I7Ao5by>

final_walk(10_000)

# |%%--%%| <nR0I7Ao5by|q4FMJgAjks>


def estimate(steps: int, w: int):
    walks = np.zeros(shape=(w, 1))
    for j in range(w):
        walk = random_walk(steps, 1)
        walks[j] = walk[-1, 0]

    # No enunciado fala pra usar isso aqui
    # rms = np.sqrt(np.mean(walks**2))

    bins = 50
    plt.hist(walks[:, 0], bins=bins)

    # Faz a "escala" do gráfico
    bin_width = (walks.max() - walks.min()) / bins
    hist_area = len(walks[:, 0]) * bin_width

    mu, std = norm.fit(walks)

    x = np.linspace(mu - 3 * std, mu + 3 * std, 100)
    plt.plot(x, norm.pdf(x, mu, std) * hist_area, linestyle="dashed", color="black")


# |%%--%%| <q4FMJgAjks|EkM5txcruP>

N = 10_000
for i in [5, 3, 2, 1]:
    estimate(i, N)
    plt.title(f"Posição após {N} caminhadas")
# |%%--%%| <EkM5txcruP|otjnJYQS1U>
r"""°°°
### C

A Gaussiana se torna uma boa aproximação a partir de $N = 5$, mas deixa de ser "terrível" logo para $N = 2$.
°°°"""
