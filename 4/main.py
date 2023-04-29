import numpy as np
from numba import jit
import matplotlib.pyplot as plt

# |%%--%%| <YaFo1LNmeW|mIOa4Sirek>


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


# |%%--%%| <mIOa4Sirek|5ig2ZWHJXT>


@jit(nopython=True)
def algoritmo_de_metropolis(L: int, T: float, passos: int):
    energia: np.ndarray = np.zeros(passos, dtype=np.int32)
    magnetização: np.ndarray = np.zeros(passos, dtype=np.int32)

    spins: np.ndarray = np.array([-1, 1], dtype=np.int8)

    variações_de_energia = np.array([8.0, 4.0, 0.0, -4.0, -8.0], dtype=np.float64)
    expoentes = np.exp(variações_de_energia * 1 / T)

    N = L * L
    S = np.random.choice(spins, N)

    viz = vizinhos(N)

    for i in range(passos):
        for k in np.arange(N):
            índice = int(S[k] * np.sum(S[viz[k]]) * 0.5 + 2)
            if np.random.rand() < expoentes[índice]:
                S[k] = -1 * S[k]
        energia[i] = -np.sum(S * (S[viz[:, 0]] + S[viz[:, 1]]))
        magnetização[i] = np.sum(S)

    return energia, magnetização


# |%%--%%| <5ig2ZWHJXT|rOA70HzMYL>

N = 0
comprimento = 100
temperatura = 3
PASSOS_DE_MONTECARLO = 1000
energias = np.zeros((N, PASSOS_DE_MONTECARLO))
magnetizações = np.zeros((N, PASSOS_DE_MONTECARLO))

for i in range(N):
    energias[i], magnetizações[i] = algoritmo_de_metropolis(
        comprimento, temperatura, PASSOS_DE_MONTECARLO
    )

for e in energias:
    plt.title(f"Rede: {comprimento} Temperatura: {temperatura}")
    plt.xlabel("Número de passos de Monte Carlo")
    plt.ylabel("Energia")
    plt.plot(e)

# plt.savefig(f"energias{comprimento}T{temperatura}.png")
plt.show()

for m in magnetizações:
    plt.title(f"Rede: {comprimento} Temperatura: {temperatura}")
    plt.xlabel("Número de passos de Monte Carlo")
    plt.ylabel("Magnetização")
    plt.plot(m)

# plt.savefig(f"magnetos{comprimento}T{temperatura}.png")
plt.show()

# |%%--%%| <rOA70HzMYL|Ihu3hlgRdY>
r"""°°°
# Análise

## Geração das Instâncias

Foram fixadas 10 repetições em todos os experimentos gerados (N=10). Inicialmente, o comprimento da rede foi fixado, C=32 e variou-se a temperatura: 0.4, 1.2, 2.1, 3. Depois, a temperatura foi fixada, T=1.7 e o comprimento da rede foi modificado: 24, 48, 72, 100. Também foi gerado um caso "mínimo", que combina o mínimo de comprimento (C=24) e temperatura (T=0.4) propostos, assim como um caso "máximo", que combina o comprimento máximo da rede (C=100) e temperatura (T=3).

## Energia

### Variação de Temperatura

![E32T0.4](img/energias32T0.4.png)  
![E32T1.2](img/energias32T1.2.png)  
![E32T2](img/energias32T2.png)  
![E32T3](img/energias32T3.png)  
![E32T3x](img/energias32T3++.png)  

O aumento de temperatura torna a convergência muito mais lenta. Quando a temperatura é muito baixa, é possível que a energia converga para um valor mais baixo. Tentou-se fazer mais passos de Monte Carlo para averiguar se a convergência estava próxima, mas não houve diferença em relação à divergência entre 1000 e 2000 passos.

### Variação de Comprimento

![E24T1.7](img/energias24T1.7.png)  
![E48T1.7](img/energias48T1.7.png)  
![E72T1.7](img/energias72T1.7.png)  
![E100T1.7](img/energias100T1.7.png)  

O tamanho da rede não possui tanta influência como a temperatura, na convergência. E energia total do sistema diminui conforme se aumenta o tamanho, o que faz parecer que a variação entre os picos e vales próximos da convergência é menor (eu diria que parece ser aproximadamente a mesma, mas não calculei isso). Não é possível afirmar se é mais comum que a convergência não ocorra com o aumento do tamanho da rede, com base nas imagens. Apesar de ser possível existir um viés.


### Mínimo e Máximo

![E24T0.4](img/energias24T0.4.png)  
![E100T3](img/energias100T3.png)  

Analisando os casos extremos, e com base nas análises anteriores, é possível supor que o aumento da temperatura é o principal fator que contribui para a não convergência da energia.

## Magnetização

### Variação de Temperatura

![M32T0.4](img/magnetos32T0.4.png)  
![M32T1.2](img/magnetos32T1.2.png)  
![M32T2](img/magnetos32T2.png)  
![M32T3](img/magnetos32T3.png)  
![M32T3x](img/magnetos32T3++.png)  

Para a temperatura mais baixa, 3 instâncias não convergiram, ficaram repetindo em padrões. Com o aumento da temperatura (T=2), houve uma tendência maior a convergir, mas com um aumento ainda maior (T=3), o gráfico voltou a ficar uma completa bagunça. Tal como para a energia, um aumento do número de passos de Monte Carlo não ajudou.

### Variação de Comprimento

![M24T1.7](img/magnetos24T1.7.png)  
![M48T1.7](img/magnetos48T1.7.png)  
![M72T1.7](img/magnetos72T1.7.png)  
![M100T1.7](img/magnetos100T1.7.png)  

Como no caso da energia, o aumento da rede aumentou os valores para os quais a magnetização converge e, também similarmente ao caso energético, não é possível afirmar que um aumento do tamanho da rede dificulta a convergência.

### Mínimo e Máximo

![M24T0.4](img/magnetos24T0.4.png)  
![M100T3](img/magnetos100T3.png)  

Novamente, o caso "exagerado" ficou não inteligível. Não houve convergência em nenhuma instância. Já o caso "básico" teve algumas instâncias que também não convergiram, mas no geral é bem comportado, convergindo rapidamente.
°°°"""
