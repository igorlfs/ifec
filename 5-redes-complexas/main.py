"""Atividade 5: Redes Complexas."""

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import rustworkx as rx

# |%%--%%| <OpBIMW50Ls|OcM19yL9Qg>

L = 1000
Z = 2
P = 2
IMG_PATH = "./5-redes-complexas/img/"

# |%%--%%| <OcM19yL9Qg|tGB5KTITxm>


def generate_matrix(size: int, z: int, p: float):
    matrix_adj = np.zeros((size, size))
    for i in np.arange(size):
        for j in np.arange(z):
            matrix_adj[i][(i + j + 1) % size] = 1
    num_connections = int(p * size * z / 2)
    pairs = np.random.randint(
        low=0,
        high=size,
        size=(num_connections, 2),
    )
    for i, j in pairs:
        matrix_adj[i][j] = 1
    return matrix_adj


# |%%--%%| <tGB5KTITxm|gx4PNjhNAq>


def print_graph(g: nx.Graph):
    nx.draw_circular(g)


# |%%--%%| <gx4PNjhNAq|aIXMqSnEVE>


def plot_bins(grx: rx.PyGraph):
    distances = rx.distance_matrix(grx).astype(int)
    distances_flattened = np.bincount(np.reshape(distances, distances.size))
    dist_range = np.arange(distances_flattened.size)
    # Nós dividimos por 2 pois a matriz de distâncias é simétrica
    plt.bar(dist_range, distances_flattened / 2)
    plt.xticks(dist_range)
    plt.title("Distruibuição das distâncias entre os nós do grafo")
    plt.xlabel("Distância")
    plt.ylabel("Número de ocorrências")


# |%%--%%| <aIXMqSnEVE|M0xNJbW9ov>


def gen_plot_save(size: int, neighbors: int, prob: float):
    adj_matrix = generate_matrix(size, neighbors, prob)
    graph = nx.Graph(adj_matrix)
    print_graph(graph)
    plt.savefig(f"{IMG_PATH}Graph L={size} Z={neighbors} P={prob}.png")
    plt.show()
    graph_rx = rx.networkx_converter(graph)
    plot_bins(graph_rx)
    plt.savefig(f"{IMG_PATH}Bins L={size} Z={neighbors} P={prob}.png")


# |%%--%%| <M0xNJbW9ov|WD1bzJACfQ>


def gen_plot(size: int, neighbors: int, prob: float):
    adj_matrix = generate_matrix(size, neighbors, prob)
    graph = nx.Graph(adj_matrix)
    graph_rx = rx.networkx_converter(graph)
    plot_bins(graph_rx)


# |%%--%%| <WD1bzJACfQ|UCooGWOo7c>

# Me descomente para salvar as imagens
# gen_plot_save(1000, 2, 0.2)


# |%%--%%| <UCooGWOo7c|YMKnIZp0CZ>

# Me descomente para salvar as imagens
# gen_plot_save(1000, 2, 0.02)

# |%%--%%| <YMKnIZp0CZ|qYsvvXy1Sg>
r"""°°°
## B.2

Conforme o valor de $p$ aumenta, mais arestas são adicionadas ao grafo, o que, em média, diminui as distâncias entre os vértices. Dessa maneira, o ponto médio do histograma se aproxima de valores cada vez menores até atingir o seu mínimo: um grafo completo, em que a menor distância entre quaisquer 2 vértices é 1 (estamos desconsiderando o caso em que um vértice está ligado em si mesmo).

Não existe um valor de $p$ que garante os "seis graus de separação", uma vez que o modelo é probabílistico. No entanto, em nosso experimentos, avaliamos empiricamente que o valor $p=1.8$ muitas vezes produzia grafos de modo que a maior distância entre os grafos era 6. Portanto, parece um valor seguro para termos os 6 graus de separação no grafo com os parâmetros propostos.
°°°"""
# |%%--%%| <qYsvvXy1Sg|wMn9Mrw2su>
