r"""°°°
# Exercício avaliativo 1
## Introdução a Física Estatística e Computacional

Luís Felipe Ramos Ferreira - 2019022553

Igor Lacerda Faria da Silva - 2020041973

Gabriel Rocha Martins - 2019006639
°°°"""
# |%%--%%| <hhAnwT2oUw|fxCq8VVq5E>

import numpy as np
import matplotlib.pyplot as plt

# |%%--%%| <fxCq8VVq5E|uB0SICV32l>

from typing import Callable

# |%%--%%| <uB0SICV32l|IUaidL4GFq>

Y_MAX_1 = 1
INF_1 = 0
SUP_1 = 1
SIZE_1 = 1000
ITERATIONS_PLOT = 1000
SIZES = [100, 1000, 10000]

# |%%--%%| <IUaidL4GFq|YIKXtt19iS>

x = np.linspace(0, 1, num=500)
y = 1 - x**2

plt.plot(x, y)
plt.show()

# |%%--%%| <YIKXtt19iS|EP2n26YOmI>


def funct_1(x: float) -> float:
    return 1 - x**2


# |%%--%%| <EP2n26YOmI|W6P36uSpVa>


def first_method(inf: float, sup: float, funct, size: int, y_max: int):
    inside = 0
    for i in range(size):
        x = np.random.uniform(inf, sup, 1)
        y = np.random.uniform(0, y_max, 1)
        expected_y = funct(x)
        if expected_y > y:
            inside += 1

    return inside / size * y_max * (sup - inf)


# |%%--%%| <W6P36uSpVa|hIOA3Kje1S>


def second_method(inf: float, sup: float, funct, size: int):
    mutiplier = (sup - inf) / size
    x = np.random.uniform(inf, sup, size)
    x = funct(x)
    return mutiplier * np.sum(x)


# |%%--%%| <hIOA3Kje1S|tob9o6ohhg>


def plot_hist_iterate_method(
    iterations: int, inf: float, sup: float, funct, size: int, y
):
    data = np.zeros(iterations)
    for i in range(iterations):
        data[i] = choose_method(inf, sup, funct, size, y)
    plt.hist(data)


# |%%--%%| <tob9o6ohhg|aeI8BzcCf5>


def choose_method(inf: float, sup: float, funct, size: int, y):
    if y is not None:
        return first_method(inf, sup, funct, size, y)
    else:
        return second_method(inf, sup, funct, size)


# |%%--%%| <aeI8BzcCf5|oQYU2M50WQ>

plot_hist_iterate_method(ITERATIONS_PLOT, INF_1, SUP_1, funct_1, SIZE_1, None)

# |%%--%%| <oQYU2M50WQ|iJsXvfyynG>

for i in SIZES:
    plot_hist_iterate_method(ITERATIONS_PLOT, INF_1, SUP_1, funct_1, i, Y_MAX_1)
    plot_hist_iterate_method(ITERATIONS_PLOT, INF_1, SUP_1, funct_1, i, None)
    plt.show()

# |%%--%%| <iJsXvfyynG|N6ZllCxLtl>

INF_2 = 0
SUP_2 = 1
Y_MAX_2 = 3

# |%%--%%| <N6ZllCxLtl|kd3wtOCzpX>


def plot_2():
    x = np.linspace(0, 1, 100)
    y = np.e**x
    plt.plot(x, y)


# |%%--%%| <kd3wtOCzpX|WoeccuvDbf>


def funct_2(x: float) -> float:
    return np.e**x


# |%%--%%| <WoeccuvDbf|bkc3zNKBGb>

plot_hist_iterate_method(ITERATIONS_PLOT, INF_2, SUP_2, funct_2, 1000, Y_MAX_2)

# |%%--%%| <bkc3zNKBGb|xltxGYe6q0>

for i in SIZES:
    plot_hist_iterate_method(ITERATIONS_PLOT, INF_2, SUP_2, funct_2, i, Y_MAX_2)
    plot_hist_iterate_method(ITERATIONS_PLOT, INF_2, SUP_2, funct_2, i, None)
    plt.show()

# |%%--%%| <xltxGYe6q0|nNkIFdjELH>

INF_3 = 0
SUP_3 = np.pi
Y_MAX_3 = 1

# |%%--%%| <nNkIFdjELH|tmO3Tr9UQi>


def plot_3():
    x = np.linspace(INF_3, SUP_3, 100)
    y = np.sin(x) ** 2
    plt.title("Função 3 - $\sin(x)^2")
    plt.plot(x, y)


plot_3()

# |%%--%%| <tmO3Tr9UQi|mzivS3wjH5>


def funct_3(x: float) -> float:
    return np.sin(x) ** 2


# |%%--%%| <mzivS3wjH5|0JgqBkv03n>

# plot_hist_iterate_method(ITERATIONS_PLOT, INF_3, SUP_3, funct_3, 100, Y_MAX_3)

# |%%--%%| <0JgqBkv03n|ndMEd2YPDo>
r"""°°°
# Função 3
°°°"""
# |%%--%%| <ndMEd2YPDo|KFDFTBySNV>

for i in SIZES:
    plot_hist_iterate_method(ITERATIONS_PLOT, INF_3, SUP_3, funct_3, i, Y_MAX_3)
    plot_hist_iterate_method(ITERATIONS_PLOT, INF_3, SUP_3, funct_3, i, None)
    plt.show()

# |%%--%%| <KFDFTBySNV|ehEZ9MbGL0>
r"""°°°
Em primeiro lugar, como esperado pelo Teorema do Limite Central, a distribuição
dos valores gerados nos histogramas se aproxima de uma distribuição normal,
com a média muito próxima do valor analítico. Além disso, em todos os casos, o
método 2 apresentou um conjunto de resultados cujos valores possuem um desvio
padrão menor do que os gerados pelo método 1, o que nos leva a inferir que ele
teve um desempenho mais promissor na estimativa das integrais. Entretanto,
devemos tentar entender o porque disso.

As funções testadas com o método 1, de amostragem de pontos abaixo da curva,
possuem "uma camada de aleatoriedade maior" do que as testadas com o método 2,
uma vez que devemos gerar pontos aleatórios, sendo que estes possuem
coordenadas x e y. O método de Monte Carlo, utilizando o valor médio, seleciona
aleatoriamente apenas o valor de x, tornando sua distribuição de valores menos
incerta. Assim, o desvio padrão do segundo método se mostra menor. 
°°°"""
# |%%--%%| <ehEZ9MbGL0|7FAEGObrRw>


def calculo_erro(values, mean, N):
    variancia = np.square(values - mean).mean()
    desvio = np.sqrt(variancia)
    return desvio / np.sqrt(N)


# |%%--%%| <7FAEGObrRw|Ddlk1fCE08>
r"""°°°
# Exercício 4
°°°"""
# |%%--%%| <Ddlk1fCE08|jGY4loShRJ>
r"""°°°
Neste exercício utilizaremos o método 2 de Monte Carlo para aproximar o valor
de um integral em 9 dimensões
°°°"""
# |%%--%%| <jGY4loShRJ|PmuoidDmAk>
r"""°°°
Criando função:
°°°"""
# |%%--%%| <PmuoidDmAk|89syOrRWv3>


def funct4(x: list):
    return 1 / ((x[0] + x[1]) * x[2] + (x[3] + x[4]) * x[5] + (x[6] + x[7]) * x[8])


# |%%--%%| <89syOrRWv3|qXPSrvmMFW>
r"""°°°
Criando funçao que generaliza a aplicaçao do método 2 ( MonteCarlo ) para esse caso
°°°"""
# |%%--%%| <qXPSrvmMFW|apwEs1zQik>


def MonteCarlo_9d(N, funcao: Callable):
    acumulador = 0
    for i in range(N):
        acumulador = acumulador + funcao(np.random.uniform(0, 1, 9))
    return acumulador / N


# |%%--%%| <apwEs1zQik|XnZ4LSeipw>
r"""°°°
Usando funçao para criar 1000 amostras com N=100 e aproximar o valor dessa integral
°°°"""
# |%%--%%| <XnZ4LSeipw|WFnFrNdAoU>

ITERATIONS_4 = 1000

# |%%--%%| <WFnFrNdAoU|eJjvmq6Vtn>


def carlao(size: float):
    amostra = np.zeros(ITERATIONS_4)
    for i in range(ITERATIONS_4):
        amostra[i] = MonteCarlo_9d(size, funct4)
    return amostra.mean(), amostra


# |%%--%%| <eJjvmq6Vtn|EucJAaK1yb>

media_1, amostra_1 = carlao(10**2)

# |%%--%%| <EucJAaK1yb|oTn4pzzZK7>

calculo_erro(amostra_1, media_1, 10**2)

# |%%--%%| <oTn4pzzZK7|P8z3f8G7W8>

plt.hist(amostra_1)

# |%%--%%| <P8z3f8G7W8|RcDGk5WzrH>

media_2, amostra_2 = carlao(10**3)

# |%%--%%| <RcDGk5WzrH|0URzhDm8jv>

calculo_erro(amostra_2, media_2, 10**2)

# |%%--%%| <0URzhDm8jv|PKJB5Wngn3>

plt.hist(amostra_2)

# |%%--%%| <PKJB5Wngn3|T6YsRxWzds>

media_3, amostra_3 = carlao(10**4)

# |%%--%%| <T6YsRxWzds|lvloOZ0y3Q>

calculo_erro(amostra_3, media_3, 10**2)

# |%%--%%| <lvloOZ0y3Q|UH8ZIHamU8>

plt.hist(amostra_3)

# |%%--%%| <UH8ZIHamU8|ZyQePKSO1g>

# media_4, amostra_4 = carlao(10**5)

# |%%--%%| <ZyQePKSO1g|4xMi6H3WEH>

# calculo_erro(amostra_4, media_4, 10)

# |%%--%%| <4xMi6H3WEH|KsSdyyIR9q>

# plt.hist(amostra_4)
