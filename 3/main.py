r"""°°°
# Exercício avaliativo 1
## Introdução a Física Estatística e Computacional

Luís Felipe Ramos Ferreira - 2019022553

Igor Lacerda Faria da Silva - 2020041973

Gabriel Rocha Martins - 2019006639
°°°"""
# |%%--%%| <9mXIl9nkXy|iRL5gMsstH>

import numpy as np
import matplotlib.pyplot as plt

# |%%--%%| <iRL5gMsstH|PbFbtyTq7I>

from typing import Callable

# |%%--%%| <PbFbtyTq7I|xN6impb1pJ>

ITERATIONS = 1000
SIZES = [10**2, 10**3, 10**4]
errors: list[float] = []

# |%%--%%| <xN6impb1pJ|HsYJ16xuwZ>

def calculo_erro(values: np.ndarray, mean: float) -> float:
    variancia = np.square(values - mean).mean()
    desvio = np.sqrt(variancia)
    return desvio / np.sqrt(values.size)

# |%%--%%| <HsYJ16xuwZ|h7MZ8QtJvI>

def first_method(inf: float, sup: float, funct, size: int, y_max: int):
    inside = 0
    for i in range(size):
        x = np.random.uniform(inf, sup, 1)
        y = np.random.uniform(0, y_max, 1)
        expected_y = funct(x)
        if expected_y > y:
            inside += 1

    return inside / size * y_max * (sup - inf)

# |%%--%%| <h7MZ8QtJvI|cDJUexvCuk>

def second_method(inf: float, sup: float, funct, size: int):
    mutiplier = (sup - inf) / size
    x = np.random.uniform(inf, sup, size)
    y = funct(x)
    return mutiplier * np.sum(y)

# |%%--%%| <cDJUexvCuk|i8XSZc97EB>

def choose_method(inf: float, sup: float, funct, size: int, y):
    if y is not None:
        return first_method(inf, sup, funct, size, y)
    else:
        return second_method(inf, sup, funct, size)

# |%%--%%| <i8XSZc97EB|mNvG3mxrvl>

def plot_hist_iterate_method(
    iterations: int, inf: float, sup: float, funct, size: int, y
):
    data: np.ndarray = np.zeros(iterations)
    for i in range(iterations):
        data[i] = choose_method(inf, sup, funct, size, y)
    plt.hist(data)
    return calculo_erro(data, data.mean())

# |%%--%%| <mNvG3mxrvl|to9NuhwRB6>

def plot_all(inf: float, sup: float, funct: Callable, y: float | None):
    for size in SIZES:
        errors.append(plot_hist_iterate_method(ITERATIONS, inf, sup, funct, size, y))
        errors.append(plot_hist_iterate_method(ITERATIONS, inf, sup, funct, size, None))
        plt.show()

# |%%--%%| <to9NuhwRB6|SY2pi1SESw>
r"""°°°
# Função 1
°°°"""
# |%%--%%| <SY2pi1SESw|jBAPrBcCXh>

def plot_1():
    x = np.linspace(0, 1, 100)
    y = 1 - x**2
    plt.title("$1 - x^2$")
    plt.plot(x, y)


plot_1()

# |%%--%%| <jBAPrBcCXh|cREtW1Tuma>

def funct_1(x: float) -> float:
    return 1 - x**2

# |%%--%%| <cREtW1Tuma|EEzd0Fa9KD>

INF_1 = 0
SUP_1 = 1
Y_MAX_1 = 1

# |%%--%%| <EEzd0Fa9KD|ap1uyUoh2a>

plot_all(INF_1, SUP_1, funct_1, Y_MAX_1)

# |%%--%%| <ap1uyUoh2a|HsiF964HAC>
r"""°°°
# Função 2
°°°"""
# |%%--%%| <HsiF964HAC|qZ2Opt7df2>

def plot_2():
    x = np.linspace(0, 1, 100)
    y = np.e**x
    plt.title("$e^x$")
    plt.plot(x, y)


plot_2()

# |%%--%%| <qZ2Opt7df2|2GOVvLSQRa>

def funct_2(x: float) -> float:
    return np.e**x

# |%%--%%| <2GOVvLSQRa|iTySn7cLeM>

INF_2 = 0
SUP_2 = 1
Y_MAX_2 = 3

# |%%--%%| <iTySn7cLeM|jLyFt484BR>

plot_all(INF_2, SUP_2, funct_2, Y_MAX_2)

# |%%--%%| <jLyFt484BR|5S7mj5cET5>
r"""°°°
# Função 3
°°°"""
# |%%--%%| <5S7mj5cET5|A0GC30Ab6c>

def plot_3():
    x = np.linspace(INF_3, SUP_3, 100)
    y = np.sin(x) ** 2
    plt.title("$\sin(x)^2$")
    plt.plot(x, y)


plot_3()

# |%%--%%| <A0GC30Ab6c|yMAiVqu0zm>

def funct_3(x: float) -> float:
    return np.sin(x) ** 2

# |%%--%%| <yMAiVqu0zm|ftlodFzdN2>

INF_3 = 0
SUP_3 = np.pi
Y_MAX_3 = 1

# |%%--%%| <ftlodFzdN2|juhIGzR5e2>

plot_all(INF_3, SUP_3, funct_3, Y_MAX_3)

# |%%--%%| <juhIGzR5e2|ZfJUYuVduk>
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
# |%%--%%| <ZfJUYuVduk|EqX5Dpr7kM>

errors

# |%%--%%| <EqX5Dpr7kM|7JYWek0hFb>
r"""°°°
## Erros

O cálculo dos erros estimativos seguiram o que era esperado: quanto maior o
número de valores gerados para estimar a integral, menor será a diferença entre
a estimativa e o valor analítico. É importante salientar que os erros foram
calculados considerando que não há correlação entre os valores gerados, ou
seja, assumimos que o gerador de números aleatórios é perfeito. Na prática esse
pode não ser o caso, mas o gerador de números aleatório da biblioteca Numpy
garante um alto nível de independência entre os números gerados.
°°°"""
# |%%--%%| <7JYWek0hFb|RJAGPYrv7f>
r"""°°°
# Exercício 4
°°°"""
# |%%--%%| <RJAGPYrv7f|kuYz30wLFK>
r"""°°°
Neste exercício utilizaremos o método 2 de Monte Carlo para aproximar o valor
de um integral em 9 dimensões
°°°"""
# |%%--%%| <kuYz30wLFK|oT2wFQ1tMy>

def funct_4(x: list):
    return 1 / ((x[0] + x[1]) * x[2] + (x[3] + x[4]) * x[5] + (x[6] + x[7]) * x[8])

# |%%--%%| <oT2wFQ1tMy|ABrpcT46Wo>
r"""°°°
Criando função que generaliza a aplicação do método 2 ( MonteCarlo ) para esse caso
°°°"""
# |%%--%%| <ABrpcT46Wo|fpFwFhLwoE>

def MonteCarlo_9d(N, funct: Callable):
    acumulador = 0
    for i in range(N):
        acumulador = acumulador + funct(np.random.uniform(0, 1, 9))
    return acumulador / N

# |%%--%%| <fpFwFhLwoE|ij1f4jB38d>
r"""°°°
Usando função para criar 1000 amostras com N=100 e aproximar o valor dessa integral
°°°"""
# |%%--%%| <ij1f4jB38d|T0UeeKAQ9n>

ITERATIONS_4 = 1000
SIZES_4 = [10**2, 10**3, 10**4]

# |%%--%%| <T0UeeKAQ9n|id5PZloQXB>

def carlao(size: float):
    amostra = np.zeros(ITERATIONS_4)
    for i in range(ITERATIONS_4):
        amostra[i] = MonteCarlo_9d(size, funct_4)
    return amostra.mean(), amostra

# |%%--%%| <id5PZloQXB|GMMjPcjxWm>

for size in SIZES_4:
    media, amostra = carlao(size)
    plt.hist(amostra)
    plt.show()
    print(calculo_erro(amostra, media))

# |%%--%%| <GMMjPcjxWm|CcNFvmrZb5>
r"""°°°
O cálculo da integral da função de 9 variáveis se comportou como esperado, dado
que com o aumento no tamanho da amostra para fazer essa aproximação, o desvio
padrão diminuiu. Além disso, é possível notar que a aproximação utilizando o
método 2 apresenta um desempenho melhor do que o esperado, dado que mesmo com o
aumento de números aleatórios criados houve uma certa manutenção no tempo de
execução.
°°°"""