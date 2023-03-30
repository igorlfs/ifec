r"""°°°
# Física Estatística Computacional
## Igor Lacerda Faria
### 2020041973
°°°"""
# |%%--%%| <P9Q9nHUSYa|nyp6W4LD1o>

import matplotlib.pyplot as plt
import numpy as np


# |%%--%%| <nyp6W4LD1o|AMPUjpFnB4>

N = 1000

# |%%--%%| <AMPUjpFnB4|jisTwfEszs>
r"""°°°
# Exercício 1
°°°"""
# |%%--%%| <jisTwfEszs|obb9pKX9d9>


def func(r: float, y0: float):
    y = [y0]
    for i in range(N - 1):
        y.append(r * y[i] * (1 - y[i]))
    x = range(N)
    plt.xlabel("Geração")
    plt.ylabel("População")
    plt.title(f"r = {r} $y_0$ = {y0}")
    plt.plot(x, y)


func(0.5, 0.5)

# |%%--%%| <obb9pKX9d9|4u926Jf8W8>
r"""°°°
# Exercício 2
°°°"""
# |%%--%%| <4u926Jf8W8|FSVNzg49Mc>

rs = [2.5, 3.1, 3.5, 3.7]
for r in rs:
    func(r, 0.5)
    plt.show()

# |%%--%%| <FSVNzg49Mc|7zy98ZsIhl>
r"""°°°
# Exercício 3
°°°"""
# |%%--%%| <7zy98ZsIhl|nQLpp3StgA>

y0s = [0.25, 0.5, 0.75]
for r in rs:
    for y0 in y0s:
        func(r, y0)
    plt.show()

# |%%--%%| <nQLpp3StgA|swoTSru8RO>
r"""°°°
# Exercício 4
°°°"""
# |%%--%%| <swoTSru8RO|RyRiwi0H4y>


y0s = [0.5, 0.501, 0.5001]
for r in rs:
    for y0 in y0s:
        func(r, y0)
    plt.show()

# |%%--%%| <RyRiwi0H4y|Bc9t9Zl0cL>
r"""°°°
# Exercício 5
°°°"""
# |%%--%%| <Bc9t9Zl0cL|9OeBaMk8jb>

r_linspace = np.linspace(10e-5, 4, 10000)
Y: list[float] = list()
R: list[float] = list()
for r in r_linspace:
    R.append(r)
    y = np.random.random()
    for _ in range(N - 1):
        y = (r * y) * (1 - y)
    Y.append(y)
plt.plot(R, Y, ls="", marker=",")
plt.ylabel("$y_0$")
plt.xlabel("$r$")
plt.show()
