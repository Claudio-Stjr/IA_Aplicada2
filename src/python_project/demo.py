# Integrantes: Alexsandro Narducci, Claudio Strohaecker Júnior, Lucas Cãndido Belletti
import math
import matplotlib.pyplot as plt

def correlacao(x, y):
    """Calcula a correlacao linear entre duas listas de valores."""
    media_x = sum(x) / len(x)
    media_y = sum(y) / len(y)

    numerador = sum(
        (xi - media_x) * (yi - media_y)
        for xi, yi in zip(x, y)
    )

    soma_x = sum((xi - media_x) ** 2 for xi in x)
    soma_y = sum((yi - media_y) ** 2 for yi in y)

    # não há tratamento de divisão por zero — presume variância positiva
    return numerador / math.sqrt(soma_x * soma_y)

# y = β0 + β1x
def regressao(x, y): 
    """Calcula o intercepto e a inclinacao da reta de regressao."""
    media_x = sum(x) / len(x)
    media_y = sum(y) / len(y)

    beta1 = sum(
        (xi - media_x) * (yi - media_y)
        for xi, yi in zip(x, y)
    ) / sum((xi - media_x) ** 2 for xi in x)

    beta0 = media_y - beta1 * media_x

    return beta0, beta1


def demo():
    """Compara correlacao e regressao linear em quatro conjuntos de dados."""
    datasets = [
        (
            "Dataset 1",
            [10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5],
            [8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68],
        ),
        (
            "Dataset 2",
            [10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5],
            [9.14, 8.14, 8.47, 8.77, 9.26, 8.10, 6.13, 3.10, 9.13, 7.26, 4.74],
        ),
        (
            "Dataset 3",
            [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 19],
            [6.58, 5.76, 7.71, 8.84, 8.47, 7.04, 5.25, 5.56, 7.91, 6.89, 12.50],
        ),
        (
            "Dataset 4",
            [10,     8,      9,   11,   14,    6,    4,   12,     7,    5],
            [7.46, 6.77, 7.11, 7.81, 8.84, 6.08, 5.39, 8.15, 6.42, 5.73], # corrigir para 8.50 para que ele possa se ajustar na reta
                                                                                 # 12.72, outlier
        ),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.ravel()

    for ax, (nome, x, y) in zip(axes, datasets):
        # r e a reta podem ser influenciados por outliers e padrões não lineares
        r = correlacao(x, y)
        beta0, beta1 = regressao(x, y)

        # 1. Gráfico de dispersão
        ax.scatter(x, y, color="royalblue", label="Dados")

        # 3. Linha da regressão
        x_linha = [min(x), max(x)]
        y_linha = [
            beta0 + beta1 * x_linha[0],
            beta0 + beta1 * x_linha[1],
        ]

        ax.plot(
            x_linha,
            y_linha,
            color="red",
            label="Regressão linear",
        )

        # 4. Título com os resultados
        ax.set_title(
            f"{nome}\n"
            f"r = {r:.4f} | "
            f"y = {beta0:.4f} + {beta1:.4f}x"
        )

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.grid(True)
        ax.legend()

        print(f"{nome}:")
        print(f"  Correlação: r = {r:.4f}")
        print(f"  Regressão: y = {beta0:.4f} + {beta1:.4f}x\n")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    demo()


    # questão 3 - Qual dos datasets não é apropriado para regressão linear? Justifique sua resposta.

    # O Dataset 2 não é apropriado para regressão linear, pois apresenta uma 
    # relação curvilínea entre as variáveis X e Y. A regressão linear utilizada no exercício 
    # pressupõe uma relação aproximadamente linear entre as variáveis, enquanto os dados do Dataset 2 
    # apresentam um padrão não linear. Portanto, uma reta não representa adequadamente o comportamento
    # dos dados. Nesse caso, seria mais adequado utilizar um modelo de regressão não linear ou transformar
    # as variáveis antes de realizar a modelagem.
    
    
    # questão 4 - Ao analisar o gráfico de dispersão e o resultado da regressão linear para o dataset 4, observa-se um problema. 
    # O que deveria ser feito antes de ajustar o modelo de regressão? Justifique sua resposta.
    
    # Antes de ajustar o modelo de regressão linear para o Dataset 4, deve-se identificar e analisar o
    # outlier presente nos dados, representado pelo ponto (13, 12.74). Esse ponto está distante dos demais
    # e pode exercer grande influência sobre a reta de regressão e sobre o coeficiente de correlação, 
    # prejudicando a representação do comportamento geral dos dados. Portanto, deve-se verificar se 
    # esse valor está correto e, caso seja um erro, corrigi-lo ou removê-lo antes de ajustar o modelo.
    # ou seja, se corrigimos o ponto x13 y 12.74 para x13 y 8.50, a reta de regressão se ajusta melhor aos 
    # dados e o coeficiente de correlação aumenta, refletindo uma relação mais forte entre as variáveis X e Y.