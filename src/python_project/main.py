import math
import plotly.graph_objects as go


def correlacao(x, y):
    """Calcula o coeficiente de correlação linear r."""

    if len(x) != len(y):
        raise ValueError("Os vetores precisam ter o mesmo tamanho.")

    media_x = sum(x) / len(x)
    media_y = sum(y) / len(y)

    numerador = sum(
        (xi - media_x) * (yi - media_y)
        for xi, yi in zip(x, y)
    )

    parte_x = sum((xi - media_x) ** 2 for xi in x)
    parte_y = sum((yi - media_y) ** 2 for yi in y)

    denominador = math.sqrt(parte_x * parte_y)

    if denominador == 0:
        raise ValueError("Não é possível calcular a correlação.")

    return numerador / denominador


def regressao(x, y):
    """Calcula beta0 e beta1 da regressão linear."""

    if len(x) != len(y):
        raise ValueError("Os vetores precisam ter o mesmo tamanho.")

    media_x = sum(x) / len(x)
    media_y = sum(y) / len(y)

    numerador = sum(
        (xi - media_x) * (yi - media_y)
        for xi, yi in zip(x, y)
    )

    denominador = sum((xi - media_x) ** 2 for xi in x)

    if denominador == 0:
        raise ValueError("Não é possível calcular a regressão.")

    beta1 = numerador / denominador
    beta0 = media_y - beta1 * media_x

    return beta0, beta1


def criar_grafico(x, y, nome):
    """Cria o gráfico de dispersão e a reta de regressão."""

    beta0, beta1 = regressao(x, y)

    x_min = min(x)
    x_max = max(x)

    reta_x = [x_min, x_max]
    reta_y = [
        beta0 + beta1 * x_min,
        beta0 + beta1 * x_max
    ]

    figura = go.Figure()

    figura.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="markers",
            name="Dados"
        )
    )

    figura.add_trace(
        go.Scatter(
            x=reta_x,
            y=reta_y,
            mode="lines",
            name="Regressão"
        )
    )

    figura.update_layout(
        title=f"{nome} - Dispersão e regressão",
        xaxis_title="X",
        yaxis_title="Y"
    )

    figura.show()


def main():
    x1 = [10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5]
    y1 = [8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68]

    x2 = [10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5]
    y2 = [9.14, 8.14, 8.47, 8.77, 9.26, 8.10, 6.13, 3.10, 9.13, 7.26, 4.74]

    x3 = [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 19]
    y3 = [6.58, 5.76, 7.71, 8.84, 8.47, 7.04, 5.25, 5.56, 7.91, 6.89, 12.50]

    x4 = [10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5]
    y4 = [7.46, 6.77, 12.74, 7.11, 7.81, 8.84, 6.08, 5.39, 8.15, 6.42, 5.73]

    datasets = [
        ("Dataset 1", x1, y1),
        ("Dataset 2", x2, y2),
        ("Dataset 3", x3, y3),
        ("Dataset 4", x4, y4),
    ]

    for nome, x, y in datasets:
        r = correlacao(x, y)
        beta0, beta1 = regressao(x, y)

        print(f"\n{nome}")
        print(f"Correlação: r = {r:.4f}")
        print(f"Regressão: y = {beta0:.4f} + {beta1:.4f}x")

        criar_grafico(x, y, nome)


if __name__ == "__main__":
    main()