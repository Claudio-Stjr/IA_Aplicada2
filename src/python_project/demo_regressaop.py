"""Demonstracao de regressao polinomial e overfitting."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score


GRAUS_PRINCIPAIS = (1, 2, 3, 8)
SEMENTE = 42


def carregar_dados(caminho: str | Path) -> tuple[np.ndarray, np.ndarray]:
	"""Le a primeira e a segunda coluna de um CSV sem cabecalho."""
	dados = np.loadtxt(caminho, delimiter=",")
	if dados.ndim != 2 or dados.shape[1] < 2:
		raise ValueError("O arquivo deve conter pelo menos duas colunas numericas.")
	return dados[:, 0], dados[:, 1]


def avaliar_polinomio(coeficientes: np.ndarray, x: np.ndarray) -> np.ndarray:
	"""Avalia coeficientes em ordem decrescente usando a regra de Horner."""
	resultado = np.zeros_like(np.asarray(x, dtype=float))
	for coeficiente in coeficientes:
		resultado = resultado * x + coeficiente
	return resultado


def ajustar_e_avaliar(
	x_treino: np.ndarray,
	y_treino: np.ndarray,
	x_avaliacao: np.ndarray,
	y_avaliacao: np.ndarray,
	grau: int,
) -> tuple[np.ndarray, float, float]:
	"""Ajusta com polyfit e devolve coeficientes, EQM e R2."""
	coeficientes = np.polyfit(x_treino, y_treino, grau)
	previsoes = avaliar_polinomio(coeficientes, x_avaliacao)
	return (
		coeficientes,
		mean_squared_error(y_avaliacao, previsoes),
		r2_score(y_avaliacao, previsoes),
	)


def dividir_dados(
	x: np.ndarray, y: np.ndarray, percentual_teste: float = 0.1
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
	"""Divide os dados aleatoriamente, mantendo 10% no conjunto de teste."""
	if not 0 < percentual_teste < 1:
		raise ValueError("O percentual de teste deve estar entre 0 e 1.")
	rng = np.random.default_rng(SEMENTE)
	indices = rng.permutation(x.size)
	quantidade_teste = max(1, int(round(x.size * percentual_teste)))
	indices_teste = indices[:quantidade_teste]
	indices_treino = indices[quantidade_teste:]
	return x[indices_treino], x[indices_teste], y[indices_treino], y[indices_teste]


def imprimir_resultados(
	graus: tuple[int, ...],
	x: np.ndarray,
	y: np.ndarray,
	x_treino: np.ndarray,
	y_treino: np.ndarray,
	x_teste: np.ndarray,
	y_teste: np.ndarray,
) -> dict[int, np.ndarray]:
	"""Imprime as metricas dos dados completos, treino e teste."""
	print("\nDados completos")
	print("grau | EQM")
	coeficientes_completos = {}
	for grau in graus:
		coeficientes, eqm, _ = ajustar_e_avaliar(x, y, x, y, grau)
		coeficientes_completos[grau] = coeficientes
		print(f"{grau:4d} | {eqm:.8f}")

	print("\nTreino e teste (10% para teste)")
	print("grau | EQM treino | R2 treino | EQM teste | R2 teste")
	resultados_teste = {}
	for grau in graus:
		_, eqm_treino, r2_treino = ajustar_e_avaliar(
			x_treino, y_treino, x_treino, y_treino, grau
		)
		_, eqm_teste, r2_teste = ajustar_e_avaliar(
			x_treino, y_treino, x_teste, y_teste, grau
		)
		resultados_teste[grau] = (eqm_teste, r2_teste)
		print(
			f"{grau:4d} | {eqm_treino:10.8f} | {r2_treino:9.6f} | "
			f"{eqm_teste:9.8f} | {r2_teste:8.6f}"
		)
	melhor_grau = min(resultados_teste, key=lambda grau: resultados_teste[grau][0])
	print(
		f"\nMelhor desempenho no teste: grau {melhor_grau} "
		f"(menor EQM = {resultados_teste[melhor_grau][0]:.8f})."
	)
	if 20 in resultados_teste and resultados_teste[20][0] > resultados_teste[melhor_grau][0]:
		print(
			"Conclusao: o grau 20 ajusta melhor o treino, mas oscila no teste; "
			"isso caracteriza overfitting."
		)
	return coeficientes_completos


def criar_graficos(
	x: np.ndarray,
	y: np.ndarray,
	x_treino: np.ndarray,
	y_treino: np.ndarray,
	x_teste: np.ndarray,
	y_teste: np.ndarray,
	coeficientes_completos: dict[int, np.ndarray],
	mostrar: bool,
	diretorio_saida: str | None,
) -> None:
	"""Cria o grafico completo e o grafico do ajuste feito apenas no treino."""
	x_linha = np.linspace(x.min(), x.max(), 600)
	cores = {1: "#d73027", 2: "#1a9850", 3: "#252525", 8: "#e6ab02", 20: "#7b3294"}
	plt.rcParams.update({
		"font.size": 10,
		"axes.titleweight": "bold",
		"axes.titlesize": 14,
		"axes.labelsize": 11,
		"legend.frameon": True,
		"legend.framealpha": 0.95,
	})

	def preparar_eixo(eixo: plt.Axes, titulo: str, subtitulo: str) -> None:
		eixo.set_title(titulo, loc="left", pad=18)
		eixo.text(
			0,
		1.01,
			subtitulo,
			transform=eixo.transAxes,
			color="#666666",
			fontsize=9,
			va="bottom",
		)
		eixo.set_xlabel("Variavel independente (x)")
		eixo.set_ylabel("Variavel dependente (y)")
		eixo.grid(True, color="#d9d9d9", linewidth=0.8, alpha=0.7)
		eixo.set_axisbelow(True)
		eixo.spines["top"].set_visible(False)
		eixo.spines["right"].set_visible(False)

	fig_completo, eixo = plt.subplots(figsize=(11, 6.5), constrained_layout=True)
	fig_completo.patch.set_facecolor("#fafafa")
	eixo.set_facecolor("#fafafa")
	eixo.scatter(
		x,
		y,
		color="#2c7fb8",
		edgecolors="white",
		linewidth=0.7,
		s=42,
		alpha=0.82,
		label="Observacoes",
	)
	for grau, coeficientes in coeficientes_completos.items():
		eixo.plot(
			x_linha,
			avaliar_polinomio(coeficientes, x_linha),
			color=cores[grau],
			linewidth=2.2 if grau != 20 else 1.8,
			linestyle="--" if grau == 20 else "-",
			label=f"Polinomio grau {grau}",
		)
	preparar_eixo(
		eixo,
		"Regressao polinomial",
		f"Ajuste dos modelos aos {len(x)} pontos do conjunto completo",
	)
	eixo.legend(loc="upper left", bbox_to_anchor=(1.01, 1), title="Modelos")

	fig_dividido, eixos = plt.subplots(1, 2, figsize=(15, 6.5), constrained_layout=True)
	fig_dividido.patch.set_facecolor("#fafafa")
	for eixo in eixos:
		eixo.set_facecolor("#fafafa")

	def adicionar_pontos(eixo: plt.Axes) -> None:
		eixo.scatter(
			x_treino,
			y_treino,
			color="#2c7fb8",
			edgecolors="white",
			linewidth=0.7,
			s=42,
			alpha=0.78,
			label="Treinamento (90%)",
		)
		eixo.scatter(
			x_teste,
			y_teste,
			color="#f46d43",
			edgecolors="white",
			linewidth=0.9,
			s=64,
			alpha=0.95,
			label="Teste (10%)",
		)

	adicionar_pontos(eixos[0])
	for grau in GRAUS_PRINCIPAIS:
		coeficientes, _, _ = ajustar_e_avaliar(
			x_treino, y_treino, x_treino, y_treino, grau
		)
		eixos[0].plot(
			x_linha,
			avaliar_polinomio(coeficientes, x_linha),
			color=cores[grau],
			linewidth=2.2 if grau != 20 else 1.8,
			linestyle="--" if grau == 20 else "-",
			label=f"Polinomio grau {grau}",
		)
	preparar_eixo(eixos[0], "Modelos de graus 1 a 8", "Ajustados somente com o treinamento")
	eixos[0].legend(loc="upper left", title="Dados e modelos")

	adicionar_pontos(eixos[1])
	coeficientes, _, _ = ajustar_e_avaliar(
		x_treino, y_treino, x_treino, y_treino, 20
	)
	eixos[1].plot(
		x_linha,
		avaliar_polinomio(coeficientes, x_linha),
		color=cores[20],
		linewidth=1.8,
		linestyle="--",
		label="Polinomio grau 20",
	)
	preparar_eixo(eixos[1], "Grau 20: overfitting", "A escala evidencia a oscilacao da curva")
	eixos[1].legend(loc="upper left", title="Dados e modelo")

	if diretorio_saida:
		diretorio = Path(diretorio_saida)
		diretorio.mkdir(parents=True, exist_ok=True)
		fig_completo.savefig(
			diretorio / "regressao_polinomial_completa.png",
			dpi=180,
			bbox_inches="tight",
		)
		fig_dividido.savefig(
			diretorio / "regressao_polinomial_treino.png",
			dpi=180,
			bbox_inches="tight",
		)
	if mostrar:
		plt.show()
	else:
		plt.close("all")


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument(
		"--data",
		default=Path(__file__).resolve().parents[2] / "data_preg.csv",
		help="caminho do CSV com x e y",
	)
	parser.add_argument("--save-dir", help="diretorio para salvar os graficos")
	parser.add_argument("--no-show", action="store_true", help="nao abrir janelas")
	args = parser.parse_args()

	x, y = carregar_dados(args.data)
	x_treino, x_teste, y_treino, y_teste = dividir_dados(x, y)
	graus = (*GRAUS_PRINCIPAIS, 20)
	coeficientes_completos = imprimir_resultados(
		graus, x, y, x_treino, y_treino, x_teste, y_teste
	)
	criar_graficos(
		x,
		y,
		x_treino,
		y_treino,
		x_teste,
		y_teste,
		coeficientes_completos,
		mostrar=not args.no_show,
		diretorio_saida=args.save_dir,
	)


if __name__ == "__main__":
	main()


# A) Qual arquivo foi utilizado e qual e o formato dos dados?
# Resposta: Foi utilizado o arquivo data_preg.csv. A primeira coluna contem os
# valores de x e a segunda coluna contem os valores de y. O arquivo possui duas
# colunas numericas, sem cabecalho.

# B) Como foi feito o grafico de dispersao?
# Resposta: Os pares (x, y) foram representados com eixo x para a variavel
# independente e eixo y para a variavel dependente. Os pontos azuis mostram as
# observacoes do conjunto completo.

# C) Como foi ajustada a regressao polinomial de grau 1?
# Resposta: O ajuste foi feito com numpy.polyfit(x, y, 1). A curva foi
# desenhada em vermelho. O vetor devolvido por polyfit vem em ordem decrescente
# de potencia: [beta_1, beta_0], e nao [beta_0, beta_1].

# D) Como foi ajustada e desenhada a regressao de grau 2?
# Resposta: Os coeficientes foram obtidos com numpy.polyfit(x, y, 2). A funcao
# avaliar_polinomio calcula y = beta_0 + beta_1*x + beta_2*x**2 manualmente,
# usando a regra de Horner, sem usar uma funcao pronta para avaliar a curva.
# Ela e mostrada em verde.

# E) Como foi desenhada a regressao polinomial de grau 3?
# Resposta: O modelo foi ajustado com polyfit e avaliado pela mesma funcao
# manual avaliar_polinomio. A curva de grau 3 e mostrada em preto.

# F) Como foi desenhada a regressao polinomial de grau 8?
# Resposta: O modelo foi ajustado com polyfit e avaliado na grade de 600 pontos
# entre o menor e o maior valor de x. A curva de grau 8 e mostrada em amarelo.

# G) Qual modelo teve o menor EQM nos dados completos?
# Resposta: Nos dados completos, os EQMs foram aproximadamente: grau 1 =
# 0.26593426, grau 2 = 0.13852649, grau 3 = 0.07869725, grau 8 = 0.05870935
# e grau 20 = 0.05196887. Portanto, o grau 20 foi o mais preciso nesse mesmo
# conjunto usado para ajustar os modelos. Isso, sozinho, nao garante melhor
# desempenho em dados novos.

# H) Como os dados foram divididos em treinamento e teste?
# Resposta: Os indices foram embaralhados com numpy.random.default_rng usando
# a semente 42. Foram usados 10% dos dados no teste e os 90% restantes no
# treinamento. A semente torna a divisao reproduzivel.

# I) Como foram refeitos os ajustes usando apenas os dados de treinamento?
# Resposta: Para cada grau, numpy.polyfit foi aplicado somente a x_treino e
# y_treino. As curvas resultantes foram desenhadas no segundo grafico, junto
# com os pontos de treinamento e de teste.

# J) Qual foi o EQM calculado somente nos dados de teste?
# Resposta: Os EQMs de teste foram aproximadamente: grau 1 = 0.20784251,
# grau 2 = 0.13499874, grau 3 = 0.10170281, grau 8 = 0.11767827 e grau 20 =
# 76.24500972. O menor EQM de teste foi o do grau 3.

# K) Quais foram os valores do coeficiente de determinacao R2?
# Resposta: No treinamento, os valores de R2 foram aproximadamente 0.511623
# (grau 1), 0.750374 (grau 2), 0.861076 (grau 3), 0.893075 (grau 8) e
# 0.908932 (grau 20). No teste, foram 0.565411, 0.717724, 0.787344, 0.753940
# e -158.424819, respectivamente. O R2 mede a proporcao da variabilidade de y
# explicada pelo modelo em relacao a prever sempre a media de y. Um R2 negativo
# no teste indica desempenho pior que essa previsao pela media.

# L) Qual modelo foi mais preciso neste caso?
# Resposta: Considerando a capacidade de generalizacao, o modelo de grau 3 foi
# o mais preciso: apresentou o menor EQM de teste e o maior R2 de teste entre
# os modelos analisados. O grau 8 e o grau 20 reduziram mais o erro de treino,
# mas tiveram desempenho inferior no teste.

# M) O que acontece com o modelo de grau 20?
# Resposta: A curva de grau 20 oscila fortemente, especialmente nas bordas do
# intervalo. Ela consegue acompanhar o ruido dos dados de treinamento, por
# isso apresenta EQM de treino baixo, mas produz previsoes muito ruins para os
# dados de teste. Esse comportamento e overfitting: aumentar a complexidade
# melhora o ajuste dos dados conhecidos, mas prejudica a generalizacao. Assim,
# o grau 20 nao apresenta melhor desempenho nos dados de teste.
