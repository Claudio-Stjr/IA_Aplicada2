from __future__ import annotations

import argparse
import io
import os
from pathlib import Path
from urllib.request import urlopen

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def regmultipla(X: np.ndarray, y: np.ndarray) -> np.ndarray:
	"""Calcula beta para y = X beta.

	X deve incluir uma primeira coluna de 1s para o intercepto.
	``lstsq`` e usado em vez de calcular explicitamente a inversa de X.T @ X,
	pois tambem funciona quando as colunas sao quase colineares.
	"""
	X = np.asarray(X, dtype=float)
	y = np.asarray(y, dtype=float).reshape(-1)

	if X.ndim != 2:
		raise ValueError("X deve ser uma matriz bidimensional.")
	if y.ndim != 1 or X.shape[0] != y.shape[0]:
		raise ValueError("X e y devem ter o mesmo numero de observacoes.")

	beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
	return beta


def regressao_simples(x: np.ndarray, y: np.ndarray) -> np.ndarray:
	"""Calcula [intercepto, inclinacao] para uma regressao linear simples."""
	x = np.asarray(x, dtype=float).reshape(-1)
	y = np.asarray(y, dtype=float).reshape(-1)
	if x.size != y.size or x.size < 2:
		raise ValueError("x e y devem ter o mesmo tamanho e pelo menos duas observacoes.")
	matriz = np.column_stack((np.ones(x.size), x))
	return regmultipla(matriz, y)


def _dataframe_from_array(values: np.ndarray) -> pd.DataFrame:
	values = pd.DataFrame(values).apply(pd.to_numeric, errors="coerce").dropna().to_numpy()
	if values.ndim != 2 or values.shape[1] < 3:
		raise ValueError("O arquivo deve conter pelo menos tres colunas numericas.")
	return pd.DataFrame(
		values[:, :3],
		columns=["tamanho", "quartos", "preco"],
	).dropna()


def carregar_dados(data_path: str | os.PathLike[str] | None = None,
				   url: str | None = None) -> pd.DataFrame:
	"""Carrega CSV ou MAT e devolve as tres colunas da atividade."""
	if url:
		with urlopen(url) as response:
			content = response.read()
		suffix = Path(url.split("?", 1)[0]).suffix.lower()
		if suffix == ".mat":
			try:
				from scipy.io import loadmat
			except ImportError as exc:
				raise ImportError(
					"Para ler .mat, instale scipy: python -m pip install scipy"
				) from exc
			values = next(
				value for value in loadmat(io.BytesIO(content)).values()
				if isinstance(value, np.ndarray) and value.ndim == 2 and value.shape[1] >= 3
			)
			return _dataframe_from_array(values)
		return _dataframe_from_array(
			pd.read_csv(io.BytesIO(content), sep=None, engine="python", header=None).to_numpy()
		)

	default_data_path = Path(__file__).resolve().parents[2] / "data (1).csv"
	path = Path(data_path) if data_path else default_data_path
	if not path.exists():
		raise FileNotFoundError(
			f"Arquivo de dados nao encontrado: {path}. "
			"Use --data caminho.csv, --data caminho.mat ou --url URL."
		)

	if path.suffix.lower() == ".mat":
		try:
			from scipy.io import loadmat
		except ImportError as exc:
			raise ImportError(
				"Para ler .mat, instale scipy: python -m pip install scipy"
			) from exc
		matrices = loadmat(path)    
		candidates = [
			value for value in matrices.values()
			if isinstance(value, np.ndarray) and value.ndim == 2 and value.shape[1] >= 3
		]
		if not candidates:
			raise ValueError("Nenhuma matriz com pelo menos tres colunas foi encontrada no .mat.")
		return _dataframe_from_array(candidates[0])

	data = pd.read_csv(path, sep=None, engine="python", header=None)
	return _dataframe_from_array(data.to_numpy())


def criar_graficos(dados: pd.DataFrame, beta: np.ndarray,
				   correlacoes: tuple[float, float], save_dir: str | None = None):
	"""Cria os graficos 2D e 3D pedidos na atividade."""
	tamanho = dados["tamanho"].to_numpy()
	quartos = dados["quartos"].to_numpy()
	preco = dados["preco"].to_numpy()
	r_tamanho, r_quartos = correlacoes
	reg_tamanho = regressao_simples(tamanho, preco)
	reg_quartos = regressao_simples(quartos, preco)

	fig_2d, axes = plt.subplots(1, 2, figsize=(13, 5))
	for axis, variable, label, correlation, regression in [
		(axes[0], tamanho, "Tamanho da casa", r_tamanho, reg_tamanho),
		(axes[1], quartos, "Numero de quartos", r_quartos, reg_quartos),
	]:
		axis.scatter(variable, preco, color="royalblue", label="Dados")
		line_x = np.linspace(variable.min(), variable.max(), 100)
		line_y = regression[0] + regression[1] * line_x
		axis.plot(line_x, line_y, color="crimson", label="Regressao linear simples")
		axis.set_xlabel(label)
		axis.set_ylabel("Preco")
		axis.set_title(f"Correlacao com o preco: r = {correlation:.4f}")
		axis.grid(alpha=0.25)
		axis.legend()
	fig_2d.tight_layout()

	fig_3d = plt.figure(figsize=(10, 7))
	axis_3d = fig_3d.add_subplot(111, projection="3d")
	axis_3d.scatter(tamanho, quartos, preco, color="royalblue", label="Dados")
	grid_tamanho = np.linspace(tamanho.min(), tamanho.max(), 30)
	grid_quartos = np.linspace(quartos.min(), quartos.max(), 30)
	mesh_tamanho, mesh_quartos = np.meshgrid(grid_tamanho, grid_quartos)
	mesh_preco = beta[0] + beta[1] * mesh_tamanho + beta[2] * mesh_quartos
	axis_3d.plot_surface(mesh_tamanho, mesh_quartos, mesh_preco,
						 alpha=0.35, cmap="viridis")
	axis_3d.set_xlabel("Tamanho da casa")
	axis_3d.set_ylabel("Numero de quartos")
	axis_3d.set_zlabel("Preco")
	axis_3d.set_title("Regressao linear multipla")
	axis_3d.text2D(
		0.03,
		0.95,
		"Correlacoes com o preco\n"
		f"Tamanho da casa: r = {r_tamanho:.4f}\n"
		f"Numero de quartos: r = {r_quartos:.4f}",
		transform=axis_3d.transAxes,
		va="top",
		fontsize=10,
		bbox={
			"boxstyle": "round,pad=0.5",
			"facecolor": "white",
			"edgecolor": "slategray",
			"alpha": 0.9,
		},
	)
	axis_3d.legend()
	axis_3d.view_init(elev=25, azim=-55)

	if save_dir:
		output = Path(save_dir)
		output.mkdir(parents=True, exist_ok=True)
		fig_2d.savefig(output / "regressoes_2d.png", dpi=150, bbox_inches="tight")
		fig_3d.savefig(output / "regressao_multipla_3d.png", dpi=150, bbox_inches="tight")
	return fig_2d, fig_3d


def analisar(dados: pd.DataFrame, mostrar_graficos: bool = True,
			 save_dir: str | None = None) -> np.ndarray:
	"""Imprime a analise, estima beta e compara com scikit-learn."""
	print("\nEstatistica descritiva (.describe()):")
	print(dados.describe())
	casa_mais_barata = dados.loc[dados["preco"].idxmin()]
	casa_mais_cara = dados.loc[dados["preco"].idxmax()]
	print(f"\nMedia de preco: {dados['preco'].mean():.2f}")
	print(f"Menor preco: {casa_mais_barata['preco']:.2f}")
	print(f"Quartos na casa mais cara: {int(casa_mais_cara['quartos'])}")

	variaveis = dados[["tamanho", "quartos"]].to_numpy()
	X = np.column_stack((np.ones(len(dados)), variaveis))
	y = dados["preco"].to_numpy()
	beta = regmultipla(X, y)
	correlacoes = (dados["tamanho"].corr(dados["preco"]),
				   dados["quartos"].corr(dados["preco"]))
	print("\nCoeficientes beta [intercepto, tamanho, quartos]:")
	print(beta)
	print(f"Correlacao tamanho/preco: {correlacoes[0]:.6f}")
	print(f"Correlacao quartos/preco: {correlacoes[1]:.6f}")
	reg_tamanho = regressao_simples(dados["tamanho"], y)
	reg_quartos = regressao_simples(dados["quartos"], y)
	print(f"Regressao simples tamanho/preco: preco = {reg_tamanho[0]:.2f} + "
	      f"{reg_tamanho[1]:.2f} * tamanho")
	print(f"Regressao simples quartos/preco: preco = {reg_quartos[0]:.2f} + "
	      f"{reg_quartos[1]:.2f} * quartos")

	casa_teste = np.array([1.0, 1650.0, 3.0])
	print(f"\nPreco estimado para 1650 e 3 quartos: {casa_teste @ beta:.2f}")
	for quantidade in (2, 4, 5):
		print(f"Preco estimado para 1650 e {quantidade} quartos: "
			  f"{np.array([1.0, 1650.0, quantidade]) @ beta:.2f}")
	print("A variacao e beta[2] por quarto, mantendo o tamanho constante.")

	try:
		from sklearn.linear_model import LinearRegression
	except ImportError:
		print("\nComparacao sklearn: indisponivel. Instale scikit-learn para comparar.")
	else:
		sklearn_model = LinearRegression().fit(variaveis, y)
		sklearn_beta = np.r_[sklearn_model.intercept_, sklearn_model.coef_]
		print(f"Comparacao sklearn: {sklearn_beta}")
		print(f"Diferenca maxima: {np.max(np.abs(beta - sklearn_beta)):.12g}")

	criar_graficos(dados, beta, correlacoes, save_dir)
	if mostrar_graficos:
		plt.show()
	else:
		plt.close("all")
	return beta


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--data", help="caminho para data.csv ou data.mat")
	parser.add_argument("--url", help="URL de um data.csv ou data.mat")
	parser.add_argument("--save-dir", help="diretorio para salvar os graficos")
	parser.add_argument("--no-show", action="store_true", help="nao abrir as janelas dos graficos")
	args = parser.parse_args()
	if args.data and args.url:
		parser.error("use --data ou --url, nao os dois")
	dados = carregar_dados(args.data, args.url)
	analisar(dados, mostrar_graficos=not args.no_show, save_dir=args.save_dir)


if __name__ == "__main__":
	main()

# A) Qual arquivo foi utilizado e qual e o formato dos dados?
# R Foi utilizado data (1).csv. Cada linha possui, nesta ordem:
# tamanho da casa, numero de quartos e preco. 

# B) Qual a analise estatistica inicial da base?
# O dataset possui 47 casas. A media dos precos e 340412.77, o menor
# preco e 169900.00 e a casa mais cara possui 5 quartos.

# C) Como foram construidos X e y?
# X possui uma coluna de 1 para o intercepto, uma coluna para o
# tamanho e uma coluna para os quartos: X = [1, tamanho, quartos]. O vetor y
# possui os precos das casas.

# D) Quais sao as correlacoes e regressoes simples?
# A correlacao tamanho/preco e 0.854988 e a correlacao quartos/preco
# e 0.442262. As regressões simples sao:
# preco = 71270.57 + 134.53 * tamanho
# preco = 110034.27 + 72669.73 * quartos
# Os graficos 2D exibem essas regressões simples.

# E) Como foi calculada a regressao multipla?
# A funcao regmultipla(X, y) calcula os coeficientes por minimos
# quadrados. O resultado e beta = [89597.765961, 139.21063465,
# -8737.91542019], na ordem [intercepto, tamanho, quartos].

# F) Como a regressao foi apresentada em tres dimensoes?
# O grafico 3D mostra os pontos reais e o plano de regressao. A
# equacao nao aparece no titulo para manter o grafico limpo; ela e impressa no
# terminal.

# G) Onde aparecem os coeficientes de correlacao?
# Os valores r(tamanho, preco) = 0.854988 e r(quartos, preco) =
# 0.442262 aparecem no quadro do grafico 3D e tambem no terminal.

# H) Qual o preco estimado para tamanho 1650 e 3 quartos? O que
# acontece ao variar o numero de quartos?
# O preco estimado para 1650 e 3 quartos e 293081.57. Mantendo o
# tamanho em 1650, as previsoes para 2, 4 e 5 quartos sao 301819.48,
# 284343.65 e 275605.74. Nesta amostra, beta_2 = -8737.92; portanto, o modelo
# associa mais quartos a menor preco quando o tamanho e mantido constante.
# Isso e uma associacao estatistica condicionada ao tamanho, nao uma regra
# causal sobre o valor real das casas.

# I) O resultado foi comparado com uma biblioteca Python?
# Sim. A implementacao foi comparada com
# sklearn.linear_model.LinearRegression. Os coeficientes coincidem, com
# diferenca numerica maxima de aproximadamente 1.46e-11. Isso significa que
# a maior diferenca absoluta entre os coeficientes dos dois metodos foi
# 0.0000000000146, um valor menor que 1e-10 (0.0000000001). Portanto, os
# resultados podem ser considerados iguais na precisao numerica utilizada.