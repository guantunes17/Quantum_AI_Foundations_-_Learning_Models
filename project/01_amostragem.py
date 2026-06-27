"""
01_amostragem.py
================
ETAPA 1 - Amostragem estratificada do dataset Amazon Review Polarity.

CONTEXTO
--------
O dataset completo tem 3,6 MILHOES de reviews de treino (1,5 GB) e 400 mil de
teste. Treinar nisso em CPU levaria horas. Como o objetivo do trabalho e
DEMONSTRAR a tecnica (e nao bater recorde), tiramos uma AMOSTRA menor e
BALANCEADA (metade positiva, metade negativa).

O que este script faz:
  1) le os CSVs brutos em "pedacos" (chunks) para nao estourar a memoria;
  2) coleta 25.000 reviews de cada classe no treino e 5.000 de cada no teste;
  3) cria a coluna `rotulo` (0 = negativo, 1 = positivo) e junta titulo+texto;
  4) embaralha e salva as amostras em data/;
  5) salva tambem uma "amostra minima" versionavel em amostra_minima/.

Estrutura do CSV bruto: 3 colunas, SEM cabecalho ->  classe, titulo, texto
  classe 1 = NEGATIVO  |  classe 2 = POSITIVO
"""
from pathlib import Path
import pandas as pd

from utils import DIR_DADOS, garantir_pastas, CSV_TREINO_BRUTO, CSV_TESTE_BRUTO

# --- Parametros (mude aqui se quiser amostras maiores/menores) --------------
ALVO_POR_CLASSE_TREINO = 25_000   # 25k + 25k = 50k de treino
ALVO_POR_CLASSE_TESTE = 5_000     # 5k + 5k = 10k de teste
SEMENTE = 42                      # reprodutibilidade: a mesma amostra sempre
TAMANHO_CHUNK = 100_000           # quantas linhas lemos por vez


def amostrar(caminho_csv: Path, alvo_por_classe: int) -> pd.DataFrame:
    """
    Le o CSV bruto em chunks e coleta `alvo_por_classe` reviews de CADA classe.
    Acumulamos ate completar a cota das duas classes e entao paramos - assim
    nao precisamos ler o arquivo inteiro nem carregar 1,5 GB na memoria.
    """
    leitor = pd.read_csv(
        caminho_csv,
        header=None,                         # o arquivo nao tem cabecalho
        names=["classe", "titulo", "texto"],
        dtype=str,                           # lemos tudo como texto...
        keep_default_na=False,               # ...e nao tratamos vazio como NaN
        chunksize=TAMANHO_CHUNK,
    )
    partes = {1: [], 2: []}
    contagem = {1: 0, 2: 0}
    for chunk in leitor:
        chunk["classe"] = chunk["classe"].astype(int)
        for c in (1, 2):
            if contagem[c] < alvo_por_classe:
                falta = alvo_por_classe - contagem[c]
                sub = chunk[chunk["classe"] == c].head(falta)
                partes[c].append(sub)
                contagem[c] += len(sub)
        print(f"   ... chunk lido | negativos={contagem[1]:>6} positivos={contagem[2]:>6}")
        if contagem[1] >= alvo_por_classe and contagem[2] >= alvo_por_classe:
            break
    return pd.concat(partes[1] + partes[2], ignore_index=True)


def preparar(df: pd.DataFrame) -> pd.DataFrame:
    """Cria `rotulo` (0/1) e `texto` (titulo + corpo); descarta o resto."""
    # classe 1 = negativo -> 0 ; classe 2 = positivo -> 1
    rotulo = (df["classe"] == 2).astype(int)
    # juntamos titulo + texto (ambos carregam sentimento) e trocamos o
    # "\n" literal que o dataset usa para quebra de linha por um espaco.
    texto = (df["titulo"].fillna("") + " " + df["texto"].fillna(""))
    texto = texto.str.replace("\\n", " ", regex=False)
    return pd.DataFrame({"rotulo": rotulo, "texto": texto})


def main():
    garantir_pastas()
    print("=" * 70)
    print("ETAPA 1 - AMOSTRAGEM ESTRATIFICADA")
    print("=" * 70)

    print("\n[1/3] Amostrando o TREINO (alvo: %d por classe)..." % ALVO_POR_CLASSE_TREINO)
    treino = preparar(amostrar(CSV_TREINO_BRUTO, ALVO_POR_CLASSE_TREINO))
    print("\n[2/3] Amostrando o TESTE (alvo: %d por classe)..." % ALVO_POR_CLASSE_TESTE)
    teste = preparar(amostrar(CSV_TESTE_BRUTO, ALVO_POR_CLASSE_TESTE))

    # Embaralhamos: senao ficaria tudo-negativo e depois tudo-positivo, o que
    # atrapalha o treino e a divisao treino/validacao mais adiante.
    treino = treino.sample(frac=1, random_state=SEMENTE).reset_index(drop=True)
    teste = teste.sample(frac=1, random_state=SEMENTE).reset_index(drop=True)

    # Amostras de trabalho -> data/ (ignorado pelo git, pois e regeneravel)
    treino.to_csv(DIR_DADOS / "amostra_treino.csv", index=False)
    teste.to_csv(DIR_DADOS / "amostra_teste.csv", index=False)

    # Amostra MINIMA versionavel (para alguem rodar sem baixar o dataset gigante)
    dir_min = Path(__file__).resolve().parent / "amostra_minima"
    dir_min.mkdir(exist_ok=True)
    treino.head(1500).to_csv(dir_min / "amostra_treino_min.csv", index=False)
    teste.head(500).to_csv(dir_min / "amostra_teste_min.csv", index=False)

    print("\n[3/3] Pronto!")
    print(f"   Treino: {len(treino)} linhas (positivos={int(treino.rotulo.sum())}, "
          f"negativos={int((treino.rotulo == 0).sum())})")
    print(f"   Teste : {len(teste)} linhas (positivos={int(teste.rotulo.sum())}, "
          f"negativos={int((teste.rotulo == 0).sum())})")
    print(f"   Arquivos salvos em: {DIR_DADOS}")
    print("\n   Exemplo de uma review da amostra:")
    print(f"     rotulo={treino.iloc[0]['rotulo']} | texto={treino.iloc[0]['texto'][:150]!r} ...")
    print("\n   >>> Proximo passo:  python 02_baseline_svm.py")


if __name__ == "__main__":
    main()
