"""
04_word2vec.py
==============
ETAPA 4 - Embeddings de palavras com word2vec (gensim).

O que e um embedding? E um vetor de numeros que representa uma palavra de forma
que palavras com sentido parecido fiquem perto no espaco. O word2vec aprende
isso sozinho, olhando quais palavras aparecem em contextos parecidos.

Aqui nos:
  1) treinamos o word2vec no nosso corpus de treino;
  2) mostramos palavras semelhantes (so para ganhar intuicao);
  3) montamos a MATRIZ DE EMBEDDING alinhada ao vocabulario da etapa 3, para
     INICIALIZAR a camada Embedding da LSTM (em vez de comecar do zero aleatorio).

Visto na Aula 3 (word2vec.ipynb).
"""
import json

import numpy as np
import pandas as pd
from gensim.models import Word2Vec

from utils import DIR_DADOS, DIR_MODELOS, garantir_pastas, tokenizar

DIM = 200          # dimensao dos vetores -> DEVE casar com embedding_dim da LSTM (etapa 5)
JANELA = 5         # quantas palavras de contexto olhar de cada lado
MIN_COUNT = 2      # ignora palavras muito raras
SG = 1             # 1 = skip-gram (bom para corpora menores); 0 = CBOW
EPOCAS = 5
SEMENTE = 42


def main():
    garantir_pastas()
    print("=" * 70)
    print("ETAPA 4 - WORD2VEC (EMBEDDINGS)")
    print("=" * 70)

    treino = pd.read_csv(DIR_DADOS / "amostra_treino.csv", keep_default_na=False)
    frases = [tokenizar(t) for t in treino["texto"]]

    print("\n[1/3] Treinando o word2vec no corpus de treino...")
    modelo = Word2Vec(
        sentences=frases, vector_size=DIM, window=JANELA, min_count=MIN_COUNT,
        sg=SG, epochs=EPOCAS, workers=4, seed=SEMENTE,
    )
    modelo.save(str(DIR_MODELOS / "word2vec.model"))
    print(f"   Vocabulario do word2vec: {len(modelo.wv)} palavras")

    print("\n[2/3] Palavras mais semelhantes (intuicao do que ele aprendeu):")
    for w in ["good", "bad", "great", "terrible", "cheap", "love"]:
        if w in modelo.wv:
            sims = ", ".join(f"{p}({s:.2f})" for p, s in modelo.wv.most_similar(w, topn=5))
            print(f"   {w:9s} ~ {sims}")

    print("\n[3/3] Montando a matriz de embedding alinhada ao vocabulario (etapa 3)...")
    with open(DIR_DADOS / "vocab_to_int.json", encoding="utf-8") as f:
        vocab_to_int = json.load(f)
    vocab_size = len(vocab_to_int) + 1

    # Comecamos com vetores aleatorios pequenos; a linha 0 (padding) fica zerada.
    rng = np.random.default_rng(SEMENTE)
    matriz = rng.normal(0.0, 0.1, size=(vocab_size, DIM)).astype("float32")
    matriz[0] = 0.0

    # Para cada palavra do nosso vocabulario que o word2vec conhece, copiamos o vetor.
    achados = 0
    for palavra, idx in vocab_to_int.items():
        if palavra in modelo.wv:
            matriz[idx] = modelo.wv[palavra]
            achados += 1
    print(f"   Cobertura: {achados}/{len(vocab_to_int)} palavras com vetor word2vec")

    np.save(DIR_DADOS / "embedding_matrix.npy", matriz)
    print(f"   Matriz salva: {matriz.shape} -> data/embedding_matrix.npy")
    print("\n   >>> Proximo passo:  python 05_treino_lstm.py")


if __name__ == "__main__":
    main()
