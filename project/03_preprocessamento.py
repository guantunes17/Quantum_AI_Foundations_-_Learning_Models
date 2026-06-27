"""
03_preprocessamento.py
======================
ETAPA 3 - Preparar o texto para a rede neural (LSTM).

A LSTM nao entende palavras, so numeros. Entao precisamos:
  1) tokenizar     -> quebrar cada review em palavras (limpando pontuacao);
  2) vocabulario   -> dar um numero inteiro para cada palavra (so do TREINO,
                      para nao "vazar" informacao do teste);
  3) codificar     -> trocar cada review pela lista de inteiros das suas palavras;
  4) padding       -> deixar todas as sequencias com o mesmo tamanho (200);
  5) split         -> separar 10% do treino para VALIDACAO.

Tudo e salvo em data/ como arrays .npy, prontos para a etapa 5.
"""
import json
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from utils import DIR_DADOS, garantir_pastas, tokenizar, pad_features

MIN_FREQ = 2        # palavra precisa aparecer >= 2x no treino para entrar no vocab
SEQ_LENGTH = 200    # tamanho fixo das sequencias (igual a Aula 3)
FRACAO_VAL = 0.10   # 10% do treino viram validacao
SEMENTE = 42


def construir_vocabulario(tokens_treino):
    """Mapeia palavra -> inteiro. Indices comecam em 1 (0 = padding/desconhecida)."""
    contador = Counter()
    for toks in tokens_treino:
        contador.update(toks)
    # mantemos so palavras frequentes, ordenadas da mais comum para a menos comum
    palavras = [p for p, c in contador.most_common() if c >= MIN_FREQ]
    return {p: i for i, p in enumerate(palavras, start=1)}


def codificar(lista_de_tokens, vocab_to_int):
    """Troca cada palavra pelo seu inteiro; desconhecida vira 0."""
    return [[vocab_to_int.get(w, 0) for w in toks] for toks in lista_de_tokens]


def main():
    garantir_pastas()
    print("=" * 70)
    print("ETAPA 3 - PRE-PROCESSAMENTO PARA A LSTM")
    print("=" * 70)

    treino = pd.read_csv(DIR_DADOS / "amostra_treino.csv", keep_default_na=False)
    teste = pd.read_csv(DIR_DADOS / "amostra_teste.csv", keep_default_na=False)

    print("\n[1/5] Tokenizando os textos...")
    tok_treino = [tokenizar(t) for t in treino["texto"]]
    tok_teste = [tokenizar(t) for t in teste["texto"]]
    media = np.mean([len(t) for t in tok_treino])
    print(f"   Comprimento medio das reviews (em palavras): {media:.1f}")

    print("\n[2/5] Construindo o vocabulario (apenas com o TREINO)...")
    vocab_to_int = construir_vocabulario(tok_treino)
    print(f"   Tamanho do vocabulario: {len(vocab_to_int)} palavras")

    print("\n[3/5] Codificando as reviews em sequencias de inteiros...")
    int_treino = codificar(tok_treino, vocab_to_int)
    int_teste = codificar(tok_teste, vocab_to_int)
    print(f"   Exemplo: {tok_treino[0][:8]} -> {int_treino[0][:8]}")

    print(f"\n[4/5] Padding para seq_length={SEQ_LENGTH} e split treino/validacao...")
    X_treino_full = pad_features(int_treino, SEQ_LENGTH)
    X_teste = pad_features(int_teste, SEQ_LENGTH)
    y_treino_full = treino["rotulo"].to_numpy()
    y_teste = teste["rotulo"].to_numpy()
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_treino_full, y_treino_full,
        test_size=FRACAO_VAL, random_state=SEMENTE, stratify=y_treino_full,
    )

    print("\n[5/5] Salvando arrays em data/...")
    np.save(DIR_DADOS / "train_x.npy", X_tr)
    np.save(DIR_DADOS / "train_y.npy", y_tr)
    np.save(DIR_DADOS / "val_x.npy", X_val)
    np.save(DIR_DADOS / "val_y.npy", y_val)
    np.save(DIR_DADOS / "test_x.npy", X_teste)
    np.save(DIR_DADOS / "test_y.npy", y_teste)
    with open(DIR_DADOS / "vocab_to_int.json", "w", encoding="utf-8") as f:
        json.dump(vocab_to_int, f, ensure_ascii=False)
    meta = {"seq_length": SEQ_LENGTH, "vocab_size": len(vocab_to_int) + 1, "min_freq": MIN_FREQ}
    with open(DIR_DADOS / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"   train_x={X_tr.shape}  val_x={X_val.shape}  test_x={X_teste.shape}")
    print(f"   vocab_size (com o 0 de padding) = {meta['vocab_size']}")
    print("\n   >>> Proximo passo:  python 04_word2vec.py")


if __name__ == "__main__":
    main()
