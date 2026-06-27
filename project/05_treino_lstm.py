"""
05_treino_lstm.py
=================
ETAPA 5 - Treinar a rede LSTM.

Aqui acontece o "aprendizado": mostramos as reviews codificadas para a rede,
ela chuta se sao positivas/negativas, medimos o erro (BCELoss) e ajustamos os
pesos (Adam) para errar menos. Repetimos por algumas epocas.

A cada epoca medimos o desempenho na VALIDACAO (dados que a rede nao treina)
para perceber se esta aprendendo de verdade (e nao apenas decorando).

Tudo roda em CPU; por isso a amostra menor e o modelo enxuto.
"""
import json

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from utils import DIR_DADOS, DIR_MODELOS, garantir_pastas
from modelo_lstm import SentimentRNN

# --- Hiperparametros (mude aqui para experimentar) --------------------------
EMBEDDING_DIM = 200    # DEVE casar com a dimensao do word2vec (etapa 4)
HIDDEN_DIM = 128       # tamanho da "memoria" da LSTM
N_LAYERS = 2           # camadas de LSTM empilhadas
DROP_PROB = 0.5        # dropout entre camadas da LSTM (combate overfitting)
BATCH_SIZE = 128       # reviews processadas por vez
EPOCAS = 4             # quantas passadas completas pelos dados de treino
LR = 1e-3              # taxa de aprendizado do Adam
CLIP = 5               # "gradient clipping": evita gradientes explosivos (comum em RNN)
SEMENTE = 42


def carregar(nome):
    return np.load(DIR_DADOS / nome)


def main():
    garantir_pastas()
    torch.manual_seed(SEMENTE)
    print("=" * 70)
    print("ETAPA 5 - TREINANDO A LSTM")
    print("=" * 70)

    # --- 1) Dados (arrays gerados na etapa 3) ---
    X_tr, y_tr = carregar("train_x.npy"), carregar("train_y.npy")
    X_val, y_val = carregar("val_x.npy"), carregar("val_y.npy")
    with open(DIR_DADOS / "meta.json") as f:
        meta = json.load(f)
    vocab_size = meta["vocab_size"]
    seq_length = meta["seq_length"]
    print(f"\n[1/4] Treino={X_tr.shape}  Validacao={X_val.shape}  vocab_size={vocab_size}")

    # --- 2) Matriz de embedding word2vec (etapa 4), se existir ---
    caminho_emb = DIR_DADOS / "embedding_matrix.npy"
    embedding_matrix = None
    if caminho_emb.exists():
        embedding_matrix = np.load(caminho_emb)
        assert embedding_matrix.shape == (vocab_size, EMBEDDING_DIM), (
            f"Embedding {embedding_matrix.shape} != ({vocab_size}, {EMBEDDING_DIM}). "
            "Rode a etapa 4 com o mesmo DIM."
        )
        print(f"[2/4] Embeddings word2vec carregados: {embedding_matrix.shape}")
    else:
        print("[2/4] Sem word2vec; a Embedding comeca aleatoria (rode a etapa 4 para usar).")

    # --- 3) DataLoaders (servem os dados em lotes embaralhados) ---
    train_ds = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    # --- 4) Modelo, funcao de perda e otimizador ---
    modelo = SentimentRNN(
        vocab_size, EMBEDDING_DIM, HIDDEN_DIM, N_LAYERS,
        drop_prob=DROP_PROB, embedding_matrix=embedding_matrix,
    )
    criterio = nn.BCELoss()                                   # erro p/ classificacao binaria
    otimizador = torch.optim.Adam(modelo.parameters(), lr=LR)
    n_params = sum(p.numel() for p in modelo.parameters())
    print(f"[3/4] Modelo criado ({n_params:,} parametros):")
    print(modelo)

    print(f"\n[4/4] Treinando por {EPOCAS} epocas (lotes de {BATCH_SIZE})...\n")
    n_lotes = len(train_loader)
    melhor_val = float("inf")   # menor perda de validacao vista (para o checkpoint)
    melhor_epoca = 0
    for epoca in range(1, EPOCAS + 1):
        modelo.train()
        perdas = []
        for i, (xb, yb) in enumerate(train_loader, 1):
            otimizador.zero_grad()
            saida = modelo(xb)                                # probabilidades (batch,)
            perda = criterio(saida, yb.float())
            perda.backward()                                  # calcula gradientes
            nn.utils.clip_grad_norm_(modelo.parameters(), CLIP)
            otimizador.step()                                 # ajusta os pesos
            perdas.append(perda.item())
            if i % 100 == 0 or i == n_lotes:
                print(f"   epoca {epoca} | lote {i:>4}/{n_lotes} | perda media {np.mean(perdas):.4f}")

        # --- Validacao ao fim da epoca ---
        modelo.eval()
        v_perdas, acertos, total = [], 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                saida = modelo(xb)
                v_perdas.append(criterio(saida, yb.float()).item())
                pred = (saida >= 0.5).long()                  # >=0.5 -> positivo
                acertos += (pred == yb).sum().item()
                total += len(yb)
        perda_val = float(np.mean(v_perdas))
        acc_val = acertos / total
        print(f"   >>> fim epoca {epoca}: perda_val={perda_val:.4f} | acuracia_val={acc_val:.4f}")

        # CHECKPOINT do MELHOR modelo (o de menor perda de validacao).
        # Repare nos logs: a perda de TREINO cai sempre, mas a de VALIDACAO
        # para de cair e volta a subir -> isso e OVERFITTING (a rede comeca a
        # "decorar" o treino em vez de generalizar). Por isso guardamos o melhor
        # momento, e nao a ultima epoca.
        if perda_val < melhor_val:
            melhor_val = perda_val
            melhor_epoca = epoca
            torch.save(modelo.state_dict(), DIR_MODELOS / "lstm.pt")
            print("       (melhor resultado ate agora -> modelo salvo)")
        print()

    # --- Salva os hiperparametros (para 06 e 07 reconstruirem a rede) ---
    hparams = {
        "embedding_dim": EMBEDDING_DIM, "hidden_dim": HIDDEN_DIM,
        "n_layers": N_LAYERS, "drop_prob": DROP_PROB,
        "vocab_size": vocab_size, "seq_length": seq_length,
    }
    with open(DIR_DADOS / "meta_lstm.json", "w", encoding="utf-8") as f:
        json.dump(hparams, f, indent=2)
    print(f"   Melhor modelo: epoca {melhor_epoca} (perda_val={melhor_val:.4f})")
    print(f"   Modelo salvo em: {DIR_MODELOS / 'lstm.pt'}")
    print("\n   >>> Proximo passo:  python 06_avaliacao.py")


if __name__ == "__main__":
    main()
