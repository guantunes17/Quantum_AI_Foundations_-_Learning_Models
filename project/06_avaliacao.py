"""
06_avaliacao.py
===============
ETAPA 6 - Avaliar a LSTM no conjunto de TESTE e comparar com o baseline.

O conjunto de teste sao 10.000 reviews que a rede NUNCA viu. E o teste honesto
de generalizacao. Reportamos:
  - acuracia, precisao, recall e F1 (classification_report);
  - matriz de confusao (figura);
  - tabela comparando a LSTM com o baseline TF-IDF+SVM (etapa 2).
"""
import json

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from utils import DIR_DADOS, DIR_MODELOS, DIR_FIGURAS, garantir_pastas
from modelo_lstm import SentimentRNN


def main():
    garantir_pastas()
    print("=" * 70)
    print("ETAPA 6 - AVALIACAO DA LSTM NO TESTE")
    print("=" * 70)

    # --- W&B: run propria para a avaliacao, no mesmo projeto do treino ---
    wandb.init(
        project="quantum-commerce-sentimento",
        job_type="avaliacao",
        tags=["eval"],
    )

    # --- 1) Recria o modelo com os mesmos hiperparametros e carrega os pesos ---
    X_te = np.load(DIR_DADOS / "test_x.npy")
    y_te = np.load(DIR_DADOS / "test_y.npy")
    with open(DIR_DADOS / "meta_lstm.json") as f:
        h = json.load(f)
    modelo = SentimentRNN(
        h["vocab_size"], h["embedding_dim"], h["hidden_dim"],
        h["n_layers"], drop_prob=h["drop_prob"],
    )
    modelo.load_state_dict(torch.load(DIR_MODELOS / "lstm.pt", weights_only=True))
    modelo.eval()
    print(f"\n[1/3] Modelo carregado. Avaliando {len(X_te)} reviews de teste...")

    # --- 2) Previsoes (em lotes para nao pesar a memoria) ---
    preds = []
    with torch.no_grad():
        for i in range(0, len(X_te), 256):
            xb = torch.from_numpy(X_te[i:i + 256])
            preds.append((modelo(xb) >= 0.5).long().numpy())
    y_pred = np.concatenate(preds)

    acc = accuracy_score(y_te, y_pred)
    print(f"\n[2/3] >>> Acuracia da LSTM no teste: {acc:.4f}\n")
    print(classification_report(y_te, y_pred, target_names=["negativo", "positivo"]))

    # Matriz de confusao -> figura
    cm = confusion_matrix(y_te, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",
                xticklabels=["neg", "pos"], yticklabels=["neg", "pos"])
    plt.xlabel("Previsto"); plt.ylabel("Verdadeiro")
    plt.title(f"Matriz de Confusao - LSTM (acc={acc:.3f})")
    plt.tight_layout()
    caminho_fig = DIR_FIGURAS / "matriz_confusao_lstm.png"
    plt.savefig(caminho_fig, dpi=120)
    plt.close()
    print(f"   Matriz de confusao salva em: {caminho_fig}")

    rel = classification_report(
        y_te, y_pred, target_names=["negativo", "positivo"], output_dict=True
    )
    metricas_lstm = {
        "modelo": "LSTM (word2vec)",
        "acuracia": round(acc, 4),
        "precisao_macro": round(rel["macro avg"]["precision"], 4),
        "recall_macro": round(rel["macro avg"]["recall"], 4),
        "f1_macro": round(rel["macro avg"]["f1-score"], 4),
    }
    with open(DIR_DADOS / "metricas_lstm.json", "w", encoding="utf-8") as f:
        json.dump(metricas_lstm, f, indent=2, ensure_ascii=False)

    # --- W&B: metricas finais de teste + matriz de confusao ---
    wandb.log({
        "teste_acuracia": metricas_lstm["acuracia"],
        "teste_precisao_macro": metricas_lstm["precisao_macro"],
        "teste_recall_macro": metricas_lstm["recall_macro"],
        "teste_f1_macro": metricas_lstm["f1_macro"],
    })
    try:
        # Grafico nativo do W&B: interativo, com contagens por celula no dashboard.
        wandb.log({
            "matriz_confusao": wandb.plot.confusion_matrix(
                y_true=y_te.tolist(),
                preds=y_pred.tolist(),
                class_names=["negativo", "positivo"],
            )
        })
    except Exception as e:
        # Fallback: loga a figura PNG que ja foi gerada acima.
        print(f"   (wandb.plot.confusion_matrix falhou: {e}; usando a figura PNG como fallback)")
        wandb.log({"matriz_confusao": wandb.Image(str(caminho_fig))})

    # --- 3) Comparacao com o baseline (etapa 2) ---
    print("\n[3/3] Comparacao final (teste):")
    caminho_base = DIR_DADOS / "metricas_baseline.json"
    linhas = [metricas_lstm]
    if caminho_base.exists():
        with open(caminho_base, encoding="utf-8") as f:
            linhas.insert(0, json.load(f))
    print(f"\n   {'modelo':<22}{'acuracia':>10}{'precisao':>10}{'recall':>10}{'f1':>8}")
    print("   " + "-" * 60)
    for m in linhas:
        print(f"   {m['modelo']:<22}{m['acuracia']:>10}{m['precisao_macro']:>10}"
              f"{m['recall_macro']:>10}{m['f1_macro']:>8}")
    print("\n   >>> Proximo passo:  python 07_inferencia.py")

    # --- W&B: tabela comparando LSTM vs baseline SVM (se o baseline existir) ---
    tabela_comparacao = wandb.Table(
        columns=["modelo", "acuracia", "precisao_macro", "recall_macro", "f1_macro"],
        data=[[m["modelo"], m["acuracia"], m["precisao_macro"], m["recall_macro"], m["f1_macro"]]
              for m in linhas],
    )
    wandb.log({"comparacao_lstm_vs_baseline": tabela_comparacao})

    wandb.finish()


if __name__ == "__main__":
    main()
