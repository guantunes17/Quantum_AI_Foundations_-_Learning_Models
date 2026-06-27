"""
02_baseline_svm.py
==================
ETAPA 2 - Baseline classico: TF-IDF + SVM linear.

POR QUE UM BASELINE?
--------------------
Antes de partir para a rede neural (LSTM), criamos um modelo SIMPLES e rapido
para ter um "numero de referencia". Se a LSTM nao superar (ou ao menos empatar
com) este baseline, e sinal de que algo esta errado. Alem disso, esta
comparacao e exatamente o que o enunciado pede no topico 2 (justificar a
arquitetura comparando com alternativas).

TECNICA (vista nas Aulas 2 e 3):
  - TF-IDF  : transforma cada review em um vetor de "importancia de palavras".
  - SVM     : o LinearSVC procura um hiperplano que separe positivo de negativo.

Saidas: acuracia, precisao, recall, F1, matriz de confusao (figura) e um JSON
com as metricas (usado depois na comparacao final e no documento).
"""
import json
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # backend que SALVA a imagem sem abrir janela
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from nltk.corpus import stopwords

from utils import DIR_DADOS, DIR_FIGURAS, garantir_pastas


def main():
    garantir_pastas()
    print("=" * 70)
    print("ETAPA 2 - BASELINE: TF-IDF + SVM LINEAR")
    print("=" * 70)

    # --- 1) Carrega as amostras geradas na etapa 1 ---
    treino = pd.read_csv(DIR_DADOS / "amostra_treino.csv")
    teste = pd.read_csv(DIR_DADOS / "amostra_teste.csv")
    X_treino, y_treino = treino["texto"].astype(str), treino["rotulo"]
    X_teste, y_teste = teste["texto"].astype(str), teste["rotulo"]
    print(f"\n[1/4] Dados carregados: {len(treino)} treino / {len(teste)} teste")

    # --- 2) Vetorizacao TF-IDF ---
    #   stop_words : palavras muito comuns (the, a, is...) que pouco ajudam.
    #   ngram_range=(1,2): considera palavras isoladas E pares (ex.: "not good").
    #   min_df=5   : ignora termos que aparecem em < 5 reviews (provavel ruido).
    #   max_features: limita o vocabulario aos 50 mil termos mais relevantes.
    stops = stopwords.words("english")
    vetorizador = TfidfVectorizer(
        stop_words=stops, ngram_range=(1, 2), min_df=5, max_features=50_000
    )
    print("\n[2/4] Ajustando o TF-IDF no TREINO e transformando os textos...")
    Xtr = vetorizador.fit_transform(X_treino)   # aprende o vocabulario no treino
    Xte = vetorizador.transform(X_teste)        # aplica o MESMO vocabulario no teste
    print(f"   Vocabulario TF-IDF: {len(vetorizador.vocabulary_)} termos")
    print(f"   Matriz de treino (esparsa): {Xtr.shape}")

    # --- 3) Treina o SVM linear ---
    print("\n[3/4] Treinando o SVM linear (LinearSVC)...")
    modelo = LinearSVC(C=1.0)                    # C controla a regularizacao
    modelo.fit(Xtr, y_treino)

    # --- 4) Avalia no conjunto de TESTE ---
    print("\n[4/4] Avaliando no conjunto de TESTE...")
    y_pred = modelo.predict(Xte)
    acc = accuracy_score(y_teste, y_pred)
    print(f"\n   >>> Acuracia: {acc:.4f}\n")
    print("   Relatorio de classificacao:")
    print(classification_report(y_teste, y_pred, target_names=["negativo", "positivo"]))

    # Matriz de confusao -> figura
    cm = confusion_matrix(y_teste, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["neg", "pos"], yticklabels=["neg", "pos"])
    plt.xlabel("Previsto"); plt.ylabel("Verdadeiro")
    plt.title(f"Matriz de Confusao - SVM (acc={acc:.3f})")
    plt.tight_layout()
    caminho_fig = DIR_FIGURAS / "matriz_confusao_svm.png"
    plt.savefig(caminho_fig, dpi=120)
    plt.close()
    print(f"   Matriz de confusao salva em: {caminho_fig}")

    # Salva as metricas para o documento e para a comparacao final (etapa 06)
    rel = classification_report(
        y_teste, y_pred, target_names=["negativo", "positivo"], output_dict=True
    )
    metricas = {
        "modelo": "TF-IDF + LinearSVC",
        "acuracia": round(acc, 4),
        "precisao_macro": round(rel["macro avg"]["precision"], 4),
        "recall_macro": round(rel["macro avg"]["recall"], 4),
        "f1_macro": round(rel["macro avg"]["f1-score"], 4),
    }
    with open(DIR_DADOS / "metricas_baseline.json", "w", encoding="utf-8") as f:
        json.dump(metricas, f, indent=2, ensure_ascii=False)
    print(f"   Metricas salvas em: {DIR_DADOS / 'metricas_baseline.json'}")
    print("\n   >>> Proximo passo:  python 03_preprocessamento.py")


if __name__ == "__main__":
    main()
