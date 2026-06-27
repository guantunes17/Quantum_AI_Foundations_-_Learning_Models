"""
07_inferencia.py
================
ETAPA 7 - Usar o modelo treinado para classificar frases NOVAS.

E aqui que se ve "como o resultado seria consumido" no dia a dia da empresa:
chega uma avaliacao nova, o modelo responde a probabilidade de ser positiva.

Uso:
  python 07_inferencia.py                 # classifica algumas frases de exemplo
  python 07_inferencia.py "texto aqui"    # classifica uma frase
  python 07_inferencia.py -i              # modo interativo: digite reviews uma a uma

Obs.: o modelo foi treinado em reviews em INGLES, entao teste com frases em ingles.
"""
import json
import sys

import numpy as np
import torch

from utils import DIR_DADOS, DIR_MODELOS, tokenizar, pad_features
from modelo_lstm import SentimentRNN

# Algumas frases de exemplo (positivas, negativas e uma "neutra" dificil)
FRASES_EXEMPLO = [
    "This product is amazing, exactly what I needed. Highly recommend!",
    "Terrible quality, it broke after one day. Complete waste of money.",
    "Absolutely love it, best purchase I have made this year.",
    "Worst customer service ever, I will never buy here again.",
    "It works fine, nothing special but it does the job.",
]


def classificar(modelo, vocab_to_int, seq_length, frase):
    """Recebe uma frase e devolve (rotulo, probabilidade)."""
    tokens = tokenizar(frase)
    inteiros = [[vocab_to_int.get(w, 0) for w in tokens]]   # palavra desconhecida -> 0
    x = pad_features(inteiros, seq_length)
    with torch.no_grad():
        prob = modelo(torch.from_numpy(x)).item()
    rotulo = "POSITIVO" if prob >= 0.5 else "NEGATIVO"
    return rotulo, prob


def main():
    print("=" * 70)
    print("ETAPA 7 - INFERENCIA (CLASSIFICAR FRASES NOVAS)")
    print("=" * 70)

    with open(DIR_DADOS / "vocab_to_int.json", encoding="utf-8") as f:
        vocab_to_int = json.load(f)
    with open(DIR_DADOS / "meta_lstm.json") as f:
        h = json.load(f)

    modelo = SentimentRNN(
        h["vocab_size"], h["embedding_dim"], h["hidden_dim"],
        h["n_layers"], drop_prob=h["drop_prob"],
    )
    modelo.load_state_dict(torch.load(DIR_MODELOS / "lstm.pt", weights_only=True))
    modelo.eval()

    args = sys.argv[1:]

    # Modo interativo: voce digita reviews uma a uma e ve a previsao na hora.
    if args and args[0] in ("-i", "--interativo"):
        print("\nModo interativo - digite uma review em ingles e tecle Enter.")
        print("(uma linha vazia encerra)\n")
        while True:
            try:
                frase = input("review> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not frase:
                break
            rotulo, prob = classificar(modelo, vocab_to_int, h["seq_length"], frase)
            print(f"   [{rotulo}]  p(positivo)={prob:0.3f}\n")
        return

    # Sem '-i': classifica a frase passada por argumento, ou as de exemplo.
    frases = [" ".join(args)] if args else FRASES_EXEMPLO
    print()
    for frase in frases:
        rotulo, prob = classificar(modelo, vocab_to_int, h["seq_length"], frase)
        print(f"   [{rotulo}]  p(positivo)={prob:0.3f}")
        print(f"      \"{frase}\"\n")


if __name__ == "__main__":
    main()
