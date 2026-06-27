"""
modelo_lstm.py
==============
Definicao da REDE NEURAL LSTM de analise de sentimento.

Fica num arquivo separado porque tres etapas usam a mesma rede:
  - 05_treino_lstm.py   (treina)
  - 06_avaliacao.py     (avalia no teste)
  - 07_inferencia.py    (classifica frases novas)

A arquitetura segue a ideia do Sentiment_RNN das Aulas 3/6:
  indices de palavras -> EMBEDDING -> LSTM -> camada linear -> SIGMOIDE -> prob.
"""
import torch
import torch.nn as nn


class SentimentRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, n_layers,
                 output_size=1, drop_prob=0.5, embedding_matrix=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        # 1) EMBEDDING: troca cada indice de palavra por um vetor denso treinavel.
        #    padding_idx=0 mantem a linha do "zero" (padding) sempre nula.
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        if embedding_matrix is not None:
            # Inicializa os vetores com o word2vec (transfer learning leve).
            # Eles continuam treinaveis (vao se ajustando durante o treino).
            self.embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))

        # 2) LSTM: le a sequencia de vetores e "resume" o contexto da review.
        #    batch_first=True -> tensores no formato (batch, sequencia, embedding).
        #    O dropout interno da LSTM so atua ENTRE camadas (n_layers > 1).
        self.lstm = nn.LSTM(
            embedding_dim, hidden_dim, n_layers,
            dropout=drop_prob if n_layers > 1 else 0.0,
            batch_first=True,
        )

        # 3) Regularizacao + camada de saida.
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim, output_size)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = x.long()                      # a Embedding exige inteiros (indices)
        embeds = self.embedding(x)        # (batch, seq) -> (batch, seq, embedding_dim)

        # Deixamos o PyTorch inicializar o estado oculto da LSTM com zeros a cada
        # chamada. Como cada review e INDEPENDENTE, nao faz sentido carregar
        # estado de uma review para outra (e isso evita o bug do ultimo lote
        # menor que o batch_size).
        lstm_out, _ = self.lstm(embeds)   # (batch, seq, hidden_dim)

        ultimo = lstm_out[:, -1, :]       # so a saida do ULTIMO passo de tempo
        out = self.dropout(ultimo)
        out = self.fc(out)                # (batch, 1)
        return self.sig(out).squeeze(-1)  # (batch,) -> probabilidade de ser POSITIVO
