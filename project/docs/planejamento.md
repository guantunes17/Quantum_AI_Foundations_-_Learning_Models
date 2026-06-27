# Planejamento — Trabalho Final
## Análise de Sentimento em Reviews — Quantum Commerce
Disciplina: *AI Foundation and Learning Models* — FIAP MBA (turma 1AIER)

---

## 1. Objetivo e contexto

A Quantum Commerce, varejista omnicanal, recebe um volume enorme de avaliações de clientes e quer transformá-las em um **termômetro de satisfação** automático por produto/categoria. Entre os desafios do enunciado, escolhemos **Análise de sentimento em reviews**: classificar automaticamente cada avaliação como **positiva** ou **negativa**.

Valor para o negócio: priorizar atendimento, identificar produtos problemáticos cedo e medir satisfação em escala — sem leitura manual.

## 2. Dados

- **Amazon Review Polarity** (benchmark público de reviews da Amazon rotuladas por polaridade). Colunas: `classe, título, texto`, com **classe 1 = negativo** e **2 = positivo**. Base **balanceada**.
- Volume total: **3,6M** de treino / **400k** de teste — grande demais para treinar em CPU.
- **Amostra do trabalho:** amostra **estratificada** (50/50 positivo/negativo) de **50.000** linhas de treino e **10.000** de teste. Suficiente para acurácia alta e para demonstrar todas as técnicas, treinando em minutos.
- O dataset bruto fica **fora do Git** (`.gitignore`); versionamos o código e uma amostra mínima (~2k linhas) para reprodução imediata.

## 3. Abordagem e arquitetura

**Modelo principal (código): rede recorrente LSTM** (PyTorch), no mesmo espírito do `Sentiment_RNN_Solution` das Aulas 3/6.

- Pré-processamento: minúsculas → remoção de pontuação → tokenização → **vocabulário (palavra→inteiro) construído só no treino** (evita vazamento) → *padding* para `seq_length = 200`.
- Embeddings inicializados com **word2vec** (gensim) treinado no próprio corpus (Aula 3).
- Arquitetura: `Embedding(vocab, 400) → LSTM(hidden=256, 2 camadas, dropout=0.5) → Dropout(0.3) → Linear(256→1) → Sigmoid`.
- Treino: perda `BCELoss`, otimizador `Adam(lr=1e-3)`, 4 épocas, *batch* 50, *gradient clipping*.

**Comparação com alternativas (vai no documento — tópico 2):**

| Abordagem | Captura | Custo | Papel no trabalho |
|---|---|---|---|
| TF-IDF + SVM linear (Aula 2/3) | frequência de termos; ignora ordem/contexto | baixíssimo | **baseline real** (`02_baseline_svm.py`) |
| word2vec + SVM (Aula 3) | semântica de palavras; sem ordem | baixo | citado |
| **LSTM (escolhida)** | **ordem e contexto sequencial** | médio (ok em CPU c/ amostra) | **modelo principal** |
| Transformers/BERT (Aula 7) | contexto bidirecional profundo | alto | citado como evolução futura |

Justificativa: a LSTM é o melhor **equilíbrio** entre capturar a sequência das palavras (onde o TF-IDF falha) e o custo computacional (onde o BERT pesa), e está dentro do que foi ensinado.

## 4. Estrutura de arquivos e pipeline

```
project/
  README.md                 # visão geral e como rodar
  requirements.txt          # dependências (venv Python 3.12)
  01_amostragem.py          # amostra estratificada 50k/10k a partir de train/test.csv
  02_baseline_svm.py        # TF-IDF + LinearSVC + métricas (baseline clássico)
  03_preprocessamento.py    # limpeza, tokenização, vocabulário, padding -> salva arrays
  04_word2vec.py            # treina word2vec, most_similar, matriz de embedding p/ LSTM
  05_treino_lstm.py         # define e treina a LSTM, salva o modelo
  06_avaliacao.py           # matriz de confusão + accuracy/precision/recall/f1 + heatmap
  07_inferencia.py          # classifica frases novas (demo de uso)
  data/        (gitignored) # amostras e arrays gerados
  models/      (gitignored) # modelos e artefatos treinados
  figures/     (gitignored) # gráficos (matriz de confusão etc.)
  amostra_minima/           # ~2k linhas versionadas p/ rodar na hora
  docs/
    planejamento.md         # este arquivo
    documento.md            # documento principal (4 tópicos do enunciado)
    roteiro_slides.md       # roteiro da apresentação
```

**Ordem de execução:** `01 → 02 → 03 → 04 → 05 → 06 → 07`. Cada script tem comentários didáticos em pt-BR e imprime o que está fazendo, para rodar e entender etapa a etapa (os comentários extras serão removidos numa versão final).

## 5. Mapeamento aula → projeto

- Limpeza, tokenização, vocabulário, TF-IDF — **Aula 3 (NLP)** e **Aula 2 (SVM)**.
- word2vec / embeddings — **Aula 3 (word2vec)**.
- LSTM em PyTorch (embedding + LSTM + sigmoide, `pad_features`, treino com BCELoss/Adam) — **Aula 3/6 (Sentiment_RNN)**.
- Matriz de confusão e métricas (accuracy, precision, recall, f1, `classification_report`) — **Aula 2/3**.

## 6. Ambiente e execução

- Sem GPU; 24 núcleos de CPU, ~30 GB de RAM. Tudo roda em CPU graças à amostra.
- **Ambiente virtual com Python 3.12** (o Python 3.14 do sistema é novo demais para os *wheels* do PyTorch).
- Dependências: `numpy`, `pandas`, `scikit-learn`, `torch`, `nltk`, `gensim`, `matplotlib`, `seaborn`.

## 7. Entregáveis

1. **Documento (principal)** — pt-BR, cobrindo os 4 tópicos: Problema → Justificativa da arquitetura (com comparação) → Solução/pipeline → Impacto e limitações.
2. **Roteiro de slides** — para a apresentação.
3. **Código** — os scripts numerados como prova de que a solução funciona, com métricas e matriz de confusão.

## 8. Riscos e mitigações

- *Treino lento em CPU* → amostra de 50k/10k e modelo enxuto.
- *PyTorch x Python 3.14* → venv com Python 3.12.
- *Artefatos grandes no Git* → `data/`, `models/`, `figures/` no `.gitignore`; só código + amostra mínima versionados.
- *Overfitting* → dropout, conjunto de validação, e teste verdadeiramente separado (vindo do `test.csv`).

## 9. Critérios de sucesso

- LSTM com acurácia esperada na faixa de ~85–90% no teste; baseline SVM como referência.
- Matriz de confusão e métricas (precision/recall/f1) reportadas.
- Demo de inferência classificando frases novas corretamente.
- Documento e slides cobrindo os 4 tópicos com a justificativa da escolha.
