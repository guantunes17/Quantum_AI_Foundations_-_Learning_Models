# Análise de Sentimento em Reviews — Quantum Commerce

Projeto final da disciplina **AI Foundation and Learning Models** (FIAP MBA).

Classificação automática de avaliações de clientes em **positivas** ou **negativas**,
usando uma rede neural recorrente **LSTM** (PyTorch) com embeddings **word2vec**, e
um baseline clássico **TF-IDF + SVM** (scikit-learn) como referência de comparação.

Base de dados: *Amazon Review Polarity* (reviews da Amazon rotuladas por polaridade).

## Requisitos

- Python 3.12
- O dataset bruto em `../dataset/amazon_review_polarity_csv/` (`train.csv`, `test.csv`).
  Ele não é versionado por causa do tamanho (~1,5 GB).

## Instalação

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m nltk.downloader stopwords punkt punkt_tab
```

O `wandb` (Weights & Biases) já está no `requirements.txt` — é usado para rastrear os
experimentos de treino/avaliação (veja a seção [Rastreamento de experimentos com W&B](#rastreamento-de-experimentos-com-wb)).
Antes de treinar, faça login uma única vez:

```bash
wandb login
```

## Como rodar

Os scripts são numerados e devem ser executados em ordem. Cada um imprime o que
está fazendo e salva seus resultados em `data/`, `models/` e `figures/`.

```bash
python 01_amostragem.py        # cria uma amostra balanceada (50k treino / 10k teste)
python 02_baseline_svm.py      # baseline TF-IDF + SVM (referência)
python 03_preprocessamento.py  # limpeza, vocabulário e padding para a LSTM
python 04_word2vec.py          # treina embeddings word2vec
python 05_treino_lstm.py       # treina a LSTM (alguns minutos em CPU)
python 06_avaliacao.py         # métricas + matriz de confusão + comparação
python 07_inferencia.py        # classifica frases novas
```

### Usar o modelo já treinado (sem re-treinar)

O modelo treinado (`models/lstm.pt`) e os arquivos que a inferência precisa
(`data/vocab_to_int.json`, `data/meta_lstm.json`) acompanham o repositório. Com o
ambiente instalado, dá para classificar reviews **sem rodar o treino**:

```bash
python 07_inferencia.py "this product is awful, do not buy it"
python 07_inferencia.py -i      # modo interativo: digite reviews uma a uma
```

## O que cada etapa faz

| Script | Etapa |
|--------|-------|
| `01_amostragem.py` | Amostra estratificada (50/50) do dataset completo. |
| `02_baseline_svm.py` | TF-IDF + SVM linear; gera métricas e matriz de confusão. |
| `03_preprocessamento.py` | Tokeniza, monta o vocabulário (só do treino) e aplica padding. |
| `04_word2vec.py` | Treina o word2vec e monta a matriz de embedding da LSTM. |
| `05_treino_lstm.py` | Define e treina a rede LSTM. |
| `06_avaliacao.py` | Avalia no teste e compara com o baseline. |
| `07_inferencia.py` | Classifica avaliações novas. |

Arquivos auxiliares: `utils.py` (funções comuns) e `modelo_lstm.py` (definição da rede).

## Rastreamento de experimentos com W&B

Os scripts `05_treino_lstm.py` (treino) e `06_avaliacao.py` (avaliação) registram
métricas automaticamente no [Weights & Biases](https://wandb.ai), no projeto
`quantum-commerce-sentimento`. Não há flag para desligar — rodar o treino sempre
gera uma run (por isso o `wandb login` prévio é necessário).

No dashboard fica visível:

- curvas de perda de treino vs. validação por época (e perda de treino por lote,
  para uma visão mais granular);
- acurácia de validação por época;
- métricas finais no conjunto de teste (acurácia, precisão, recall e F1 macro);
- a matriz de confusão da LSTM;
- uma tabela comparando a LSTM com o baseline TF-IDF + SVM.

As curvas de treino vs. validação deixam visível o overfitting a partir da 2ª
época — comportamento que este projeto já documenta nos comentários do treino.

## Estrutura

```
project/
├── 01_amostragem.py ... 07_inferencia.py
├── utils.py
├── modelo_lstm.py
├── requirements.txt
├── amostra_minima/      # amostra pequena versionada (para testes rápidos)
├── data/                # amostras e arrays gerados (não versionado)
├── models/              # modelos treinados (não versionado)
├── figures/             # gráficos gerados (não versionado)
└── docs/                # documento e roteiro de slides
```
