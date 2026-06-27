# Roteiro de Slides — Quantum Commerce: Análise de Sentimento em Reviews

> Roteiro para a apresentação. Cada bloco é um slide: título, tópicos para o
> slide e notas do apresentador (o que falar). Sugestão: 10–12 slides, ~10 min.

---

## Slide 1 — Capa
- **Quantum Commerce — Análise de Sentimento em Reviews**
- AI Foundation and Learning Models — FIAP MBA
- Integrantes / data

*Notas:* apresentar o tema em uma frase: "transformar avaliações de clientes em
um termômetro automático de satisfação".

---

## Slide 2 — O problema
- Quantum Commerce: varejo omnicanal, 12 países, +5 milhões de SKUs
- "Muro da escalabilidade operacional": suporte e leitura de mercado no manual não escalam
- Milhares de reviews por dia — informação valiosa presa no texto

*Notas:* contextualizar o cenário do enunciado; ninguém lê tudo manualmente.

---

## Slide 3 — O desafio escolhido
- Entre os desafios do enunciado: **análise de sentimento em reviews**
- Objetivo: classificar cada avaliação como **positiva** ou **negativa**
- Valor: priorizar suporte, detectar problemas cedo, métrica contínua de satisfação

*Notas:* deixar claro o "para quê" antes do "como".

---

## Slide 4 — Os dados
- Dataset **Amazon Review Polarity** (negativo = 1, positivo = 2), balanceado
- 3,6 M treino / 400 mil teste → usamos **amostra estratificada de 50k / 10k**
- Por quê: demonstrar a solução em CPU, em minutos, sem perder representatividade

*Notas:* justificar a amostra — é decisão de engenharia, não preguiça.

---

## Slide 5 — Caminhos possíveis (comparação)
| Abordagem | Captura | Limitação |
|---|---|---|
| TF-IDF + SVM | importância de termos | ignora a ordem das palavras |
| word2vec + SVM | semântica das palavras | a média perde a ordem |
| **LSTM** | a review como sequência | treino mais lento |
| BERT / Transformers | contexto profundo | caro, exige GPU |

*Notas:* este slide atende ao item 2 do enunciado (comparar alternativas).

---

## Slide 6 — Por que LSTM
- O que diferencia: **ordem das palavras** ("not good" ≠ "good")
- LSTM tem **memória de contexto** — entende negação e construções
- Equilíbrio: mais capacidade que os clássicos, mais barata que o BERT
- É a arquitetura de sentimento estudada nas aulas

*Notas:* a escolha é equilíbrio capacidade × custo × aderência à disciplina.

---

## Slide 7 — Pipeline da solução
1. Amostragem → 2. Pré-processamento → 3. word2vec → 4. Treino LSTM → 5. Avaliação → 6. Inferência
- Vocabulário construído **só com o treino** (sem vazamento)
- Sequências padronizadas em 200 tokens

*Notas:* mostrar que é um fluxo reproduzível, script a script.

---

## Slide 8 — A arquitetura
- `Embedding(200) → LSTM(128, 2 camadas) → Dropout → Linear → Sigmoide`
- Embeddings **inicializados com word2vec** treinado no próprio corpus
- word2vec aprendeu semântica real:
  - `terrible ≈ horrible, awful` · `great ≈ terrific, fantastic`

*Notas:* o word2vec sozinho já valida que o modelo "entende" as palavras.

---

## Slide 9 — Resultados
| Modelo | Acurácia | F1 |
|---|---|---|
| TF-IDF + SVM (baseline) | 0,886 | 0,886 |
| **LSTM (word2vec)** | 0,895 | 0,895 |

- Matriz de confusão da LSTM (figura `figures/matriz_confusao_lstm.png`)

*Notas:* a LSTM (89,5%) supera o baseline SVM (88,6%). Reforçar que o melhor
modelo foi salvo via *checkpoint* na 2ª época — depois disso a validação piora
(overfitting). Mostrar a matriz de confusão (acertos na diagonal).

---

## Slide 10 — Demo: classificando reviews novas
```
[POSITIVO] p=0.99  "This product is amazing, exactly what I needed..."
[NEGATIVO] p=0.01  "Terrible quality, it broke after one day..."
[POSITIVO] p=0.73  "It works fine, nothing special but it does the job."
```
- O modelo devolve **rótulo + probabilidade** para qualquer texto novo

*Notas:* se possível, rodar `python 07_inferencia.py "..."` ao vivo.

---

## Slide 11 — Limitações e riscos
- Treinado em **inglês**; pt-BR exige novos dados/embeddings
- Apenas **binário** (sem neutro); ironia ainda é difícil
- **Mudança de domínio**: re-treinar com dados da Quantum
- Vocabulário fixo → re-treino periódico; atenção a **viés**

*Notas:* honestidade sobre limites fortalece a proposta.

---

## Slide 12 — Impacto e evolução
- Termômetro de satisfação automático e em escala
- Menos churn, resposta mais rápida, insumo para recomendação/catálogo
- Evolução: *fine-tuning* de transformer; multi-classe (1–5 estrelas); análise por aspecto
- **Conclusão:** solução viável, fundamentada e pronta para evoluir

*Notas:* fechar amarrando no valor de negócio do Slide 3.
