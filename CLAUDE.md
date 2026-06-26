# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repository is

Course materials for the FIAP MBA discipline **"AI Foundation and Learning Models"** (turma 1AIER). It is a collection of teaching **Jupyter notebooks**, PDF lecture slides (`pdf/`), and datasets — not an application or library. There is no build, no test suite, no package, and no dependency manifest. Almost all prose, comments, and variable names are in **Portuguese (pt-BR)**; match that language when editing course content.

## Layout

Each `Aula_N_*` directory is a self-contained lesson. The course progresses from classical ML to deep learning to transformers/RL:

- `Aula_2_-_SVM_e_Regressao_Linear/` — SVM and linear regression with **scikit-learn**. Data in `bases/` (`profit.txt`, `Lemonades.csv`, `consumo_cerveja.csv`).
- `Aula_3_-_Introducao_a_NLP/` — NLP with **NLTK / spaCy / sklearn** (TF-IDF, word2vec, sentiment RNN), plus CNN/MNIST and OpenCV edge-detection notebooks. MNIST data under `data/MNIST/raw/`.
- `Aula_5/` — Neural-network fundamentals (`SimpleNN`, `mlp_notebook`, `activation_functions`). `planar_utils.py` is a local helper (decision-boundary plotting, planar dataset) imported by these notebooks. `mnist.pkl.gz` is the local MNIST pickle.
- `Aula_6/` — CNNs & RNNs in **PyTorch** (MNIST MLP, conv/maxpool visualization, CIFAR-10 with augmentation, sentiment RNN, OpenCV kernels).
- `Aula_7_-_Attention_e_RL/` — Attention demos (FinBERT over `manchetes_finbert.csv`, stock data `ohlcv_ativo.csv`) and a from-scratch RL demo. Uses **HuggingFace transformers**.
- `Aula_8_-_Transfer_Learning_e_Fine_Tuning/` — Transfer learning / fine-tuning with **transformers + PyTorch**.
- `pdf/` — Lecture slides (`Aula_0`–`Aula_7`) and the final-project brief (`Enunciado_TrabalhoFinal_AIE.pdf`).
- `dataset/` — Amazon Review Polarity dataset (see warning below).

## Tech stack

All deep learning is **PyTorch** (`torch`, `torchvision`) — including the CIFAR-10 notebook (it is *not* Keras, despite the tutorial lineage). Classical ML is **scikit-learn**. NLP uses **NLTK**, **spaCy**, and **HuggingFace transformers**. Image work uses **OpenCV** (`cv2`). Standard scientific stack throughout: `numpy`, `pandas`, `matplotlib`, `seaborn`.

There is no pinned environment. To set one up:

```bash
pip install jupyterlab numpy pandas matplotlib seaborn scikit-learn \
    torch torchvision nltk spacy transformers opencv-python
```

## Running notebooks

```bash
jupyter lab                 # or: jupyter notebook

# Execute one notebook headlessly (the closest thing to "run a single test"):
jupyter nbconvert --to notebook --execute --inplace "Aula_5/mlp_notebook.ipynb"
```

**Run notebooks with the working directory set to their own lesson folder.** They reference sibling data via relative paths (e.g. `bases/profit.txt`, `data/MNIST/`, `imagens/`), so launching Jupyter from the repo root and opening a notebook works, but executing one from a different CWD will fail to find its data.

## Conventions and gotchas

- **Filename suffixes carry meaning:** `Exercicio_*` = student exercise (often with blanks to fill); `*_corr` = *correção*, the filled-in solution/answer key; `Demo_*` = instructor demo. When asked to "solve" an exercise, the `_corr` sibling is the reference answer.
- **Duplicated notebooks:** `Aula_3` and `Aula_6` share several files; some are byte-identical (`2-conv_visualization.ipynb`, `Sentiment_RNN_Solution.ipynb`, `word2vec.ipynb`), while `1-mnist_mlp_solution_with_validation.ipynb` differs between the two. Edit the copy in the lesson you actually mean; they are not symlinked.
- **Git is mid-rename.** Tracked paths still use the old space-separated names (e.g. `Aula 2 - SVM e Regressao Linear`), while the working tree uses underscores (`Aula_2_-_SVM_e_Regressao_Linear`). `git status` is therefore dominated by deletions (old paths) plus untracked files (new paths) — this is a pending directory rename, not lost work. The underscore directories on disk are current. History is a series of "Add files via upload" commits (GitHub web uploads).
- **Datasets are not in git.** `dataset/` holds the Amazon Review Polarity dataset (`train.csv` ~1.5 GB, `test.csv` ~168 MB, plus ~657 MB archives), which exceeds GitHub's 100 MB limit and is excluded by `.gitignore` (`/dataset/*`, keeping only `dataset/README.md`). Download it from [Kaggle](https://www.kaggle.com/datasets/kritanjalijain/amazon-reviews) and extract into `dataset/` per `dataset/README.md`. (`Aula_5/mnist.pkl.gz` and the MNIST `raw/*ubyte` files are small and tracked.)

## Final project — "Quantum Commerce"

`pdf/Enunciado_TrabalhoFinal_AIE.pdf` defines the capstone: propose and justify an ML/DL architecture for a real omnichannel-retail problem (product classification/scoring, fraud/anomaly detection, review sentiment, catalog computer vision, or generative content enrichment). Deliverable is a slide deck or written document (optionally a demo notebook) covering problem, architecture justification vs. alternatives, solution/pipeline, and impact/limitations. Due **2026-07-12**.
