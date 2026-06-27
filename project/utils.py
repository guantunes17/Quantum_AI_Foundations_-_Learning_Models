"""
utils.py
========
Funcoes e caminhos COMPARTILHADOS por todos os scripts do projeto.

Por que um arquivo de utilidades?
  - Evita repetir o mesmo codigo de limpeza / caminhos em cada etapa.
  - Cada script (01, 02, ...) importa daqui o que precisa e fica focado
    apenas na SUA etapa.

(Os comentarios extensos sao didaticos e serao removidos na versao final.)
"""
from pathlib import Path
import re
import numpy as np

# ----------------------------------------------------------------------
# CAMINHOS DO PROJETO
# Usamos o local DESTE arquivo como raiz. Assim os scripts funcionam
# independentemente da pasta de onde voce chama o `python` (o CWD).
# ----------------------------------------------------------------------
RAIZ = Path(__file__).resolve().parent          # .../project
DIR_DADOS = RAIZ / "data"                        # amostras e arrays gerados
DIR_MODELOS = RAIZ / "models"                    # modelos treinados
DIR_FIGURAS = RAIZ / "figures"                   # graficos (matriz de confusao)

# O dataset bruto fica fora do projeto, na pasta dataset/ do repositorio.
DIR_DATASET = RAIZ.parent / "dataset" / "amazon_review_polarity_csv"
CSV_TREINO_BRUTO = DIR_DATASET / "train.csv"
CSV_TESTE_BRUTO = DIR_DATASET / "test.csv"


def garantir_pastas():
    """Cria data/, models/ e figures/ se ainda nao existirem."""
    for d in (DIR_DADOS, DIR_MODELOS, DIR_FIGURAS):
        d.mkdir(parents=True, exist_ok=True)


def limpar_texto(texto: str) -> str:
    """
    Normaliza um texto de review em 3 passos:
      1) coloca tudo em minusculas;
      2) remove tudo que NAO for letra (pontuacao, numeros, simbolos);
      3) colapsa espacos repetidos.
    E o mesmo espirito do pre-processamento da Aula 3.
    """
    texto = texto.lower()
    texto = re.sub(r"[^a-z\s]", " ", texto)   # so deixamos letras a-z e espacos
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto


def tokenizar(texto: str) -> list:
    """Limpa e quebra o texto em uma lista de palavras (tokens)."""
    return limpar_texto(texto).split()


def pad_features(sequencias, seq_length: int = 200):
    """
    Deixa TODAS as sequencias com o mesmo comprimento (seq_length):
      - review mais curta -> completa com zeros A ESQUERDA;
      - review mais longa -> corta para seq_length.
    Identico ao pad_features da Aula 3 (Sentiment_RNN). O zero a esquerda
    e o "padding" que a rede aprende a ignorar.
    """
    features = np.zeros((len(sequencias), seq_length), dtype=int)
    for i, linha in enumerate(sequencias):
        linha = list(linha)[:seq_length]
        if len(linha) > 0:
            features[i, -len(linha):] = np.array(linha)
    return features
