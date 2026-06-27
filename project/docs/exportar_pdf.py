"""
exportar_pdf.py
===============
Gera os PDFs de entrega a partir dos arquivos Markdown:
  - documento.md       -> documento.pdf       (A4 retrato, estilo documento)
  - roteiro_slides.md  -> roteiro_slides.pdf   (A4 paisagem, 1 slide por pagina)

Requer (uma vez):  pip install markdown weasyprint
Uso:               python docs/exportar_pdf.py
"""
from pathlib import Path

import markdown
from weasyprint import HTML

AQUI = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Estilo do DOCUMENTO (A4 retrato)
# ---------------------------------------------------------------------------
CSS_DOCUMENTO = """
@page { size: A4; margin: 2cm; @bottom-center { content: counter(page); font-size: 9pt; color: #999; } }
body { font-family: 'DejaVu Sans', sans-serif; font-size: 11pt; line-height: 1.5; color: #1a1a1a; }
h1 { font-size: 20pt; color: #b30000; border-bottom: 2px solid #b30000; padding-bottom: 4px; }
h2 { font-size: 15pt; color: #b30000; margin-top: 1.2em; }
h3 { font-size: 12.5pt; color: #333; }
table { border-collapse: collapse; width: 100%; margin: 1em 0; font-size: 10pt; }
th, td { border: 1px solid #bbb; padding: 5px 8px; text-align: left; }
th { background: #f2dede; }
code { background: #f4f4f4; padding: 1px 4px; border-radius: 3px; font-family: 'DejaVu Sans Mono', monospace; font-size: 9.5pt; }
pre { background: #f7f7f7; border: 1px solid #e0e0e0; padding: 10px; }
pre code { background: none; font-size: 9pt; }
img { max-width: 70%; display: block; margin: 1em auto; }
"""

# ---------------------------------------------------------------------------
# Estilo dos SLIDES (A4 paisagem; cada "---" do Markdown quebra a pagina)
# ---------------------------------------------------------------------------
CSS_SLIDES = """
@page { size: A4 landscape; margin: 1.6cm; }
body { font-family: 'DejaVu Sans', sans-serif; font-size: 15pt; line-height: 1.45; color: #1a1a1a; }
h1 { font-size: 22pt; color: #b30000; }
h2 { font-size: 19pt; color: #b30000; }
hr { border: none; height: 0; margin: 0; page-break-after: always; }
table { border-collapse: collapse; width: 100%; font-size: 13pt; margin: 0.4em 0; }
th, td { border: 1px solid #bbb; padding: 6px 10px; }
th { background: #f2dede; }
pre { background: #f7f7f7; border: 1px solid #e0e0e0; padding: 8px; font-size: 11pt; }
code { font-family: 'DejaVu Sans Mono', monospace; }
em { color: #555; }
"""


def exportar(md_path: Path, pdf_path: Path, css: str):
    texto = md_path.read_text(encoding="utf-8")
    corpo = markdown.markdown(texto, extensions=["tables", "fenced_code", "sane_lists"])
    html = (
        "<html><head><meta charset='utf-8'><style>" + css + "</style></head>"
        "<body>" + corpo + "</body></html>"
    )
    # base_url = pasta do .md, para resolver imagens com caminho relativo (../figures/...)
    HTML(string=html, base_url=str(md_path.parent)).write_pdf(str(pdf_path))
    print(f"   gerado: {pdf_path}  ({pdf_path.stat().st_size // 1024} KB)")


def main():
    print("Exportando PDFs...")
    exportar(AQUI / "documento.md", AQUI / "documento.pdf", CSS_DOCUMENTO)
    exportar(AQUI / "roteiro_slides.md", AQUI / "roteiro_slides.pdf", CSS_SLIDES)
    print("Pronto.")


if __name__ == "__main__":
    main()
