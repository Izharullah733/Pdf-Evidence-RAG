"""Render the four report sections into a reproducible four-page PDF."""

from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer


def build_report():
    source = Path("docs/technical_report.md").read_text(encoding="utf-8")
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            "ReportBody",
            fontName="Helvetica",
            fontSize=10,
            leading=14,
            spaceAfter=10,
            textColor=colors.HexColor("#172033"),
        )
    )
    story = []
    for number, section in enumerate(source.split("<!-- PAGEBREAK -->")):
        if number:
            story.append(PageBreak())
        for block in section.strip().split("\n\n"):
            # Leading title and heading may occupy consecutive lines.
            if block.startswith("#"):
                for line in block.splitlines():
                    level = "Title" if line.startswith("# ") else "Heading2"
                    story.append(Paragraph(escape(line.lstrip("# ")), styles[level]))
                story.append(Spacer(1, 6))
            else:
                story.append(Paragraph(escape(block.replace("\n", " ")), styles["ReportBody"]))

    def footer(canvas, doc):
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#64748b"))
        canvas.drawString(42, 24, "Intermediate RAG System Using Pinecone | Technical Report")
        canvas.drawRightString(A4[0] - 42, 24, str(doc.page))

    target = Path("docs/technical_report.pdf")
    SimpleDocTemplate(
        str(target),
        pagesize=A4,
        topMargin=35,
        bottomMargin=40,
        leftMargin=42,
        rightMargin=42,
        title="Intermediate RAG System Using Pinecone",
        author="",
    ).build(story, onFirstPage=footer, onLaterPages=footer)
    from pypdf import PdfReader

    count = len(PdfReader(target).pages)
    if count != 4:
        raise RuntimeError(f"Expected 4 report pages; got {count}. Adjust layout before submission.")
    print(f"Created {target} ({count} pages)")


if __name__ == "__main__":
    build_report()
