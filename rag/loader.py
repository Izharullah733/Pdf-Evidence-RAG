import hashlib
import io
import re
import unicodedata

from pypdf import PdfReader

from rag.models import Document, Page, RagError

MAX_PDF_BYTES = 20 * 1024 * 1024


def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).replace("\x00", "").replace("\u00ad", "")
    text = re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", "", text)
    return re.sub(r"\s+", " ", text).strip()


def load_pdf(data: bytes, name: str) -> Document:
    if not name.lower().endswith(".pdf") or not data.startswith(b"%PDF-"):
        raise RagError(f"{name}: upload a valid PDF file.")
    if len(data) > MAX_PDF_BYTES:
        raise RagError(f"{name}: exceeds the 20 MB limit.")
    try:
        reader = PdfReader(io.BytesIO(data), strict=False)
        if reader.is_encrypted:
            raise RagError(f"{name}: password-protected PDFs are not supported.")
        pages = [Page(i, clean_text(page.extract_text() or "")) for i, page in enumerate(reader.pages, 1)]
    except RagError:
        raise
    except Exception as exc:
        raise RagError(f"{name}: could not read this PDF; it may be damaged.") from exc
    nonempty = [page for page in pages if page.text]
    if not nonempty:
        raise RagError(f"{name}: no extractable text. Scan-only PDFs require OCR first.")
    return Document(hashlib.sha256(data).hexdigest()[:24], name, nonempty, len(pages))
