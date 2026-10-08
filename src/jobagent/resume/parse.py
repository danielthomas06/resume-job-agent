from pathlib import Path


def parse_resume(path: str | Path) -> str:
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".pdf":
        import pymupdf

        with pymupdf.open(p) as doc:
            return "\n".join(page.get_text() for page in doc).strip()
    if suffix == ".docx":
        import docx

        return "\n".join(par.text for par in docx.Document(str(p)).paragraphs).strip()
    if suffix in {".txt", ".md"}:
        return p.read_text(encoding="utf-8").strip()
    raise ValueError(f"Unsupported resume format: {suffix}")
