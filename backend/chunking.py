

from __future__ import annotations

from langchain_text_splitters import RecursiveCharacterTextSplitter

from schemas import StructuredBugReport

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Ordered so the splitter groups stack-trace frames ("at ...") together
# before falling back to line/word/character boundaries.
SEPARATORS = ["\n\n", "\n", "at ", "  ", " "]


def build_document_text(structured: StructuredBugReport) -> str:
    """
    Flattens the structured JSON payload into one coherent text document
    ready for chunking + embedding — keeping the stack trace verbatim.
    """
    env = structured.environment
    meta = structured.metadata

    sections = [
        f"Error Code: {structured.error_code or 'N/A'}",
        f"Title: {meta.title or structured.message}",
        f"Severity: {meta.severity}",
        f"Component: {meta.component or 'N/A'}",
        f"Environment: {env.os or 'N/A'} | {env.runtime or 'N/A'} | {env.version or 'N/A'}",
        f"Message: {structured.message}",
    ]
    if structured.stack_trace:
        sections.append(f"Stack Trace:\n{structured.stack_trace}")

    return "\n\n".join(sections)


def chunk_document(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """Splits text into overlapping chunks tuned for logs/stack traces."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=SEPARATORS,
    )
    return splitter.split_text(text)


def chunk_structured_report(structured: StructuredBugReport) -> list[str]:
    """Convenience wrapper: structured report -> flattened text -> chunks."""
    doc_text = build_document_text(structured)
    return chunk_document(doc_text)
