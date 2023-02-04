from dataclasses import dataclass


@dataclass
class Document:
    """Dataclass object that represents a document."""

    extracted_at: str
    id: str
    lang: str
    text: str
