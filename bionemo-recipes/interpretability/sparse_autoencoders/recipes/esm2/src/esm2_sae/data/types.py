from dataclasses import dataclass

@dataclass
class ProteinRecord:
    """Container for a protein sequence record."""
    id: str
    sequence: str
    description: str = ""

    @property
    def length(self) -> int:
        return len(self.sequence)
