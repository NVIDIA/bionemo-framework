

from enum import Enum


class Version:
    """
    Generic version class (used throughout SCDL including for new backing implementations).
    """
    major: int
    minor: int
    point: int

class SCDLVersion(Version):
    """
    Version of the SCDL schema. This is the version of the schema that is used to
    store the data in the archive.
    """
    major: int = 0
    minor: int = 0
    point: int = 0

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.point}"
    
    def __repr__(self) -> str:
        return f"SCDLVersion(major={self.major}, minor={self.minor}, point={self.point})"
    
    def __eq__(self, other: "SCDLVersion") -> bool:
        return self.major == other.major and self.minor == other.minor and self.point == other.point
    
    def __ne__(self, other: "SCDLVersion") -> bool:
        return not self == other
    
class CurrentSCDLVersion(SCDLVersion):
    """
    Current version of the SCDL schema.
    """
    major: int = 0
    minor: int = 2
    point: int = 0

class SCDLBackends(Enum):
    """
    Backends of the SCDL schema.
    """
    MEMMAP_V0 = 'memmap_v0'
