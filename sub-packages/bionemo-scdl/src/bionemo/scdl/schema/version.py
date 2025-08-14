

from enum import Enum


class Version:
    """
    Generic version class (used throughout SCDL including for new backing implementations).
    """
    
    def __init__(self, major: int = 0, minor: int = 0, point: int = 0):
        """Initialize version with major, minor, and point values."""
        self.major = major
        self.minor = minor
        self.point = point

class SCDLVersion(Version):
    """
    Version of the SCDL schema. This is the version of the schema that is used to
    store the data in the archive.
    """
    
    def __init__(self, major: int = 0, minor: int = 0, point: int = 0):
        """Initialize SCDL version with major, minor, and point values."""
        super().__init__(major, minor, point)

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
    
    def __init__(self):
        """
        Initialize with the current SCDL schema version: 0.0.9
        """
        super().__init__(major=0,
                         minor=0,
                         point=9)

# Note: Backend enums are defined in header.py to maintain consistency
# with binary serialization format which requires integer enum values
