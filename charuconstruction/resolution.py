from enum import Enum


class Resolution(Enum):
    DPI_100 = 4724  # 100 DPI corresponds to 4724 pixels per meter
    DPI_200 = 7874  # 200 DPI corresponds to 7874 pixels per meter
    DPI_300 = 11811  # 300 DPI corresponds to 11811 pixels per meter

    @property
    def serialized(self) -> str:
        return self.name

    @classmethod
    def from_serialized(cls, name: str) -> "Resolution":
        for member in cls:
            if member.name == name:
                return member
        raise ValueError(f"No matching Resolution for name: {name}")
