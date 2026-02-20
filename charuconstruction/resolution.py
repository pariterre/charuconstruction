from enum import Enum


class Resolution(Enum):
    DPI_100 = "dpi_100"
    DPI_200 = "dpi_200"
    DPI_300 = "dpi_300"

    @property
    def pixels_per_meter(self) -> int:
        if self == Resolution.DPI_100:
            return 4724
        elif self == Resolution.DPI_200:
            return 7874
        elif self == Resolution.DPI_300:
            return 11811
        else:
            raise ValueError(f"Unsupported resolution: {self}")
