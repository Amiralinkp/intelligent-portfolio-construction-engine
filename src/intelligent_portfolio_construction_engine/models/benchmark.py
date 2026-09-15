from dataclasses import dataclass



@dataclass(frozen=True)
class Benchmark:
    asset_class: str
    symbol: str
    name: str