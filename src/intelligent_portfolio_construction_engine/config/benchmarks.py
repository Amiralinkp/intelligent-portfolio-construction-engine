from intelligent_portfolio_construction_engine.models.benchmark import Benchmark


BENCHMARKS = {
    "equity": Benchmark(
        asset_class="equity",
        symbol="^GSPC",
        name="S&P 500"),
    "gold": Benchmark(
        asset_class="gold",
        symbol="GC=F",
        name="Gold Futures"),
    "crypto": Benchmark(
        asset_class="crypto",
        symbol="BTC-USD",
        name="Bitcoin")}

def get_benchmark(asset_class) -> Benchmark:
    try:
        return BENCHMARKS[asset_class]
    except KeyError as exc:
        raise ValueError(f"Unsupported asset class: {asset_class}") from exc