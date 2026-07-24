import pandas as pd

class DataSplitter:

    def split_assets(self, data):

        assets = {}

        symbols = data.columns.get_level_values(1)
        symbols = symbols.unique()

    
        for symbol in symbols:
            asset_df = data.xs(symbol, axis=1, level=1)
            asset_df.columns.name = None
            assets[symbol] = asset_df

        return assets