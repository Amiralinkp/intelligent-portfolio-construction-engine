from intelligent_portfolio_construction_engine.models.asset_profile import AssetProfile



class AssetAnalyzer:

    def __init__(self, feature_engine, provider):

        self.feature_engine = feature_engine
        self.provider = provider

    def analyze(self, symbol, asset_df):

        features = self.feature_engine.extract_features(asset_df)
        asset_info = self.provider.get_asset_info(symbol)

        profile = AssetProfile(
                            symbol=symbol,
                            sector=asset_info["sector"],
                            features=features)
        
        return profile
