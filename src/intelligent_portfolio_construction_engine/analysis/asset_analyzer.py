from intelligent_portfolio_construction_engine.models.asset_profile import AssetProfile



class AssetAnalyzer:

    def __init__(self, feature_engine):

        self.feature_engine = feature_engine


    def analyze(self, symbol, asset_df):

        features = self.feature_engine.extract_features(asset_df)
        profile = AssetProfile(
                            symbol=symbol,
                            features=features)
        
        return profile
