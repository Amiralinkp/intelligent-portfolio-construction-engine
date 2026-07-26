from intelligent_portfolio_construction_engine.models.asset_profile import AssetProfile
import numpy as np
from sklearn.preprocessing import MinMaxScaler



class FeatureScorer:

    def __init__(self, profiles, weights):
        
        self.profiles = profiles
        self.weights = weights
    def score(self):

        feature_matrix = []
        
        
        for profile in self.profiles:

            features = profile.features

            row = [ features.daily_return,
                   features.annual_return,
                   features.volatility,
                   features.max_drawdown,
                   features.roc,
                   features.atr,
                   features.rsi,
                   features.macd,
                   features.macd_signal,
                   features.macd_hist]
            
            feature_matrix.append(row)
        
        

        feature_arr = np.asanyarray(feature_matrix)
        scaler = MinMaxScaler()
        normalized_features = scaler.fit_transform(feature_arr)

        normalized_features[:, 2] = 1 - normalized_features[:, 2]  # volatility
        normalized_features[:, 3] = 1 - normalized_features[:, 3]  # max_drawdown
        normalized_features[:, 5] = 1 - normalized_features[:, 5]  # atr

        
        for profile, row in zip(self.profiles, normalized_features):


            res = []
            for i in range(len(row)):

                
                res.append(self.weights[i] * row[i])

            score = sum(res)
            profile.score = score
        

        ranked_profiles = sorted(self.profiles, key=lambda profile : profile.score, reverse=True)

        for rank, profile in enumerate(ranked_profiles, start=1):
            profile.rank = rank

        return ranked_profiles
    








        

