import pandas as pd
from sklearn.preprocessing import RobustScaler
from intelligent_portfolio_construction_engine.features.behavioral_profiling import CLUSTERING_FEATURES
from intelligent_portfolio_construction_engine.models.behavioral_profile import BehavioralProfile



def profiles_to_dataframe(profiles: list[BehavioralProfile]):
    rows = []

    for profile in profiles:
        row = {
            feature: getattr(profile, feature)
            for feature in CLUSTERING_FEATURES}
        
        row["symbol"] = profile.symbol
        row["asset_class"] = profile.asset_class
        rows.append(row)

    return pd.DataFrame(rows).set_index("symbol")


def validate_profiles(dataframe):
    feature_data = dataframe[CLUSTERING_FEATURES]

    valid_mask = feature_data.notna().all(axis=1)

    return dataframe.loc[valid_mask]

def scale_features(dataframe):
    scaler = RobustScaler()

    feature_data = dataframe[CLUSTERING_FEATURES]
    scaled_features = scaler.fit_transform(feature_data)

    scaled_dataframe = pd.DataFrame(scaled_features, index=dataframe.index, columns=CLUSTERING_FEATURES)

    return scaled_dataframe

def preprocess_behavioral_profiles(profiles: list[BehavioralProfile]):

    dataframe = profiles_to_dataframe(profiles)
    valid_dataframe = validate_profiles(dataframe)
    
    return scale_features(valid_dataframe)