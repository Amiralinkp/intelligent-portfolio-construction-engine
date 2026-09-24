import pandas as pd
from intelligent_portfolio_construction_engine.models.asset_cluster_assignment import AssetClusterAssignment


def build_asset_cluster_assignments(assets: pd.Index, labels):

    return [AssetClusterAssignment(asset=str(asset), cluster_id=int(cluster_id)) 
            for asset, cluster_id in zip(assets, labels, strict=True)]

