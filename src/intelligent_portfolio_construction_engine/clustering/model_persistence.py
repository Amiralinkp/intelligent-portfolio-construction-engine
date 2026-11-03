from pathlib import Path

import joblib

from intelligent_portfolio_construction_engine.models.clustering_model_artifact import (
    ClusteringModelArtifact,
)


class ModelPersistence:

    @staticmethod
    def save(
        artifact: ClusteringModelArtifact,
        path: Path,
    ) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        temporary_path = path.with_name(f"{path.name}.tmp")

        try:
            joblib.dump(artifact, temporary_path)
            temporary_path.replace(path)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()

        return path

    @staticmethod
    def load(path: Path) -> ClusteringModelArtifact:
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Model artifact not found: {path}")

        artifact = joblib.load(path)

        if not isinstance(artifact, ClusteringModelArtifact):
            raise TypeError("Loaded file is not a ClusteringModelArtifact.")

        return artifact