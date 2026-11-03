
import json
from pathlib import Path
from typing import Any

from intelligent_portfolio_construction_engine.clustering.model_persistence import (
    ModelPersistence,
)
from intelligent_portfolio_construction_engine.models.clustering_model_artifact import (
    ClusteringModelArtifact,
)


class ModelVersionRegistry:

    def __init__(self, root_dir: Path) -> None:
        self.root_dir = Path(root_dir)
        self.models_dir = self.root_dir / "models"
        self.manifest_path = self.root_dir / "manifest.json"

        self.models_dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        artifact: ClusteringModelArtifact,
    ) -> Path:
        version = artifact.model_version

        if not version or Path(version).name != version:
            raise ValueError("Invalid model version.")

        model_path = self.models_dir / f"{version}.joblib"

        if model_path.exists():
            raise FileExistsError(
                f"Model version already exists: {version}"
            )

        self._validate_manifest()

        ModelPersistence.save(artifact, model_path)

        manifest = self._read_manifest()

        manifest["versions"][version] = {
            "model_path": str(model_path.relative_to(self.root_dir)),
            "model_name": artifact.model_name,
            "n_clusters": artifact.n_clusters,
            "training_cutoff": artifact.training_cutoff.isoformat(),
            "feature_columns": list(artifact.feature_columns),
        }

        manifest["latest_version"] = version

        self._write_manifest(manifest)

        return model_path

    def load(
        self,
        version: str,
    ) -> ClusteringModelArtifact:
        manifest = self._read_manifest()

        if version not in manifest["versions"]:
            raise KeyError(f"Unknown model version: {version}")

        model_path = (
            self.root_dir
            / manifest["versions"][version]["model_path"]
        )

        return ModelPersistence.load(model_path)

    def list_versions(self) -> list[dict[str, Any]]:
        manifest = self._read_manifest()

        return [
            {
                "version": version,
                **metadata,
            }
            for version, metadata in manifest["versions"].items()
        ]

    def get_latest_version(self) -> str | None:
        manifest = self._read_manifest()
        return manifest["latest_version"]

    def _read_manifest(self) -> dict[str, Any]:
        if not self.manifest_path.exists():
            return {
                "latest_version": None,
                "versions": {},
            }

        with self.manifest_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            manifest = json.load(file)

        if (
            not isinstance(manifest, dict)
            or not isinstance(manifest.get("versions"), dict)
        ):
            raise ValueError("Invalid model manifest.")

        return manifest

    def _write_manifest(
        self,
        manifest: dict[str, Any],
    ) -> None:
        temporary_path = self.manifest_path.with_name(
            f"{self.manifest_path.name}.tmp"
        )

        try:
            with temporary_path.open(
                "w",
                encoding="utf-8",
            ) as file:
                json.dump(
                    manifest,
                    file,
                    indent=4,
                )

            temporary_path.replace(self.manifest_path)

        finally:
            if temporary_path.exists():
                temporary_path.unlink()

    def _validate_manifest(self) -> None:
        manifest = self._read_manifest()

        if not isinstance(manifest.get("versions"), dict):
            raise ValueError("Invalid model manifest.")
