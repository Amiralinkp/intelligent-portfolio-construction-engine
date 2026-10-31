import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from intelligent_portfolio_construction_engine.clustering.cluster_membership import (
    ClusterMembershipAnalyzer,
)
from intelligent_portfolio_construction_engine.clustering.cluster_profiling import (
    FEATURE_COLUMNS,
)
from intelligent_portfolio_construction_engine.models.cluster_alignment_result import (
    ClusterAlignmentResult,
    ClusterMatch,
    ClusterStructuralEvent,
)
from intelligent_portfolio_construction_engine.models.match_decision import (
    MatchDecision,
)
from intelligent_portfolio_construction_engine.models.cluster_profile import (
    ClusterProfile,
)


class ClusterAlignmentEngine:
    def __init__(
        self,
        membership_analyzer: ClusterMembershipAnalyzer | None = None,
    ) -> None:
        self.membership_analyzer = (
            membership_analyzer or ClusterMembershipAnalyzer()
        )

    def align(
        self,
        previous_profiles: list[ClusterProfile],
        current_profiles: list[ClusterProfile],
        reference_scale: pd.Series,
        previous_assignments=None,
        current_assignments=None,
        threshold_estimator=None,
        previous_features: dict[int, pd.DataFrame] | None = None,
        current_features: dict[int, pd.DataFrame] | None = None,
    ) -> ClusterAlignmentResult:
        self._validate_reference_scale(reference_scale)
        self._validate_profiles(previous_profiles, "previous")
        self._validate_profiles(current_profiles, "current")

        matches = []
        decisions = []
        structural_events = []

        if previous_profiles and current_profiles:
            costs = self._build_cost_matrix(
                previous_profiles,
                current_profiles,
                reference_scale,
            )

            threshold_95, threshold_99 = self._build_threshold_matrices(
                previous_profiles,
                current_profiles,
                threshold_estimator,
                previous_features,
                current_features,
                reference_scale,
            )

            matches, decisions = self._solve_assignment(
                previous_profiles,
                current_profiles,
                costs,
                threshold_95,
                threshold_99,
            )
        else:
            matches = self._build_unmatched_results(
                previous_profiles,
                current_profiles,
            )

        if previous_assignments is not None and current_assignments is not None:
            self._validate_assignments(
                previous_assignments,
                current_assignments,
            )

            membership_result = self.membership_analyzer.analyze(
                previous_assignments,
                current_assignments,
            )

            structural_events = self._detect_structural_events(
                membership_result.overlaps
            )

        return ClusterAlignmentResult(
            matches=matches,
            decisions=decisions,
            structural_events=structural_events,
        )

    @staticmethod
    def _validate_profiles(
        profiles: list[ClusterProfile],
        version_name: str,
    ) -> None:
        cluster_ids = [profile.cluster_id for profile in profiles]

        if len(cluster_ids) != len(set(cluster_ids)):
            raise ValueError(
                f"Duplicate cluster IDs in {version_name} profiles."
            )

        for profile in profiles:
            for feature in FEATURE_COLUMNS:
                if feature not in profile.centroid:
                    raise ValueError(
                        f"Missing centroid feature: {feature}"
                    )

                if not np.isfinite(profile.centroid[feature]):
                    raise ValueError(
                        f"Non-finite centroid value: {feature}"
                    )

    @staticmethod
    def _validate_reference_scale(
        reference_scale: pd.Series,
    ) -> None:
        missing = set(FEATURE_COLUMNS) - set(reference_scale.index)

        if missing:
            raise ValueError(
                f"Missing reference scale features: {sorted(missing)}"
            )

        values = reference_scale[FEATURE_COLUMNS].to_numpy(dtype=float)

        if not np.isfinite(values).all() or (values <= 0).any():
            raise ValueError(
                "Reference scale values must be finite and positive."
            )

    @staticmethod
    def _build_cost_matrix(
        previous_profiles: list[ClusterProfile],
        current_profiles: list[ClusterProfile],
        reference_scale: pd.Series,
    ) -> np.ndarray:
        scale = reference_scale[FEATURE_COLUMNS].to_numpy(dtype=float)

        previous = np.array(
            [
                [profile.centroid[feature] for feature in FEATURE_COLUMNS]
                for profile in previous_profiles
            ],
            dtype=float,
        )

        current = np.array(
            [
                [profile.centroid[feature] for feature in FEATURE_COLUMNS]
                for profile in current_profiles
            ],
            dtype=float,
        )

        normalized_previous = previous / scale
        normalized_current = current / scale

        return np.linalg.norm(
            normalized_previous[:, None, :]
            - normalized_current[None, :, :],
            axis=2,
        )

    @staticmethod
    def _build_threshold_matrices(
        previous_profiles,
        current_profiles,
        estimator,
        previous_features,
        current_features,
        reference_scale,
    ) -> tuple[np.ndarray, np.ndarray]:
        if estimator is None:
            raise ValueError("A threshold estimator is required.")

        if previous_features is None or current_features is None:
            raise ValueError("Cluster feature data is required.")

        threshold_95 = np.empty(
            (len(previous_profiles), len(current_profiles))
        )
        threshold_99 = np.empty_like(threshold_95)

        for i, old_profile in enumerate(previous_profiles):
            if old_profile.cluster_id not in previous_features:
                raise ValueError(
                    f"Missing features for old cluster "
                    f"{old_profile.cluster_id}."
                )

            for j, new_profile in enumerate(current_profiles):
                if new_profile.cluster_id not in current_features:
                    raise ValueError(
                        f"Missing features for new cluster "
                        f"{new_profile.cluster_id}."
                    )

                result = estimator.estimate(
                    previous_features[old_profile.cluster_id],
                    current_features[new_profile.cluster_id],
                    reference_scale,
                )

                threshold_95[i, j] = result.threshold_95
                threshold_99[i, j] = result.threshold_99

        if (
            not np.isfinite(threshold_95).all()
            or not np.isfinite(threshold_99).all()
            or (threshold_95 <= 0).any()
            or (threshold_99 < threshold_95).any()
        ):
            raise ValueError("Invalid pairwise thresholds.")

        return threshold_95, threshold_99

    @staticmethod
    def _solve_assignment(
        previous_profiles,
        current_profiles,
        costs,
        threshold_95,
        threshold_99,
    ):
        old_count, new_count = costs.shape
        size = old_count + new_count

        invalid_cost = 1e9
        unmatched_penalty = 0.5

        valid = costs <= threshold_99

        normalized_costs = costs / threshold_99

        matrix = np.full((size, size), invalid_cost)

        matrix[:old_count, :new_count] = np.where(
            valid,
            normalized_costs,
            invalid_cost,
        )

        for i in range(old_count):
            matrix[i, new_count + i] = unmatched_penalty

        for j in range(new_count):
            matrix[old_count + j, j] = unmatched_penalty

        matrix[old_count:, new_count:] = 0.0

        rows, columns = linear_sum_assignment(matrix)

        matches = []
        decisions = []

        matched_old = set()
        matched_new = set()

        for row, column in zip(rows, columns, strict=True):
            if row >= old_count or column >= new_count:
                continue

            if not valid[row, column]:
                continue

            old_id = previous_profiles[row].cluster_id
            new_id = current_profiles[column].cluster_id
            distance = float(costs[row, column])

            if distance <= threshold_95[row, column]:
                status = "matched"
                confidence = "within_95"
            else:
                status = "conditional"
                confidence = "within_99"

            decisions.append(
                MatchDecision(
                    old_cluster_id=old_id,
                    new_cluster_id=new_id,
                    match_cost=distance,
                    threshold_95=float(threshold_95[row, column]),
                    threshold_99=float(threshold_99[row, column]),
                    match_status=status,
                    confidence_level=confidence,
                )
            )

            matches.append(
                ClusterMatch(
                    old_cluster_id=old_id,
                    new_cluster_id=new_id,
                    match_cost=distance,
                    match_status=status,
                )
            )

            matched_old.add(row)
            matched_new.add(column)

        for i, profile in enumerate(previous_profiles):
            if i not in matched_old:
                matches.append(
                    ClusterMatch(
                        old_cluster_id=profile.cluster_id,
                        new_cluster_id=None,
                        match_cost=None,
                        match_status="retired",
                    )
                )

        for j, profile in enumerate(current_profiles):
            if j not in matched_new:
                matches.append(
                    ClusterMatch(
                        old_cluster_id=None,
                        new_cluster_id=profile.cluster_id,
                        match_cost=None,
                        match_status="new",
                    )
                )

        return matches, decisions

    @staticmethod
    def _build_unmatched_results(
        previous_profiles,
        current_profiles,
    ):
        matches = [
            ClusterMatch(
                old_cluster_id=profile.cluster_id,
                new_cluster_id=None,
                match_cost=None,
                match_status="retired",
            )
            for profile in previous_profiles
        ]

        matches.extend(
            ClusterMatch(
                old_cluster_id=None,
                new_cluster_id=profile.cluster_id,
                match_cost=None,
                match_status="new",
            )
            for profile in current_profiles
        )

        return matches

    @staticmethod
    def _validate_assignments(
        previous_assignments,
        current_assignments,
    ) -> None:
        previous_assets = [
            assignment.asset for assignment in previous_assignments
        ]
        current_assets = [
            assignment.asset for assignment in current_assignments
        ]

        if len(previous_assets) != len(set(previous_assets)):
            raise ValueError("Duplicate assets in previous assignments.")

        if len(current_assets) != len(set(current_assets)):
            raise ValueError("Duplicate assets in current assignments.")

    @staticmethod
    def _detect_structural_events(overlaps):
        events = []

        old_to_new = {}
        new_to_old = {}

        for overlap in overlaps:
            old_to_new.setdefault(overlap.old_cluster_id, set()).add(
                overlap.new_cluster_id
            )
            new_to_old.setdefault(overlap.new_cluster_id, set()).add(
                overlap.old_cluster_id
            )

        for old_id, new_ids in old_to_new.items():
            if len(new_ids) > 1:
                events.append(
                    ClusterStructuralEvent(
                        event_type="split",
                        old_cluster_ids=(old_id,),
                        new_cluster_ids=tuple(sorted(new_ids)),
                        shared_assets=sum(
                            item.shared_assets
                            for item in overlaps
                            if item.old_cluster_id == old_id
                        ),
                    )
                )

        for new_id, old_ids in new_to_old.items():
            if len(old_ids) > 1:
                events.append(
                    ClusterStructuralEvent(
                        event_type="merge",
                        old_cluster_ids=tuple(sorted(old_ids)),
                        new_cluster_ids=(new_id,),
                        shared_assets=sum(
                            item.shared_assets
                            for item in overlaps
                            if item.new_cluster_id == new_id
                        ),
                    )
                )

        return events