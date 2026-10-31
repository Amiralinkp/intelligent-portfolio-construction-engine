from collections import defaultdict

from intelligent_portfolio_construction_engine.models.asset_cluster_assignment import (
    AssetClusterAssignment,
)
from intelligent_portfolio_construction_engine.models.cluster_membership_result import (
    ClusterMembershipResult,
    ClusterOverlap,
)


class ClusterMembershipAnalyzer:
    def analyze(
        self,
        previous_assignments: list[AssetClusterAssignment],
        current_assignments: list[AssetClusterAssignment],
    ) -> ClusterMembershipResult:

        previous_groups = self._group_assignments(previous_assignments)
        current_groups = self._group_assignments(current_assignments)

        overlaps = []

        for old_cluster_id, old_assets in previous_groups.items():
            for new_cluster_id, new_assets in current_groups.items():
                shared_assets = old_assets.intersection(new_assets)

                if not shared_assets:
                    continue

                overlaps.append(
                    ClusterOverlap(
                        old_cluster_id=old_cluster_id,
                        new_cluster_id=new_cluster_id,
                        shared_assets=len(shared_assets),
                        old_coverage=len(shared_assets) / len(old_assets),
                        new_coverage=len(shared_assets) / len(new_assets),
                    )
                )

        return ClusterMembershipResult(overlaps=overlaps)

    @staticmethod
    def _group_assignments(assignments):
        groups = defaultdict(set)

        for assignment in assignments:
            groups[assignment.cluster_id].add(assignment.asset)

        return dict(groups)