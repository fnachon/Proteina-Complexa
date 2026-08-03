"""Backbone continuity reward model.

Penalizes broken or clashing CA-CA backbone connectivity directly from
generated coordinates, without requiring a folding model. Intended to steer
reward-guided search (best-of-n, beam-search, fk-steering, mcts) away from
samples with geometrically invalid backbones.
"""

import torch
from loguru import logger

from proteinfoundation.rewards.base_reward import BaseRewardModel, standardize_reward

IDEAL_CA_DIST = 3.80  # Angstrom, typical trans-peptide CA-CA spacing
DEFAULT_BREAK_THRESHOLD = 4.5
DEFAULT_CLASH_THRESHOLD = 3.0


class BackboneContinuityRewardModel(BaseRewardModel):
    """Scores consecutive CA-CA distances against the ideal ~3.8 A spacing.

    Cheap, structure-only reward (no folding model) suitable for guiding
    search over raw flow-matching samples before any refinement/folding step.
    """

    IS_FOLDING_MODEL = False
    SUPPORTS_GRAD = False
    SUPPORTS_SAVE_PDB = False

    def __init__(
        self,
        break_threshold: float = DEFAULT_BREAK_THRESHOLD,
        clash_threshold: float = DEFAULT_CLASH_THRESHOLD,
        break_penalty: float = 10.0,
        clash_penalty: float = 10.0,
        deviation_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.break_threshold = break_threshold
        self.clash_threshold = clash_threshold
        self.break_penalty = break_penalty
        self.clash_penalty = clash_penalty
        self.deviation_weight = deviation_weight

    @staticmethod
    def _ca_coords_from_pdb(pdb_path: str) -> torch.Tensor:
        coords = []
        with open(pdb_path) as f:
            for line in f:
                if line.startswith("ATOM") and line[12:16].strip() == "CA":
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    coords.append((x, y, z))
        return torch.tensor(coords, dtype=torch.float32)

    def score(self, pdb_path: str, requires_grad: bool = False, **kwargs) -> dict:
        self._check_capabilities(requires_grad=requires_grad)

        try:
            ca = self._ca_coords_from_pdb(pdb_path)
        except (OSError, ValueError) as e:
            logger.warning(f"BackboneContinuityRewardModel: failed to read {pdb_path}: {e}")
            return standardize_reward(reward={"backbone_continuity": torch.tensor(0.0)}, total_reward=0.0)

        if ca.shape[0] < 2:
            return standardize_reward(reward={"backbone_continuity": torch.tensor(0.0)}, total_reward=0.0)

        d = torch.linalg.norm(ca[1:] - ca[:-1], dim=-1)
        deviation = (d - IDEAL_CA_DIST).abs().sum()
        n_broken = (d > self.break_threshold).sum()
        n_clash = (d < self.clash_threshold).sum()

        penalty = self.deviation_weight * deviation + self.break_penalty * n_broken + self.clash_penalty * n_clash
        total_reward = -penalty

        return standardize_reward(
            reward={
                "backbone_continuity": -penalty,
                "n_broken": n_broken.float(),
                "n_clash": n_clash.float(),
                "mean_ca_dist": d.mean(),
            },
            total_reward=total_reward,
        )
