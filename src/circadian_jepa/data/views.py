"""View generation for JEPA training on scRNA-seq counts.

Supports three count-augmentation strategies (selected via ``view_mode``):

- ``"symmetric_split"``: each UMI is independently assigned to view A or view B
  via Binomial(n, 0.5). The two views partition the original counts and are at
  ~50% depth each. Maximum invariance signal but most aggressive — can wipe out
  signal in low-count genes (e.g., circadian clock genes).

- ``"asymmetric"`` (default): view A keeps the full counts; view B is a thinned
  version via Binomial(n, p) with ``p = thinning_p``. Convention: feed view A
  to the **teacher** and view B to the **student**. Preserves the maximum-
  information supervisory signal at the cost of breaking view symmetry.

- ``"light_independent"``: both views are independently thinned via
  Binomial(n, p) with ``p = thinning_p_light``. Views share most UMIs and are
  each at ~``thinning_p_light`` of original depth. Mildest augmentation; rely
  more heavily on regularizers to prevent collapse.

Orthogonally, same-batch pairing can be enabled to pair cells sharing the same
batch label (e.g., same sacrifice timepoint in a circadian experiment). The
chosen ``view_mode`` then controls how each cell's counts are corrupted before
being placed into view A and view B respectively.
"""

from __future__ import annotations

import warnings

import numpy as np
import torch

VALID_MODES = ("symmetric_split", "asymmetric", "light_independent")


class ViewGenerator:
    """Generate paired views of single-cell count data for JEPA training.

    Parameters
    ----------
    view_mode:
        One of ``"symmetric_split"``, ``"asymmetric"``, ``"light_independent"``.
        See module docstring for semantics. Default ``"asymmetric"``.
    thinning_p:
        Keep probability for the thinned view in ``"asymmetric"`` mode.
        Higher = milder thinning. Default 0.8.
    thinning_p_light:
        Keep probability for both views in ``"light_independent"`` mode.
        Default 0.8.
    same_batch:
        If True, pair cells with the same batch label (see ``make_batch_pairs``)
        rather than producing two views of the same cell. Default False.
    batch_key:
        Key used downstream by callers to retrieve batch labels (kept for API
        compatibility; this class doesn't read it directly). Default ``"batch"``.
    mask_prob:
        Probability of zeroing each gene in each view post-normalization.
        Applied independently per view. Default 0.0.
    seed:
        Seed for the partner-sampling RNG used in ``make_batch_pairs``.
    """

    def __init__(
        self,
        view_mode: str = "asymmetric",
        thinning_p: float = 0.8,
        thinning_p_light: float = 0.8,
        same_batch: bool = False,
        batch_key: str = "batch",
        mask_prob: float = 0.0,
        seed: int = 0,
    ) -> None:
        if view_mode not in VALID_MODES:
            raise ValueError(
                f"view_mode must be one of {VALID_MODES}, got {view_mode!r}"
            )
        if not 0.0 < thinning_p <= 1.0:
            raise ValueError(f"thinning_p must be in (0, 1], got {thinning_p}")
        if not 0.0 < thinning_p_light <= 1.0:
            raise ValueError(
                f"thinning_p_light must be in (0, 1], got {thinning_p_light}"
            )
        if not 0.0 <= mask_prob < 1.0:
            raise ValueError(f"mask_prob must be in [0, 1), got {mask_prob}")

        self.view_mode = view_mode
        self.thinning_p = thinning_p
        self.thinning_p_light = thinning_p_light
        self.same_batch = same_batch
        self.batch_key = batch_key
        self.mask_prob = mask_prob
        self.seed = seed

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize(counts: torch.Tensor) -> torch.Tensor:
        """Total-count normalize to 1e4, then log1p. counts: (..., n_genes)."""
        totals = counts.sum(dim=-1, keepdim=True).clamp(min=1.0)
        return torch.log1p(counts / totals * 1e4)

    @staticmethod
    def _binomial_thin(counts: torch.Tensor, p: float) -> torch.Tensor:
        """Thin each integer count by Binomial(n, p). Returns float tensor."""
        if p >= 1.0:
            return counts.float()
        probs = torch.full_like(counts.float(), p)
        return (
            torch.distributions.Binomial(total_count=counts.long(), probs=probs)
            .sample()
            .float()
        )

    def _apply_mask(self, x: torch.Tensor) -> torch.Tensor:
        if self.mask_prob > 0:
            keep = torch.bernoulli(torch.full_like(x, 1.0 - self.mask_prob))
            return x * keep
        return x

    # ------------------------------------------------------------------
    # Core view generation: two views of the same cell
    # ------------------------------------------------------------------
    def _split_counts(self, counts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply the configured view_mode to produce raw (un-normalized) views.

        Returns float tensors of integer-valued counts.
        """
        counts = counts.float()

        if self.view_mode == "symmetric_split":
            # Split each UMI: each goes to A with prob 0.5, else to B.
            split_a = self._binomial_thin(counts, p=0.5)
            split_b = counts - split_a
            return split_a, split_b

        elif self.view_mode == "asymmetric":
            # View A: full counts. View B: thinned.
            view_a = counts.clone()
            view_b = self._binomial_thin(counts, p=self.thinning_p)
            return view_a, view_b

        elif self.view_mode == "light_independent":
            # Both views independently thinned.
            view_a = self._binomial_thin(counts, p=self.thinning_p_light)
            view_b = self._binomial_thin(counts, p=self.thinning_p_light)
            return view_a, view_b

        else:  # pragma: no cover — guarded by __init__
            raise RuntimeError(f"Unhandled view_mode: {self.view_mode}")

    def make_pair(self, counts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate two views of the same cells under the configured view_mode.

        Parameters
        ----------
        counts:
            Integer count tensor of shape (n_cells, n_genes).

        Returns
        -------
        view_a, view_b:
            Normalized float tensors of the same shape. In ``"asymmetric"``
            mode, ``view_a`` is the full-count "clean" view (intended for the
            teacher) and ``view_b`` is the thinned view (intended for the
            student). In symmetric modes, the two views are interchangeable.
        """
        raw_a, raw_b = self._split_counts(counts)
        view_a = self._apply_mask(self._normalize(raw_a))
        view_b = self._apply_mask(self._normalize(raw_b))
        return view_a, view_b

    # ------------------------------------------------------------------
    # Same-batch pairing
    # ------------------------------------------------------------------
    def make_batch_pairs(
        self,
        counts: torch.Tensor,
        batch_labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """For each cell, pair it with a same-batch partner and apply view_mode.

        The two cells in each pair are different cells assumed to share the
        underlying phase variable (e.g., same circadian timepoint). The
        configured ``view_mode`` then controls how each cell's counts are
        corrupted before being placed into view A and view B.

        Semantics by mode:

        - ``"symmetric_split"``: each cell is split via Binomial(n, 0.5);
          view A gets the first half of the original cell, view B gets the
          first half of the partner. (The unused halves are discarded — this
          makes both views at ~50% depth, consistent with non-batch behavior.)
        - ``"asymmetric"``: view A is the original cell at full depth; view B
          is the partner cell thinned by ``thinning_p``. NB: the asymmetry is
          across cells here, which is conceptually murkier than in the single-
          cell case. Use with care.
        - ``"light_independent"``: each cell independently thinned by
          ``thinning_p_light``.

        Parameters
        ----------
        counts:
            Integer count tensor of shape (n_cells, n_genes).
        batch_labels:
            1-D integer tensor of shape (n_cells,) with batch IDs.

        Returns
        -------
        view_a, view_b: each (n_cells, n_genes), normalized float tensors.
        """
        if self.view_mode == "asymmetric":
            warnings.warn(
                "Combining view_mode='asymmetric' with same_batch=True is "
                "conceptually murky: view A is the original cell at full depth "
                "while view B is a *different* (partner) cell thinned. Consider "
                "'light_independent' or 'symmetric_split' for batch pairing.",
                UserWarning,
                stacklevel=2,
            )

        partner_idx = self._sample_partners(batch_labels)
        counts = counts.float()
        partner_counts = counts[partner_idx]

        if self.view_mode == "symmetric_split":
            raw_a = self._binomial_thin(counts, p=0.5)
            raw_b = self._binomial_thin(partner_counts, p=0.5)

        elif self.view_mode == "asymmetric":
            raw_a = counts.clone()
            raw_b = self._binomial_thin(partner_counts, p=self.thinning_p)

        elif self.view_mode == "light_independent":
            raw_a = self._binomial_thin(counts, p=self.thinning_p_light)
            raw_b = self._binomial_thin(partner_counts, p=self.thinning_p_light)

        else:  # pragma: no cover
            raise RuntimeError(f"Unhandled view_mode: {self.view_mode}")

        view_a = self._apply_mask(self._normalize(raw_a))
        view_b = self._apply_mask(self._normalize(raw_b))
        return view_a, view_b

    def _sample_partners(self, batch_labels: torch.Tensor) -> torch.Tensor:
        """Return partner index for each cell (same batch, random permutation)."""
        n = len(batch_labels)
        partner_idx = torch.arange(n)
        rng = np.random.default_rng(self.seed)
        for b in batch_labels.unique():
            idx = (batch_labels == b).nonzero(as_tuple=True)[0].numpy()
            if len(idx) < 2:
                continue
            shuffled = rng.permutation(idx)
            partner_idx[idx] = torch.tensor(shuffled, dtype=torch.long)
        return partner_idx
