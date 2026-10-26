"""Regression tests for the first two-axis mirror-group insertion exception."""

from types import SimpleNamespace
import unittest

from rib_layout_core import load_case
from rib_layout_algorithms.model import Rib
from rib_layout_algorithms.optimization import RibLayoutOptimizer
from rib_layout_algorithms.symmetry import mirror_rib, missing_mirror_partners


class FourRibAdditionTests(unittest.TestCase):
    def setUp(self):
        cfg = load_case(1, quick=True)
        cfg["mirror_symmetry"] = ["x", "y"]
        self.optimizer = RibLayoutOptimizer(SimpleNamespace(), cfg)
        seed = Rib((2.0, 2.0), (6.0, 4.0), 2.0, "seed")
        x = mirror_rib(seed, "x", 20.0, 20.0)
        y = mirror_rib(seed, "y", 20.0, 20.0)
        xy = mirror_rib(x, "y", 20.0, 20.0)
        self.group = [seed, x, y, xy]
        self.center = Rib((0.0, 10.0), (20.0, 10.0), 2.0, "center")

    def complete(self, chosen, active=(), limit=2):
        return self.optimizer._mirror_complete_addition_batch(
            chosen, self.group + [self.center], active, limit
        )

    def test_selection_accepts_first_four_rib_group_and_stops_batch(self):
        scores = {rib.key: 1.0 for rib in self.group}
        scores[self.group[0].key] = 10.0
        scores[self.center.key] = 9.0
        self.optimizer.model.candidate_efficiency = (
            lambda rib, thickness, result: scores[rib.key]
        )
        _, chosen = self.optimizer._select_addition_candidates(
            self.group + [self.center], [], SimpleNamespace(), limit=2
        )
        self.assertEqual(chosen, self.group)
        self.assertFalse(missing_mirror_partners(chosen, ("x", "y"), 20, 20))
        self.assertTrue(any("four-rib exception" in line for line in self.optimizer.log))

    def test_axis_order_does_not_change_exception(self):
        self.optimizer.mirror_axes = ("y", "x")
        chosen = self.complete([self.group[0], self.center])
        self.assertEqual({rib.key for rib in chosen}, {rib.key for rib in self.group})

    def test_later_four_rib_group_is_skipped(self):
        self.assertEqual(self.complete([self.center, self.group[0]]), [self.center])
        self.assertTrue(any("group skipped" in line for line in self.optimizer.log))

    def test_later_four_rib_group_is_skipped_even_when_first_is_covered(self):
        self.assertEqual(self.complete([self.center, self.group[0]], [self.center]), [])

    def test_partly_covered_group_uses_normal_three_rib_batch(self):
        chosen = self.complete([self.group[0], self.center], [self.group[-1]])
        self.assertEqual(chosen, self.group[:-1])
        self.assertFalse(any("four-rib exception" in line for line in self.optimizer.log))

    def test_nonstandard_limit_does_not_gain_exception(self):
        self.assertEqual(self.complete([self.group[0]], limit=1), [])

    def test_single_axis_pair_and_one_plus_pair_are_unchanged(self):
        self.optimizer.mirror_axes = ("x",)
        self.assertEqual(self.complete([self.group[0], self.center]), self.group[:2])
        self.assertEqual(
            self.complete([self.center, self.group[0]]), [self.center] + self.group[:2]
        )


if __name__ == "__main__":
    unittest.main()
