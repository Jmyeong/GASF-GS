"""CPU checks of triangulation uncertainty and the response-interval loss."""

import unittest

import numpy as np
import torch

from gasf.frir.geometry import project, rays, triangulate
from gasf.frir.loss import ray_objective


class GeometryTests(unittest.TestCase):
    @staticmethod
    def camera(center):
        center = np.asarray(center, dtype=float)
        return dict(
            R=np.eye(3),
            t=-center,
            C=center,
            params=np.array([500.0, 510.0, 320.0, 240.0, 0.01, -0.001, 0.0001, 0.0]),
        )

    def test_projection_roundtrip(self):
        c = self.camera([0, 0, 0])
        uv = np.array([[320.0, 240.0], [400.0, 280.0], [50.0, 60.0]])
        np.testing.assert_allclose(project(rays(uv, c), c), uv, atol=1e-7)

    def test_known_anchor_and_uncertainty(self):
        a = self.camera([0, 0, 0])
        b = self.camera([0.8, 0, 0])
        point = np.array([0.25, 0.1, 3.0])
        result = triangulate(a, b, project(point, a), project(point, b))
        self.assertIsNotNone(result)
        np.testing.assert_allclose(result["xyz"], point, atol=1e-8)
        self.assertTrue(all(x > 0 for x in result["sigma"]))

    def test_smaller_baseline_increases_range_uncertainty(self):
        a = self.camera([0, 0, 0])
        point = np.array([0.25, 0.1, 3.0])
        values = []
        for baseline in [0.8, 0.3]:
            b = self.camera([baseline, 0, 0])
            result = triangulate(a, b, project(point, a), project(point, b))
            self.assertIsNotNone(result)
            values.append(result["sigma"][0])
        self.assertGreater(values[1], values[0])


class RayLossTests(unittest.TestCase):
    def objective(self, depths, alpha, **kwargs):
        return ray_objective(
            torch.tensor([depths], dtype=torch.float64),
            torch.tensor([alpha], dtype=torch.float64),
            torch.tensor([1.0], dtype=torch.float64),
            torch.tensor([0.01], dtype=torch.float64),
            **kwargs,
        )

    def test_inside_interval_only_mass_penalty_remains(self):
        loss, mass, geometry = self.objective([1.005], [0.5])
        self.assertEqual(float(geometry), 0.0)
        self.assertAlmostEqual(float(loss), 0.01 * (1 - float(mass)) ** 2)

    def test_wrong_peaks_cannot_cancel_as_mean_depth(self):
        _, _, geometry = self.objective([0.8, 1.2], [0.2, 0.25])
        self.assertGreater(float(geometry), 0.0)

    def test_mass_coefficient_does_not_change_geometry(self):
        _, a, g = self.objective([0.8, 1.2], [0.2, 0.25], mass_weight=0.0025)
        loss, b, h = self.objective([0.8, 1.2], [0.2, 0.25], mass_weight=0.02)
        torch.testing.assert_close(a, b)
        torch.testing.assert_close(g, h)
        torch.testing.assert_close(loss, g + 0.02 * (1 - a).square())


if __name__ == "__main__":
    unittest.main()
