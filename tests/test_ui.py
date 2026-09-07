import unittest

from qlyraxis.ui.app import scale_series


class ChartScalingTests(unittest.TestCase):
    def test_series_fits_requested_bounds(self) -> None:
        points = scale_series([0, 5, 10], width=200, height=100, padding=10)
        self.assertEqual(len(points), 3)
        self.assertEqual(points[0][0], 10)
        self.assertEqual(points[-1][0], 190)
        self.assertTrue(all(10 <= x <= 190 and 10 <= y <= 90 for x, y in points))

    def test_empty_or_impossible_chart_returns_no_points(self) -> None:
        self.assertEqual(scale_series([], 100, 100), [])
        self.assertEqual(scale_series([1], 10, 10, padding=6), [])


if __name__ == "__main__":
    unittest.main()
