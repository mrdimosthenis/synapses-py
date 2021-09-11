import unittest

from synapses_py import stats


class TestStatistics(unittest.TestCase):

    def test_rootMeanSquareError(self):
        expectedWithOutputValuesIterator = iter([
            ([0.0, 0.0, 1.0], [0.0, 0.0, 1.0]),
            ([0.0, 0.0, 1.0], [0.0, 1.0, 1.0])
        ])
        self.assertEqual(
            0.7071067811865476,
            stats.rmse(expectedWithOutputValuesIterator)
        )

    def test_score(self):
        expectedWithOutputValuesIterator = iter([
            ([0.0, 0.0, 1.0], [0.0, 0.1, 0.9]),
            ([0.0, 1.0, 0.0], [0.8, 0.2, 0.0]),
            ([1.0, 0.0, 0.0], [0.7, 0.1, 0.2]),
            ([1.0, 0.0, 0.0], [0.3, 0.3, 0.4]),
            ([0.0, 0.0, 1.0], [0.2, 0.2, 0.6])
        ])
        self.assertEqual(
            0.6,
            stats.score(expectedWithOutputValuesIterator)
        )


if __name__ == '__main__':
    unittest.main()
