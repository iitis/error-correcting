import unittest
from src.utils import gauge_transformation_nx, compute_energy_nx
from src.data_gen import generate_chimera


class TestEnvironment(unittest.TestCase):
    def test_gauge(self):
        g = generate_chimera((2, 2))
        transformed = gauge_transformation_nx(g)

        self.assertEqual(compute_energy_nx(g), compute_energy_nx(transformed))


if __name__ == '__main__':
    unittest.main()
