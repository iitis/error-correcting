import unittest
import networkx as nx
import matplotlib.pyplot as plt
from src.data_gen import generate_chimera


class TestChimeraGeneration(unittest.TestCase):
    dim1 = 2
    dim2 = 2
    draw = False

    @classmethod
    def setUpClass(cls):

        cls.g = generate_chimera((cls.dim1, cls.dim2))

    def test_shape(self):
        if self.draw:
            nx.draw(self.g, with_labels=True)
            plt.show()

    def test_nodes(self):
        self.assertEqual(self.g.number_of_nodes(), self.dim1 * self.dim2 * 8)

    def test_degrees(self):
        max_degree = max(self.g.degree, key=lambda x: x[1])[1]
        min_degree = min(self.g.degree, key=lambda x: x[1])[1]
        print(self.g.degree)
        if self.dim1 <= 2 or self.dim2 <= 2:
            self.assertTrue(max_degree == 5)
        else:
            self.assertTrue(max_degree == 6)
        #self.assertTrue(min_degree == 5)






if __name__ == '__main__':
    unittest.main()
