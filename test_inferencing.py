from re import T
from collections import namedtuple
import unittest as ut

import numpy as np
from sklearn.cluster import OPTICS

import modeling
import inferring
import util
class TestInferring(ut.TestCase):
    def test_collect_Loactions(self):
        # TODO: mock optics or use namedtuple
        class OPTICS:
            def __init__(self):
                pass

        mockClusters = [[1,7], [8,15],[16,20],[0,25],[32,40],
                        [30,49],[0,50],[85,98],[80,99],[0,99]
                       ]
        optics = OPTICS()
        optics.cluster_hierarchy_ = mockClusters
        
        r = util.build_tree(optics) 

        f = lambda cnodes: [cnode.cluster for cnode in cnodes]

        actual1 = f(inferring.collect_locations(1, r))
        actual2 = f(inferring.collect_locations(2, r))
        actual3 = f(inferring.collect_locations(3, r))
        actual4 = f(inferring.collect_locations(4, r))
        actual = [actual1, actual2, actual3, actual4]
        expected1 = [[0,50], [80, 99]]
        expected2 = [[0,25], [30, 49], [85,98]]
        expected3 = [[1,7], [8,15], [16,20], [32,40]]
        expected4 = []
        expected = [expected1, expected2, expected3, expected4]
        for i, r, e in zip(range(len(actual)), actual, expected):
            with self.subTest(i=i):
                self.assertCountEqual(r, e, "i")
    
    def test_build_matrix1(self):
        c1 = modeling.CNode()
        c2 = modeling.CNode()
        c3 = modeling.CNode()
        c4 = modeling.CNode()
        c5 = modeling.CNode()
        c6 = modeling.CNode()

        def f(c, u, t):
            c.visits[u] = t
        visits = [
            (c1,0,1), (c1,1,1), (c2,0,1), (c2,1,1), (c3,1,2),
            (c3,2,1), (c4,2,2),(c4,3,1), (c5,3,1), (c5,4,1),
            (c6,4,1)
        ]
        for v in visits:
            f(*v)

        mock_locH = [i for i in range(5)]
        clusters = [c1, c2, c3, c4, c5, c6]
        actual = inferring.build_matrix(clusters, mock_locH)
        expect = np.array([
            [1,1,0,0,0,0],
            [1,1,2,0,0,0], 
            [0,0,1,2,0,0], 
            [0,0,0,1,1,0], 
            [0,0,0,0,1,1]
        ])

        np.testing.assert_equal(expect, actual)

    def test_HIST_inference(self):
        matrix = np.array([
            [1,1,0,0,0,0],
            [1,1,2,0,0,0], 
            [0,0,1,2,0,0], 
            [0,0,0,1,1,0], 
            [0,0,0,0,1,1]
        ])
        actual_a, actual_b = inferring.HITS_inference(matrix, 3)
        expect_a = np.array([228, 722, 609, 223, 53])
        expect_b = np.array([374, 374, 791, 538, 108, 24])
        np.testing.assert_equal(actual_a, expect_a)
        np.testing.assert_equal(actual_b, expect_b)

if __name__ == '__main__':
    ut.main()