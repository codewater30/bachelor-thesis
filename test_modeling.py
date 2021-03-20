import unittest as ut
import modeling
from sklearn.cluster import OPTICS
import numpy as np
"""
1. test cnode
2. test HTBG

"""
@ut.SkipTest
class TestCNode(ut.TestCase):
    def test_contain1(self):
        subject = [5, 10]
        testCase = [
            ([1, 3], False),
            ([1, 5], False),
            ([3,6], False),
            ([5,6], True),
            ([5, 10], True),
            ([6, 8], True),
            ([8, 11], False),
            ([10, 12], True),
            ([11, 13], False)
        ]

        cnode = modeling.CNode(subject)
        for i, case in enumerate(testCase):
            with self.subTest(i=i):
                self.assertEqual(case[0] in cnode, case[1], "{}".format(case[0]))

    def test_contain2(self):
        subject = [5, 10]
        testCase = [
            (3, False),
            (5, True),
            (6, True),
            (10, True),
            (11, False)
        ]

        cnode = modeling.CNode(subject)
        for i, case in enumerate(testCase):
            with self.subTest(i=i):
                self.assertEqual(case[0] in cnode, case[1], "{}".format(case[0]))
    
class OPTICS_:
    def __init__(self):
        pass

class TestTBHG(ut.TestCase):
    # def setUp(self):
    #     np.random.seed(0)
    #     n_points_per_cluster = 25
    #     C1 = [-5, -2] + .8 * np.random.randn(n_points_per_cluster, 2)
    #     C2 = [4, -1] + .1 * np.random.randn(n_points_per_cluster, 2)
    #     C3 = [1, -2] + .2 * np.random.randn(n_points_per_cluster, 2)
    #     C4 = [-2, 3] + .3 * np.random.randn(n_points_per_cluster, 2)
    #     C5 = [3, -2] + 1.6 * np.random.randn(n_points_per_cluster, 2)
    #     C6 = [5, 6] + 2 * np.random.randn(n_points_per_cluster, 2)
    #     X = np.vstack((C1, C2, C3, C4, C5, C6))

    #     clust = OPTICS(min_samples=50, xi=.05, min_cluster_size=.05)

    #     # Run the fit
    #     clust.fit(X)

    #     self.clust = clust
    #     self.x = X

    #     self.tbhg = modeling.TBHG()
    #     self.tbhg.optics = self.optics
    #     self.tbhg.locH = X

    
    @ut.SkipTest
    def test_buildHTree(self):
        def traverse(cnode, l):
            if not cnode:
                return l
            l.append(cnode.cluster)
            if cnode.children:
                for c in cnode.children:
                    traverse(c, l)
            
        
        tbhg = modeling.TBHG()
        optics = OPTICS_()
        optics.cluster_hierarchy_ = [[0,4], [6,8], [5,8], [8,10], [0,10]]
        optics.ordering_ = [x for x in range(11)]
        tbhg.optics = optics
        root = tbhg._buildHTree()
        l = []
        traverse(root, l)

        self.assertListEqual(l[::-1], optics.cluster_hierarchy_ )
        
    def test_buildHierarchy(self):
        tbhg = modeling.TBHG()
        optics = OPTICS_()
        optics.cluster_hierarchy_ = [[0,4], [6,8], [5,8], [8,10], [0,10]]
        optics.ordering_ = [x for x in range(11)]
        tbhg.optics = optics
        r = tbhg._buildHTree()
        h = tbhg._buildHierarchy(r)
        for i in h:
            print("\n")
            for j in i:
                print(j.cluster, end="")

if __name__ == '__main__':
    ut.main()