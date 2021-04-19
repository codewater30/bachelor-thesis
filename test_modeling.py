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
    def setUp(self):
        n_points_per_cluster = 250
        np.random.seed(0)
        C1 = np.zeros((n_points_per_cluster, 3))
        C2 = np.zeros((n_points_per_cluster, 3))
        C3 = np.zeros((n_points_per_cluster, 3))
        C4 = np.zeros((n_points_per_cluster, 3))
        C5 = np.zeros((n_points_per_cluster, 3))
        C6 = np.zeros((n_points_per_cluster, 3))
        C1[:,1:3] = ([-5, -2] + .8 * np.random.randn(n_points_per_cluster, 2))
        C2[:, 1:3] = ([4, -1] + .1 * np.random.randn(n_points_per_cluster, 2))
        C3[:, 1:3] = ([0, -2] + .2 * np.random.randn(n_points_per_cluster, 2))
        C4[:,1:3] = ([-2, 3] + .3 * np.random.randn(n_points_per_cluster, 2))
        C5[:,1:3] = ([3, -2] + 1.6 * np.random.randn(n_points_per_cluster, 2))
        C6[:,1:3] = ([5, 6] + 2 * np.random.randn(n_points_per_cluster, 2))
        X = np.vstack((C1[:, 1:3], C2[:, 1:3], C3[:, 1:3], C4[:, 1:3], C5[:, 1:3], C6[:, 1:3]))

        clust = OPTICS(min_samples=50, xi=.05, min_cluster_size=.05)
        # Run the fit
        clust.fit(X)
        self.tbhg = modeling.TBH()
        self.tbhg.optics = clust
        self.tbhg.locH = (C1, C2, C3, C4, C5, C6)
        # self.tbhg = TBHG(clust)
        pass
    
    @ut.SkipTest
    def test_build_tree(self):
        def traverse(cnode, l):
            if not cnode:
                return l
            l.append(cnode.cluster)
            if cnode.children:
                for c in cnode.children:
                    traverse(c, l)
            
        
        # tbhg = modeling.TBHG()
        # optics = OPTICS_()
        # optics.cluster_hierarchy_ = [[0,4], [6,8], [5,8], [8,10], [0,10]]
        # optics.ordering_ = [x for x in range(11)]
        # tbhg.optics = optics
        # root = tbhg._buildHTree()
        root = self.tbhg._build_tree()
        
        l = []
        traverse(root, l)

        self.assertTrue(np.array_equal(l[::-1], self.tbhg.optics.cluster_hierarchy_))

    @ut.SkipTest    
    def test_build_hierarchy(self):
        tbhg = modeling.TBH()
        optics = OPTICS_()
        optics.cluster_hierarchy_ = [[0,4], [6,8], [5,8], [8,10], [0,10]]
        optics.ordering_ = [x for x in range(11)]
        tbhg.optics = optics
        r = tbhg._build_tree()
        h = tbhg._build_hierarchy(r)
        for i in h:
            print(i)
    # @ut.SkipTest
    def test_build_graph(self):
        r = self.tbhg._build_tree()
        self.tbhg.hierarchy = self.tbhg._build_hierarchy(r)
        self.tbhg._build_graph()
        
        print("hello")

@ut.SkipTest
class TestDetectStayPonts(ut.TestCase):
    def test_detect_staypoints(self):
        trace = np.loadtxt("data_NCSU/1.trace", delimiter=" ")
        
if __name__ == '__main__':
    ut.main()