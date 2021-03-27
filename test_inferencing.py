    @ut.SkipTest
    def test_collectLoactions(self):
        class OPTICS:
            def __init__(self):
                pass

        mockClusters = [[1,7], [8,15],[16,20],[0,25],[32,40],
                        [30,49],[0,50],[85,98],[80,99],[0,99]
                       ]
        tbhg = modeling.TBHG()
        optics = OPTICS()
        optics.cluster_hierarchy_ = mockClusters
        
        tbhg.optics = optics
        r = tbhg._buildHTree()
        h = tbhg._buildHierarchy(r)

        f = lambda cnodes: [cnode.cluster for cnode in cnodes]

        res1 = f(inferencing.collectLocations(1, r))
        res2 = f(inferencing.collectLocations(2, r))
        res3 = f(inferencing.collectLocations(3, r))
        res4 = f(inferencing.collectLocations(4, r))
        res = [res1, res2, res3, res4]
        expected1 = [[0,50], [80, 99]]
        expected2 = [[0,25], [30, 49], [85,98]]
        expected3 = [[1,7], [8,15], [16,20], [32,40]]
        expected4 = []
        expected = [expected1, expected2, expected3, expected4]
        for i, r, e in zip(range(len(res)), res, expected):
            with self.subTest(i=i):
                self.assertCountEqual(r, e, "i")

