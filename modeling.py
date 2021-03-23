"""location history modeling module

Todo:
    
"""
import math
from collections import defaultdict

import numpy as np
from sklearn.cluster import OPTICS

class CNode:
    def __init__(self, cluster):
        self.cluster = cluster  # using index
        self.children = list()
        self.neighbors = defaultdict(int)
    
    def addChild(self, c):
        self.children.append(c)

    def addNeighbor(self, cnode):
        self.neighbors[cnode] += 1
    
    def __contains__(self, item):
        if type(item) is list:
            if item[0] >= self.cluster[0] and item[0] <= self.cluster[1]:
                if item[1] >= self.cluster[0] and item[1] <= self.cluster[1]:
                    return True
            return False
        if np.issubdtype(type(item),np.integer):
            if item >= self.cluster[0] and item <= self.cluster[1]:
                return True
            return False
        if type(item) is np.ndarray:
            if item[0] >= self.cluster[0] and item[0] <= self.cluster[1]:
                if item[1] >= self.cluster[0] and item[1] <= self.cluster[1]:
                    return True
            return False
class TBHG:
    def __init__(self, optics, locH):#, optics: OPTICS, locH):
        self.optics = optics
        self.locH = locH
        self.hierarchy = self._buildHierarchy(self._buildHTree())
        self._buildGraph()
        
    def _buildHierarchy(self, r: CNode):
        h = []   
        level = [r]
        while level:
            h.append(level)
            children = []
            for c in level:
                children.extend(c.children)
            level = children
        return h

    def _buildGraph(self):
        """build graph on a collection of SP clusters
        """
        ordering = self.optics.ordering_
        # locH = np.ndarray.flatten(self.locH)
        locH = self.locH 
        orderOfSPs = np.zeros_like(ordering)
        orderOfSPs[ordering] = np.arange(0, ordering.size)

        def getCNode(level, sp):
            for c in level:
                if sp in c:
                    return c
            else:
                return None
                
        for l in self.hierarchy:
            offSet = index = 0
            prev = getCNode(l, orderOfSPs[index])
            for h in locH:
                for i in range(len(h)):
                    index = offSet + i
                    curr = getCNode(l, orderOfSPs[index])   
                    if curr:
                        if prev is not curr:
                            #build edge
                            if prev:
                                prev.addNeighbor(curr)
                            prev = curr
                offSet += len(h)
    
    def _buildHTree(self):
        #**
        ch = self.optics.cluster_hierarchy_
        cIter = iter(ch[::-1])
        r = CNode(next(cIter))
        try:
            cn = CNode(next(cIter))
            while True:
                while cn.cluster in r:
                    r.addChild(cn)
                    cn = self._buildHTree_helper(cn, cIter)
                # for debug    
                else:
                    print("false")
        except StopIteration:
            return r

    def _buildHTree_helper(self, r, cIter):
        while True:
            cn = CNode(next(cIter))
            while cn.cluster in r:
                r.addChild(cn)
                cn = self._buildHTree_helper(cn, cIter)
            else:
                return cn 

def detectStayPoints(traj: np.ndarray, tThresh, dThresh):
    """Detect stay points in a trajectory

    Param:
        traj: trajectory as a numpy array of the shape(n,3), each row as 
            (time, x, y)
        tThresh: time threshold that a SP(stypnt) must exceed 
        dThresh: distance threshold that limits a SP
    Return:
        styPts: series of SPs.
    """
    i = 0
    l = len(traj)
    styPts = []
    s = set()

    while i < l - 1:
        j = i + 1
        flag = False
        while j < l:
            dist = math.hypot(*(traj[i, 1:3] - traj[j, 1:3]))#
            if dist < dThresh:
                j += 1
                flag = True
            else:
                break
        
        if flag and traj[j-1, 0] - traj[i, 0] > tThresh:#
            styPt = np.mean(traj[i:j], axis=0)
            styPts.append(styPt)
            i = j
        else:
            i += 1
    return np.array(styPts)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import os, re
    data_dir = "data_NCSU"
    clust = OPTICS(min_samples=100, xi=.05, min_cluster_size=.05)
    traces_files = [trace for trace in os.listdir(data_dir) if re.match(r'\d+\.trace',trace)]
    locH = []
    for trace_file in traces_files:
        trace = np.loadtxt(os.path.join(data_dir, trace_file)) 
        locH.append(detectStayPoints(trace, 90, 10))

    X = [h[:, 1:3] for h in locH]
    X = np.vstack(tuple(X))
    clust.fit(X)

    n_lables = np.amax(clust.labels_)
    cmap = plt.get_cmap("viridis", n_lables)
    for i in range(n_lables):
        x = X[clust.labels_ == i]
        plt.scatter(x[:, 0], x[:, 1], color=cmap(i), alpha=0.5)
    x = X[clust.labels_ == -1]
    plt.scatter(x[:, 0], x[:, 1], alpha=0.1)
    plt.show()
    plt.close()
    
    tbhg = TBHG(clust, locH)

    print("hello")
