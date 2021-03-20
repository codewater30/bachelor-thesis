"""location history modeling module

Todo:
    * complete heiarchical tree
        *add array building in __buildHTree()
        *sample -> ordering getOrder()
    
"""
import functools
import numpy as np
from sklearn.cluster import OPTICS
import math
class CNode:
    def __init__(self, cluster):
        self.cluster = cluster  # using index
        self.children = list()
        self.neighbors = set()
    
    def addChild(self, c):
        self.children.append(c)

    def addNeighbor(self, cnode):
        self.neighbors.add(cnode)
    
    def __contains__(self, item):
        if type(item) is list:
            if item[0] >= self.cluster[0] and item[0] <= self.cluster[1]:
                if item[1] >= self.cluster[0] and item[1] <= self.cluster[1]:
                    return True
            return False
        if type(item) is int:
            if item >= self.cluster[0] and item <= self.cluster[1]:
                return True
            return False
        
class TBHG:
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
            n = len(l)  
            maxEdgeCount = n*(n-1)
            c = 0

            prev = getCNode(l, orderOfSPs[locH[0]])   
            for s in locH and c < maxEdgeCount:
                temp = getCNode(l, orderOfSPs[s])   
                curr = temp if temp else curr                
                if prev is not curr:
                    #build edge
                    prev.addNeighbor(curr)
                    c += 1
                prev = curr

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


