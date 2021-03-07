"""location history modeling module

Todo:
    * smiplify styPts-related works
    * add heiarchical tree
        *add interface for inclusivity check at buildHTree()
    
"""
import functools
import numpy as np
class CNode:
    def __init__(self, cluster):
        self.children = set()
        self.neighbors = set()

    def isIn(self, sp: StayPoint) -> bool:
        pass
    
    def addChild(self, c):
        self.children.add(c)

    def addNeighbor(self, cnode):
        self.neighbors.add(cnode)
    
    def __contains__(self, item: tuple):
        if item[0] >= self.cluster[0] and item[0] <= self.cluster[1]:
            if item[1] >= self.cluster[0] and item[1] <= self.cluster[1]:
                return True
        return False
    
    def __contains__(self, item: int):
        if item >= self.cluster[0] and item <= self.cluster[1]:
            return True
        return False


class TBHG:
    def __init__(self, optics: OPTICS, locH):
        self.optics = optics
        self.hierarchy = self.__buildHierarchy(self.__buildHTree(self))
        self.__buildGraph(self)

    def __buildHierarchy(self, r: CNode):
        h = []   
        level = [r]
        while level:
            h.append(level)
            children = []
            for c in level:
                children.extend(c.children)
            level = children
        return h


    


    def __buildGraph(self):
        """build graph on a collection of SP clusters
        """
        #** compute mapping: locH -> ordering_
        ordering = self.optics.ordering_
        locH = np.ndarray.flatten(self.locH)
        
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

    def __buildHTree(self, optics: OPTICS):
        #**
        cIter = iter(optics.cluster_hierarchy[::-1])
        r = CNode([0, len(optics.ordering_)])
        try:
            cn = CNode(next(cIter))
            while True:
                while cn in r:
                    r.addChild(cn)
                    cn = self.__buildHTree_helper(self, cn, cIter)
                # for debug    
                else:
                    print("false")
        except StopIteration:
            return r

    def __buildHTree_helper(self, r, cIter):
        while True:
            cn = CNode(next(cIter))
            while cn in r:
                r.addChild(cn)
                cn = self.__buildHTree_helper(self, cn, cIter)
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

    def distance(a, b):
       return np.linalg.norm(a - b)

    while i < l - 1:
        j = i + 1
        f = False
        while j < l:
            dist = np.linalg.norm(traj[i, 1:3], traj[j, 1:3])
            if dist < dThresh:
                j += 1
                flag = True
            else:
                break
        
        if traj[j-1, 0] - traj[i, 0] > tThresh and flag:
            s.update(range(i, j))

            if i == j - 1:
                styPt = np.average(traj[list(s), ...])
                styPts.append(s)
                s.clear()
        i += 1
    return np.array(styPts)

def buildHTree(t, cIter):
    try:
        while True:
            cn = CNode(next(cIter))
            if cn in t:
                t.addChild(cn)
                returned = buildHTree(cn, cIter)
                while returned in t:
                    t.addChild(returned)
                    returned = buildHTree(returned, cIter)
                else:
                    if returned == None:
                        return t
                    else: 
                        return returned
            else:
                return cn
    except StopIteration:
        return None

def buildGraph(g, locH):
    """build graph on a collection of SP clusters
    """
    c = g.getCNode(locH.getStart())   #last cluster node

    for s in locH:
        ci = g.getCNode(s)
        if c is not ci:
            #build edge
            c.addNeighbors(ci)
        c = ci
    
    return g
