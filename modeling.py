import functools
class Coordinate:
    def __init__(self, lat, lngt, time) -> None:
        self.lat = lat
        self.lngt = lngt
        self.time = time

class StayPoint:
    def __init__(self, trajectory, tThresh, dThresh) -> None:
        def crdntAdd():
            pass
        self.trajectory = trajectory
        self.tThresh = tThresh
        self.dThresh = dThresh
        self.lat = functools.reduce(crdntAdd, (c.lat for c in trajectory))/len(trajectory)
        self.lngt = functools.reduce(crdntAdd, (c.lngt for c in trajectory))/len(trajectory)

import functools
import numpy as np
class CNode:
    def __init__(self, cluster):
        self.cluster = cluster
        self.childs = set()
        self.neighbors = set()

    def isIn(self, sp: StayPoint) -> bool:
        pass
    
    def addChild(self, c):
        self.childs.add(c)

    def addNeighbors(self, cnode):
        self.neighbors.add(cnode)
class CGraph:
    def __init__(self, clusters):
        self.clusters = clusters

    def getCNode(self, sp) -> CNode:
        for c in self.clusters:
            if c.isIn(sp):
                return c


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
