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

class CNode:
    def __init__(self, cluster):
        self.cluster = cluster
        self.neighbors = set()

    def isIn(self, sp: StayPoint) -> bool:
        pass

    def link(self, cnode):
        self.neighbors.add(cnode)
class ClustersGraph:
    def __init__(self, clusters):
        self.clusters = clusters

    def getCNode(self, sp) -> CNode:
        for c in self.clusters:
            if c.isIn(sp):
                return c


def detectStayPoints(traj , tThresh, dThresh):
    """Detect stay points in a trajectory

    Param:
        tThresh: time threshold that a SP(stypnt) must exceed 
        dThresh: distance threshold that limits a SP
    Return:
        styPts: compressed trajectory that is in series of SPs.
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
            c.link(ci)
            pass
        c = ci
    
    return g
