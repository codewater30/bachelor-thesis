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

def detectStayPoints(traj , tThresh, dThresh):

    """Detect stay points in a trajectory
    Param:
        tThresh: time threshold that a SP(stypnt) must exceed 
        dThresh: distance threshold that limits a SP
    Return:
        styPts: compressed trajectory that is in series of SPs.
    """
    start, i= 0
    styPts = []
    length = len(traj)
    while i < length:
        c = traj[i]
        if c.time - traj[start].time >= tThresh:
            if isInRange(traj[start:i+1], dThresh):
                i += 1
                while i < length and isInRange(traj[start:i+1], dThresh):
                    i += 1
                styPts.append(StayPoint(traj[start: i], tThresh, dThresh))
                start = i
                i -= 1
            else:
                start += 1
                while c.time - traj[start].time >= tThresh:
                    if isInRange(traj[start:i+1], dThresh):
                        styPts.append(StayPoint(traj[start: i+1], tThresh, dThresh))
                        start = i + 1
                        break
                    start += 1
        i += 1
    return styPts            
                        


def isInRange(traj, dThresh) -> bool:
    pass

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

    """
    pass