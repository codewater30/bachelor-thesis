import numpy as np

from modeling import TBHG
from modeling import CNode

def collectLocation(searchDepth, cluster: CNode):
    children = [cluster]
    for i in range(searchDepth):
        assert children
        temp = []
        for c in children:
            temp.extend(c.children)
        children = temp

    return children

def buildMatrix(clusters: np.ndarray[CNode], locH, ordering):
    hisLens = [len(h) for h in locH]

    #init matrix 
    matrix = np.zeros((len(locH), len(clusters)))


    for i, c in enumerate(clusters):
        start, end = c.cluster
        indices = np.zeros((np.size(locH),))

        #mark occurrences 
        indices[ordering[start:end+1]] = 1

        #count times of visit to the location for each user
        s = e = 0
        vector = []
        for l in hisLens:
            s, e = e, l 
            vector.append(sum(indices[s:e]))

        matrix[:, i] = vector
        
def HITSInference(matrix: np.ndarray, times):
    a = np.ones((matrix.shape[1]))
    h = np.ones((matrix.shape[0]))

    aMatrix = np.dot(matrix.T, matrix)
    bMatrix = np.dot(matrix, matrix.T)
    for i in range(times):
        a = aMatrix.dot(a)
        h = bMatrix.dot(h)
        
    return h, a