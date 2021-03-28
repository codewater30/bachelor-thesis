import numpy as np

from modeling import TBHG
from modeling import CNode

def collect_locations(searchDepth, cluster: CNode):
    if searchDepth == 0:
        return []
    children = [cluster]
    for i in range(searchDepth):
        assert children
        temp = []
        for c in children:
            temp.extend(c.children)
        children = temp

    return children

def build_matrix(clusters, locH):
    #init matrix 
    matrix = np.zeros((len(locH), len(clusters)), dtype=np.int32)

    for i, c in enumerate(clusters):
        vector = [c.visits[u] for u in range(len(locH))]
        matrix[:, i] = vector

    return matrix
        
def HITS_inference(matrix: np.ndarray, times):
    a = np.ones((matrix.shape[1]), dtype=np.int32)
    h = np.ones((matrix.shape[0]), dtype=np.int32)

    aMatrix = np.dot(matrix.T, matrix)
    bMatrix = np.dot(matrix, matrix.T)

    a = np.linalg.matrix_power(aMatrix, times).dot(a)
    h = np.linalg.matrix_power(bMatrix, times).dot(h)
    return h, a