import numpy as np

from sklearn.cluster import OPTICS
from modeling import TBH
from modeling import CNode

import modeling
def collect_locations(ithgen, cluster: CNode):
    if ithgen == 0:
        return []
    children = [cluster]
    for i in range(ithgen):
        temp = []
        for c in children:
            temp.extend(c.children)
            if not temp:
                break
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

def inference(tbh: TBH, times=3):
    a = dict()
    for i, level in enumerate(tbh.hierarchy):
        for cluster in level:
            for j in range(1, len(tbh.hierarchy)-i):
                descendants = collect_locations(j, cluster)
                if not descendants:
                    break
                matrix = build_matrix(descendants, tbh.locH)
                a_ij, h_ij = HITS_inference(matrix, times) #h_ij用不上
                for clu, score in zip(descendants, a_ij):
                    a[clu, i] = score
    return a

