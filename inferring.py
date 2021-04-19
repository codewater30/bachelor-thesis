import numpy as np

from sklearn.cluster import OPTICS
from modeling import TBH
from modeling import CNode

def collect_locations(ithgen, cluster: CNode):
    """搜寻并返回cluster的第ithgen代后代节点们

    若ithgen为0或搜寻过程中的某一代为空，则返回空list

    Params
    ------
    ithgen : int
        第ithgen后代
        
    cluster : CNode
        祖先节点

    Returns
    -------
    children : List[CNode]
        cluster的第ithgen代节点的集合,若ithgen为0或搜寻过程中的某一代为空,
        则返回空list
    """
    if ithgen == 0:
        return []
    children = [cluster]
    for i in range(ithgen):
        temp = []
        for c in children:
            temp.extend(c.children)
        if not temp:
            return []
        children = temp

    return children

def build_matrix(clusters, locH):
    """构建集群和用户之间的矩阵
    
    矩阵行为用户，列为地点也就是集群，每一项含义是该行用户对该列地点
    的访问次数，最小为0

    Params
    ------
    clusters : ndarray of shape(2,)
        集群集合

    locH : List
        用户位置历史数据集

    Returns
    -------
    matrix : ndarray of shape(len(locH), len(clusters))
        集群和用户之间的矩阵
    """
    #init matrix 
    
    matrix = np.zeros((len(locH), len(clusters)), dtype=np.int32)

    for i, c in enumerate(clusters): 
        vector = [c.visits[u] for u in range(len(locH))]
        matrix[:, i] = vector

    return matrix
        
def HITS_inference(matrix: np.ndarray, times):
    """基于HITS算法推导用户旅行经验和地点的兴趣值

    依据maxtrix给出的地点与用户之间的访问关系，设置用户旅行经验和
    地点兴趣值初始向量h,a，依据HITS算法计算并返回迭代times次后的h，a

    Params
    ------
    matrix : 用户和地点之间的访问关系矩阵
    times : 迭代次数

    Returns
    -------
    h : ndarray of shape (#matrixRows,)
        代表各个用户旅行经验的向量

    a : ndarray of shape (#matrixColumns,)
        代表各个地点的兴趣值的向量
    """
    
    a = np.ones((matrix.shape[1]), dtype=np.int32)
    h = np.ones((matrix.shape[0]), dtype=np.int32)

    aMatrix = np.dot(matrix.T, matrix)
    bMatrix = np.dot(matrix, matrix.T)

    a = np.linalg.matrix_power(aMatrix, times).dot(a)
    h = np.linalg.matrix_power(bMatrix, times).dot(h)
    return h, a

def inference(tbh: TBH, times=3):
    """地点兴趣值推导模型

    根据传入的树形层级结构，计算每个集群节点在其不同祖先节点指定的
    范围下的兴趣值。

    Params
    ------
    tbhg : TBH
        构建完成的树型层级结构
    times : int
        HITS算法中的迭代次数

    Returns
    -------
    a : defaultdict, key: (CNode, int), value: int
        以(cluster, ascendant-level) 作为key, authority score 作为值的字典
    """
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

