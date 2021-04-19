"""位置历史建模location history modeling module

"""
import math
from collections import defaultdict
from typing import List
import numpy as np
from sklearn.cluster import OPTICS

class CNode:
    """CNode，即cluster Node，集群节点，是作为树形层级结构的节点类。

    Attribute
    ---------
    cluster : ndarray of shape(2,)
        长度为2的numpy数组，是以[start, end]形式所标识的集群，
        表示该集群在样本索引的聚类有序列表中所在的范围。

    children : List[CNode]
        子节点列表。

    visits : defaultdict, key: int, value: int
        用户访问记录，字典类型。
    """
    def __init__(self, cluster=None, children:List=None, neighbors=None, visits=None):
        self.cluster = cluster  # using index
        self.children = children if children else []
        self.neighbors = neighbors if neighbors else defaultdict(lambda: defaultdict(int))
        self.visits = visits if visits else defaultdict(int)
    
    def add_child(self, c):
        self.children.append(c)
    
    #用户路线推荐的图结构，可以忽略。
    def add_neighbor(self, cnode, user):
        self.neighbors[cnode][user] += 1
        
    def add_visit(self, user):
        self.visits[user] += 1

    def __repr__(self):
        return str(self.cluster)

    def __str__(self):
        return str(self.cluster)

    def __eq__(self, other):
        s_0, s_1 = self.cluster[0], self.cluster[1] 
        o_0, o_1 = other.cluster[0], other.cluster[1]
        return s_0 == o_0 and s_1 == o_1 

    def __lt__(self, other):
        s_0, s_1 = self.cluster[0], self.cluster[1] 
        o_0, o_1 = other.cluster[0], other.cluster[1]
        if s_0 == o_0:
            return s_1 < o_1
        return s_0 < o_0

    def __hash__(self):
        return hash(str(self.cluster))

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
class TBH:
    """树型层级(tree-based hierarchy)。

    存储停留点集群之间层级、父子关系的数据结构，是以CNode为节点的树形结构，
    但同时也记录了每层所有节点。

    Attributes
    ----------
    optics : OPTICS
        optics聚类算法类实例，是已经过拟合的实例过，包含构建层级结构
        所需要的集群信息，详见sklearn库官方文档。
    
    hierarchy : List[List[CNode]]
        层级结构，每层节点以升序排列。

    locH : List[ndarray of shape(n, 3)]
        用户位置历史数据集。
    """
    
    def __init__(self, optics=None, locH=None):
        if optics:
            self.optics = optics
            self.hierarchy = self._build_hierarchy(self._build_tree())
            if locH is not None:
                self.locH = locH
                self._build_graph()      
            
    def _build_hierarchy(self, r: CNode):
        """构建层级

        遍历树构建层级。

        Params
        ------
        r : CNode
            树的根节点。

        Returns
        -------
        h : List[List[CNode]]
            所构建的层级。
        """
        h = []   
        level = [r]
        while level:
            h.append(level)
            children = []
            for c in level:
                children.extend(c.children)
            children.sort() #每层节点升序排列
            level = children
        return h

    def _build_graph(self):
        """在树型层级的每一层构建图
        
        注：本函数大部分工作是构建图，但是是用于路线推荐的，
        只有next.add_visit(user)那一行是真正对兴趣点挖掘有用的
        （我写完了才知道论文不用做路线推荐-_-||）。
        """
        ordering = self.optics.ordering_
        locH = self.locH 
        sp2order = np.zeros_like(ordering)
        sp2order[ordering] = np.arange(0, ordering.size)

        def getCNode(level: List[CNode], sp):
            for c in level:
                if sp in c:
                    return c
            else:
                return None
                
        for l in self.hierarchy:
            off_set = index = 0
            for user, h in enumerate(locH):
                curr = getCNode(l, sp2order[index])
                if curr:
                    curr.visits[user] +=1
                for index in range(off_set, off_set+len(h)):
                    next = getCNode(l, sp2order[index])   
                    if next:
                        if curr is not next:
                            #build edge
                            if curr:
                                curr.add_neighbor(next, user) #这一部分可有可无
                            next.add_visit(user)        #这一行最重要
                            curr = next
                off_set += len(h)
    
    def _build_tree(self, cluster_hierarchy=None):
        """根据optics算法的拟合结果构建树

        Params
        ------
        cluster_hierarchy : ndarray of shape (n_clusters, 2)
            详见OPTICS的_cluster_hierarchy属性

        Returns
        -------
        r : CNode
            树的根节点
        """
        if not cluster_hierarchy:
            cluster_hierarchy = self.optics.cluster_hierarchy_
        cIter = iter(cluster_hierarchy[::-1])
        r = CNode(next(cIter))
        try:
            cn = CNode(next(cIter))
            while True:
                while cn.cluster in r:
                    r.add_child(cn)
                    cn = self._build_tree_helper(cn, cIter)
                # for debug    
                else:
                    print("false")
        except StopIteration:
            return r

    def _build_tree_helper(self, r: CNode, cIter):
        while True:
            cn = CNode(next(cIter))
            while cn.cluster in r:
                r.add_child(cn)
                cn = self._build_tree_helper(cn, cIter)
            else:
                return cn 

def detect_staypoints(traj: np.ndarray, tThresh, dThresh):
    """Detect stay points in a trajectory

    文档写的比较早了，当时用的英文，没考虑周全。
    Params
    ------
    traj : 
        trajectory as a numpy array of the shape(n,3), each row as 
        (time, x, y)

    tThresh : float
        time threshold in seconds that a SP(stypnt) must exceed

    dThresh : float
        distance threshold in meters that limits a SP

    Return
    ------
    styPts : ndarray of shape(n, 3), 
        series of Stay Points, n is the number of styPts
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
    import util
    data_dir = "data_NCSU"
    # traces_files = [trace for trace in os.listdir(data_dir) if re.match(r'\d+\.trace',trace)]
    traces_files = util.get_trace_files(data_dir)
    locH = []
    for trace_file in traces_files:
        trace = np.loadtxt(trace_file) 
        locH.append(detect_staypoints(trace, 90, 10))

    X = [h[:, 1:3] for h in locH]
    X = np.vstack(tuple(X))
    clust = OPTICS(min_samples=100, xi=.05, min_cluster_size=.05)
    clust.fit(X)
    util.plt_clusters(clust,X)
    print("hello")
