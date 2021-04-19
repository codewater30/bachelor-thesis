"""兴趣点挖掘算法整合

程序内容：
    读取trace files->检测停留点->聚类->构建TBH->兴趣值推导。
输出结果：
    1.坐标图，其中集群按照不同颜色打印
    2.示范性地打印在顶层节点条件下，所有子节点的兴趣值
"""

import util, modeling, inferring
import numpy as np
from sklearn.cluster import OPTICS

if __name__ == '__main__':
    # 读取trace file 并提取stayPoints
    # trace_files中的文件名顺序并不重要
    # locH中的元素即是每个trace中的stayPoints数组，
    # 数组shape为（n,3), n = stayPoints个数
    data_dir = "data_NCSU"
    trace_files = util.get_trace_files(data_dir)
    locH = []
    for trace_file in trace_files:
        trace = np.loadtxt(trace_file) 
        locH.append(modeling.detect_staypoints(trace, 90, 10))

    # 将第一列数据（时间戳）舍去，只将坐标作为聚类依据
    X = [h[:, 1:3] for h in locH]

    #将样本整合成一整串，便于拟合
    samples = np.vstack(tuple(X))

    #拟合并画图
    clust = OPTICS(min_samples=50, xi=.08, min_cluster_size=8)
    clust.fit(samples)
    util.plt_clusters(clust,samples)

    #构建TBH，做兴趣值推导
    tbh = util.build_tbh(clust, X)
    a = inferring.inference(tbh)

    #打印在顶层节点条件下，所有子节点的兴趣值
    l = 1
    for level in tbh.hierarchy[1:]:
        print("Lv", l)
        l += 1
        for c in level:
            print("[{0}, {1}]:{2},".format(*c.cluster, a[c,0]), end=" ")
        print("")