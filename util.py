import os, re
from matplotlib import pyplot as plt
from sklearn.cluster import OPTICS

import numpy as np
import modeling
def plt_clusters(clusters: OPTICS, samples):
    n_lables = np.amax(clusters.labels_)
    cmap = plt.get_cmap("viridis", n_lables)
    for i in range(n_lables):
        x = samples[clusters.labels_ == i]
        plt.scatter(x[:, 0], x[:, 1], color=cmap(i), alpha=0.5)
    x = samples[clusters.labels_ == -1]
    plt.scatter(x[:, 0], x[:, 1], alpha=0.1)
    plt.show()
    plt.close()
    
def get_trace_files(path):
    return [os.path.join(path, trace) for trace in os.listdir(path) if re.match(r'\d+\.trace',trace)]

def print_hierarchy(h):
    for level in h:
        print(level)

def build_hierarchy(optics):
    tbhg = modeling.TBH(optics)
    return tbhg.hierarchy

def build_tree(optics):
    tbhg = modeling.TBH()
    tbhg.optics = optics
    return tbhg._build_tree()

def build_tbhg(optics, locH):
    return modeling.TBH(optics, locH)
