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

