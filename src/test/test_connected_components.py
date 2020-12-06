import unittest
import os
import h5py
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pkgutil
from random import randint

import rag_utils as ru
package = ru
for importer, modname, ispkg in pkgutil.iter_modules(package.__path__):
    print("Found submodule %s (is a package: %s)" % (modname, ispkg))



class TestRagUtils(unittest.TestCase):

    def test_cc(self):
        from rag_utils import find_dense_subgraphs
        edges = []
        nodes = []
        bs = 10 #343
        for i in range(bs):
            file = h5py.File("rags_h5/rag_" + str(i) + ".h5", "r")
            edges.append(file["edges"][:])
            nodes.append(np.unique(edges[-1]))

            g = nx.Graph()
            g.add_nodes_from(nodes[-1])
            g.add_edges_from(edges[-1])
            assert nx.is_connected(g)

        ks = [10, 20, 30, 50]
        bsgs, bssgs = find_dense_subgraphs(edges, ks)

        for j in range(0, len(ks)):
            for i in range(bs):
                g = nx.Graph()
                g.add_nodes_from(nodes[i].tolist())
                g.add_edges_from(edges[i].tolist())
                for sg, ssg in zip(bsgs[j*bs + i], bssgs[j*bs + i]):
                    csg = nx.Graph()
                    csg.add_nodes_from(np.unique(sg).tolist())
                    csg.add_edges_from(sg.tolist())
                    assert nx.is_connected(csg)
                    edge_dict = {}
                    for edge, sedge in zip(sg, ssg):
                        assert g.has_edge(edge[0], edge[1])
                        if edge[0] not in edge_dict:
                            edge_dict[edge[0]] = sedge[0]
                        if edge[1] not in edge_dict:
                            edge_dict[edge[1]] = sedge[1]

                        assert edge_dict[edge[0]] == sedge[0] or edge_dict[edge[1]] == sedge[0]
                        assert edge_dict[edge[0]] == sedge[1] or edge_dict[edge[1]] == sedge[1]

        for j in range(0, len(bsgs)):
            for i in np.random.randint(0, len(bsgs[j]), 5):
                sg = bsgs[j][i]
                fig = plt.figure()
                #
                g = nx.Graph()
                g.add_nodes_from(nodes[j%bs].tolist())
                g.add_edges_from(edges[j%bs].tolist())
                pos = nx.spring_layout(g)

                nsg = np.unique(sg)

                nx.draw(g, pos, with_labels=True, node_color='b', edge_color='b')
                nx.draw_networkx_nodes(g, pos, nodelist=nsg.tolist(), node_color='r', node_size=50)
                nx.draw_networkx_edges(g, pos, edgelist=sg.tolist(), edge_color='r', width=1)

                directory = f"graphs_pics/graph_{ks[j // bs]}"
                if not os.path.exists(directory):
                    os.makedirs(directory)

                plt.savefig(directory + "/sub_graph_" + str(i) + ".png")
                plt.clf()
                g.clear()
                plt.close()

    def test_density(self):
        from rag_utils import find_dense_subgraphs
        edges = []
        nodes = []
        bs = 100
        for i in range(bs):
            file = h5py.File("rags_h5/rag_" + str(i) + ".h5", "r")
            edges.append(file["edges"][:])
            nodes.append(np.unique(edges[-1]))

            g = nx.Graph()
            g.add_nodes_from(nodes[-1])
            g.add_edges_from(edges[-1])
            assert nx.is_connected(g)

        ks = [10, 20, 30, 50]
        bsgs, bssgs = find_dense_subgraphs(edges, ks)
        density = 0
        n = 0
        for j in range(len(bsgs)):
            for i in range(len(bsgs[j])):
                for sg in bsgs[j][i]:
                    nsg = np.unique(sg)
                    density += sg.shape[0] / nsg.shape[0]
                    n += 1.0
        print(f"Achieved mean density is: {density/n}")

if __name__ == '__main__':
    unittest.main()
