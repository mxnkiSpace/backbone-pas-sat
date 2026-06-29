# Convierte un CNF (DIMACS) en el grafo bipartito variable-cláusula que
# espera la GNN.
#
# Diferencias respecto al original:
#   - Solo inferencia: se elimina toda la rama de labels de backbone (y).
#   - Sin dependencia `xtract`: la descompresión usa lzma/gzip/bz2 de la
#     stdlib, y además se acepta CNF plano (.cnf) — NeuroBack solo recibía
#     archivos comprimidos del cluster.
#   - Sin multiprocessing ni round-trip a disco (.pt): se devuelven objetos
#     `Data` en memoria.
#
# Origen: NeuroBack (Wang et al., ICLR 2024). MIT License,
# Copyright (c) 2024 wenxiwang. Ver LICENSES/NEUROBACK-LICENSE.

import bz2
import gzip
import lzma
import time

import torch
from torch_geometric.data import Data


class DisJointSets:
    def __init__(self, N):
        self.N = N
        self._parents = [node for node in range(N)]
        self._ranks = [1 for _ in range(N)]

        self._edges = []

    def get_wcc(self):
        wcc = {}
        for node in range(self.N):
            root = self.find(node)
            if root not in wcc:
                wcc[root] = set()
                wcc[root].add(root)
            wcc[root].add(node)

        wcc_edges = {}
        for n1, n2, attr in self._edges:
            r = self.find(n1)
            assert(n2 in wcc[r])

            if r not in wcc_edges:
                wcc_edges[r] = set()
            wcc_edges[r].add((n1, n2, tuple(attr)))

        return wcc, wcc_edges

    def find(self, u):
        assert(u < self.N)

        while u != self._parents[u]:
            self._parents[u] = self._parents[self._parents[u]]
            u = self._parents[u]
        return u

    def union(self, u, v, attr):
        assert(u < self.N and v < self.N)

        self._edges.append((u, v, attr))

        # Union by rank optimization
        root_u, root_v = self.find(u), self.find(v)
        if root_u == root_v:
            return True

        if self._ranks[root_u] > self._ranks[root_v]:
            self._parents[root_v] = root_u
        elif self._ranks[root_v] > self._ranks[root_u]:
            self._parents[root_u] = root_v
        else:
            self._parents[root_u] = root_v
            self._ranks[root_v] += 1
        return False


def _read_cnf_lines(cnf_file_path):
    """Lee un CNF (plano o comprimido .xz/.lzma/.gz/.bz2) como lista de líneas."""
    if cnf_file_path.endswith((".xz", ".lzma")):
        opener = lambda: lzma.open(cnf_file_path, "rt")
    elif cnf_file_path.endswith(".gz"):
        opener = lambda: gzip.open(cnf_file_path, "rt")
    elif cnf_file_path.endswith(".bz2"):
        opener = lambda: bz2.open(cnf_file_path, "rt")
    else:
        opener = lambda: open(cnf_file_path, "r")
    with opener() as f:
        return f.readlines()


def cnf_to_bipartite(cnf_file_path, timelim=1000):
    """CNF -> lista de objetos PyG `Data` (uno por componente débilmente conexa).

    Réplica solo-inferencia de `cnf_to_pt_bipartite` de NeuroBack.
    """
    start_time = time.time()

    lines = _read_cnf_lines(cnf_file_path)

    # --- nodos variable: en orden de primera aparición ---
    X = []
    v2n = {}
    var_num = 0
    for line in lines:
        if time.time() - start_time > timelim:
            print("warning: timeout while reading cnf")
            return None

        line = line.strip()
        if len(line) == 0:
            continue

        fe = line[0]
        if fe == "c" or fe == "p":
            continue
        else:
            lit_lst = [int(lit) for lit in line.split()[:-1]]
            for lit in lit_lst:
                var = abs(lit)
                if var not in v2n:
                    v2n[var] = len(X)
                    X.append([1])
                    var_num += 1

    # --- nodos cláusula + aristas variable->cláusula ---
    edge_index = []
    edge_attr = []
    for line in lines:
        if time.time() - start_time > timelim:
            print("warning: timeout while reading cnf")
            return None

        line = line.strip()
        if len(line) == 0:
            continue

        fe = line[0]
        if fe == "c" or fe == "p":
            continue
        else:
            lit_lst = [int(lit) for lit in line.split()[:-1]]

            cla_node_id = len(X)
            X.append([-1])  # nodo cláusula

            for lit in lit_lst:
                var = abs(lit)
                var_node_id = v2n[var]

                # solo se guarda la arista dirigida; se hace bidireccional
                # en tiempo de predicción (ver predict.py).
                edge_index.append([var_node_id, cla_node_id])

                if lit > 0:
                    edge_attr.append([1])   # literal positivo
                else:
                    assert(lit < 0)
                    edge_attr.append([-1])  # literal negativo

    assert(len(edge_index) == len(edge_attr))

    # --- componentes débilmente conexas ---
    ds = DisJointSets(len(X))
    for idx, edge in enumerate(edge_index):
        if time.time() - start_time > timelim:
            print("warning: timeout while constructing disjoint sets")
            return None
        from_node, to_node = edge[0], edge[1]
        ds.union(from_node, to_node, edge_attr[idx])
    wcc, wcc_edges = ds.get_wcc()
    del ds

    assert(len(wcc) > 0 and len(wcc_edges) > 0)

    if time.time() - start_time > timelim:
        print("warning: timeout after solving wcc")
        return None

    data_lst = []
    if len(wcc) == 1:
        # añadir nodo raíz conectado a todas las cláusulas
        root_node = len(X)
        for clause_node in range(var_num, len(X)):
            edge_index.append([root_node, clause_node])
            edge_attr.append([0])
        X.append([0])

        X = torch.tensor(X, dtype=torch.int8)
        edge_index = torch.tensor(edge_index, dtype=torch.int32)
        edge_attr = torch.tensor(edge_attr, dtype=torch.int8)

        n2v = [-1 for _ in range(len(v2n))]
        for v, n in v2n.items():
            n2v[n] = v
        assert(all(e != -1 for e in n2v))
        n2v = torch.tensor(n2v, dtype=torch.int32)

        data = Data(x=X, n2v=n2v, edge_index=edge_index.t().contiguous(), edge_attr=edge_attr)
        data_lst.append(data)
    else:
        for root, c in wcc.items():
            if time.time() - start_time > timelim:
                print("warning: timeout while enumerating wcc")
                return None

            if len(c) == 1:
                continue

            c = sorted(list(c))

            old_n2new_n = {}
            for i, n in enumerate(c):
                old_n2new_n[n] = i

            var_node_cnt = 0
            for n in c:
                if n < var_num:
                    var_node_cnt += 1
            X_sub = [X[n] for n in c]

            n2v_sub = [-1 for _ in range(var_node_cnt)]
            for v, n in v2n.items():
                if n in old_n2new_n:
                    n2v_sub[old_n2new_n[n]] = v

            edge_index_sub = []
            edge_attr_sub = []

            edges = wcc_edges[root]
            for edge in edges:
                assert(edge[0] in old_n2new_n and edge[1] in old_n2new_n)
                node_a = old_n2new_n[edge[0]]
                node_b = old_n2new_n[edge[1]]
                attr = list(edge[2])

                edge_index_sub.append([node_a, node_b])
                edge_attr_sub.append(attr)

            if len(X_sub) <= 2:
                continue

            # añadir nodo raíz
            root_node = len(X_sub)
            for clause_node in range(var_node_cnt, len(X_sub)):
                edge_index_sub.append([root_node, clause_node])
                edge_attr_sub.append([0])
            X_sub.append([0])

            X_sub = torch.tensor(X_sub, dtype=torch.int8)
            edge_index_sub = torch.tensor(edge_index_sub, dtype=torch.int32)
            edge_attr_sub = torch.tensor(edge_attr_sub, dtype=torch.int8)
            n2v_sub = torch.tensor(n2v_sub, dtype=torch.int32)

            data = Data(x=X_sub, n2v=n2v_sub, edge_index=edge_index_sub.t().contiguous(), edge_attr=edge_attr_sub)
            data_lst.append(data)

    if len(data_lst) == 0:
        print(f"warning: no data object in the data_lst: {cnf_file_path}", flush=True)

    return data_lst
