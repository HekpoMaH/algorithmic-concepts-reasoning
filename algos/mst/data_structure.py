import numpy as np


class QueryUnionFind:
    """
    Data structure for disjoint-set unions operations required for the minimum spanning tree.

    Reference: Veličković, Petar, et al. "Pointer graph networks." arXiv preprint arXiv:2006.06380 (2020).
    https://proceedings.neurips.cc/paper/2020/file/176bf6219855a6eb1f3a30903e34b6fb-Paper.pdf
    Page 5, figure 2
    """

    def __init__(self, n: int, random_state: int = 42):
        """
        :param n: overall number of elements.
        :param random_state: random state
        """
        # random priorities assigned to each element
        self.r = np.random.uniform(0, 1, size=(n))
        # pointers: initially each element points at itself
        self.pi = {u: u for u in range(n)}
        # mask values: mu[i] is set to 0 for only the paths from u and v to
        # their respective roots — no other node's state is changed
        # (i.e., mu[i]=1 for the remaining nodes)
        self.mu = np.zeros(n)
        # adjacency matrix (same meaning of pointers)
        self.Pi = np.zeros((n, n)) + np.diag(np.ones(n))

    def find(self, u: int):
        """
        Find the root of the element "u".
        This function applies path compression: upon calling find(u),
        all nodes on the path from "u" to root[u] will point to root[u].
        This self-organisation substantially reduces future querying time along the path.

        :param u: an element.
        :return: the root.
        """
        if self.pi[u] != u:
            self.pi[u] = self.find(self.pi[u])
        return self.pi[u]

    def union(self, u: int, v: int):
        """
        Perform a union of the sets containing the elements "u" and "v".

        :param u: the first element.
        :param v: the second element.
        """
        x = self.find(u)
        y = self.find(v)
        if x != y:
            # if the roots are different update the pointers!
            # check random priorities: this approach of randomised linking-by-index
            # was recently shown to achieve time complexity of O(a(n)) per operation in expectancy, which is optimal.
            if self.r[x] < self.r[y]:
                # drop the old pointer of root[u]
                self.Pi[x, self.pi[x]] = 0
                # make the old pointer root[u] point to root[v]
                self.pi[x] = y
                self.Pi[x, y] = 1

            else:
                # drop the old pointer of root[v]
                self.Pi[y, self.pi[y]] = 0
                # make the old pointer root[v] point to root[u]
                self.pi[y] = x
                self.Pi[y, x] = 1
    
    def safe_find(self, u: int):

        if self.pi[u] != u:
            return self.safe_find(self.pi[u])
        return self.pi[u]

    def safe_query(self, u: int, v: int):
        return self.safe_find(u) == self.safe_find(v)


    def query_union(self, u: int, v: int):
        """
        Check if the roots of two elements are different: if they are union the two sets.

        :param u: the first element.
        :param v: the second element.
        :return: 1 if the roots were different.
        """
        self.old_p = np.array([p for _, p in self.pi.items()])
        roots_different = 0
        if self.find(u) != self.find(v):
            # if the roots were different, then union the two sets!
            self.union(u, v)
            roots_different = 1

        self.new_p = np.array([p for _, p in self.pi.items()])
        # save modified pointers
        self.mu = (self.old_p == self.new_p).astype(int)
        return roots_different
