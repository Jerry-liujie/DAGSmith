from dataclasses import dataclass, field
from typing import Dict, Set, Iterable, Optional
from graphviz import Digraph

@dataclass
class RefactorSubgraph:
    """
    A refactor region extracted from the DAG.

    Attributes
    ----------
    nodes:
        All node uids in this subgraph (including LCAs and component nodes).
    roots:
        The lowest common ancestors (LCAs) for the component.
    edges:
        Internal edges restricted to this subgraph: uid -> children within `nodes`.
    has_external_children:
        uid -> True if the node has at least one child outside this subgraph.
    component_nodes:
        The original target nodes that triggered the extraction (subset of nodes).
    """
    nodes: Set[str]
    roots: Set[str]
    edges: Dict[str, Set[str]]
    has_external_children: Dict[str, bool]
    component_nodes: Set[str] = field(default_factory=set)
    
    @property
    def size(self) -> int:
        """Number of nodes in this subgraph."""
        return len(self.nodes)
    
    def visualize(
        self,
        filename: str = "refactor_subgraph",
        view: bool = False,
        format: str = "png",
    ) -> None:
        """
        Render a RefactorSubgraph as an image using graphviz.

        Parameters
        ----------
        filename :
            Base filename (without extension) for the output file.
        view :
            If True, open the rendered file after creation.
        format :
            Output format: 'png', 'pdf', 'svg', etc.
        """

        dot = Digraph(name="RefactorSubgraph", format=format)
        dot.attr(rankdir="TB")  # or 'LR' for left-to-right
        dot.attr("node", fontname="Helvetica")

        # Add nodes with styling
        for n in self.nodes:
            attrs = {}

            # label
            attrs["label"] = n

            # roots (LCAs) as bold boxes
            if n in self.roots:
                attrs["shape"] = "box"
                attrs["style"] = "bold"

            # nodes with external children as dashed
            if self.has_external_children.get(n, False):
                # if root already has style=bold, combine styles
                style = attrs.get("style")
                if style:
                    attrs["style"] = style + ",dashed"
                else:
                    attrs["style"] = "dashed"

            dot.node(n, **attrs)

        # Add edges
        for parent, children in self.edges.items():
            for child in children:
                dot.edge(parent, child)

        # Render to file
        dot.render(filename, view=view, cleanup=True)
        

    
class SubgraphAnalyzer:
    
    parents: Dict[str, Set[str]]
    children: Dict[str, Set[str]]

    # --- internal utilities ---
    def __init__(
        self,
        parents: Optional[Dict[str, Set[str]]] = None,
        children: Optional[Dict[str, Set[str]]] = None,
    ) -> None:
        # You can also assign these later:
        #   analyzer.parents = ...
        #   analyzer.children = ...
        self.parents: Dict[str, Set[str]] = parents or {}
        self.children: Dict[str, Set[str]] = children or {}

    def _ancestors_with_self(self, uid: str) -> Set[str]:
        """Return all ancestors of uid plus uid itself."""
        visited: Set[str] = {uid}
        stack = [uid]
        while stack:
            cur = stack.pop()
            for p in self.parents.get(cur, ()):
                if p not in visited:
                    visited.add(p)
                    stack.append(p)
        return visited

    def _has_descendant_in_set(self, start: str, target_set: Set[str]) -> bool:
        """Check if 'start' has any descendant in 'target_set' (via children edges)."""
        stack = list(self.children.get(start, ()))
        visited: Set[str] = set()
        while stack:
            cur = stack.pop()
            if cur in visited:
                continue
            visited.add(cur)
            if cur in target_set:
                return True
            stack.extend(self.children.get(cur, ()))
        return False

    def _lcas_for_component(self, component_uids: Iterable[str]) -> Set[str]:
        """Compute lowest common ancestors for all uids in the component."""
        component_uids = list(component_uids)
        if not component_uids:
            return set()

        # 1) all common ancestors (including each uid itself)
        anc_sets = [self._ancestors_with_self(uid) for uid in component_uids]
        common_anc = set.intersection(*anc_sets)
        if not common_anc:
            # no shared ancestor: treat individual nodes as roots
            return set(component_uids)

        # 2) keep only the 'lowest' ones: those that do NOT have another
        #    common ancestor as a descendant.
        lcas: Set[str] = set()
        for cand in common_anc:
            if not self._has_descendant_in_set(cand, common_anc - {cand}):
                lcas.add(cand)

        # fallback: if filtering somehow removed all, use common_anc
        return lcas or common_anc

    # --- main helper: extract subgraph for refactoring ---        
    
    def extract_refactor_subgraph(
        self,
        component_uids: Iterable[str],
    ) -> RefactorSubgraph:
        """
        Given a set of target nodes (component_uids), find their LCA root(s), then
        return the subgraph consisting only of nodes that lie on some path
        from an LCA to a component node:  LCA ->* v ->* component.

        Also mark nodes that have children outside this subgraph.
        """
        comp_set: Set[str] = set(component_uids)
        if not comp_set:
            return RefactorSubgraph(nodes=set(), roots=set(), edges={}, has_external_children={})

        lcas: Set[str] = set(self._lcas_for_component(comp_set))

        # ---- 1) Ancestor closure of component nodes (all nodes that can reach a component) ----
        anc: Set[str] = set()
        stack = list(comp_set)
        while stack:
            cur = stack.pop()
            if cur in anc:
                continue
            anc.add(cur)
            for p in self.parents.get(cur, ()):
                stack.append(p)

        # Make sure LCAs are present (in case _lcas_for_component returns nodes not in anc due to bugs upstream)
        anc |= lcas

        # ---- 2) Keep only nodes that are reachable from LCAs while staying inside anc ----
        if lcas:
            down: Set[str] = set()
            stack = list(lcas)
            while stack:
                cur = stack.pop()
                if cur in down:
                    continue
                if cur not in anc:
                    continue
                down.add(cur)
                for ch in self.children.get(cur, ()):
                    if ch in anc:
                        stack.append(ch)

            sub_nodes: Set[str] = anc & down
        else:
            # No LCAs: fall back to "internal roots" of the ancestor closure.
            # (This prevents weird behavior while still returning a coherent subgraph.)
            sub_nodes = anc
            lcas = {n for n in sub_nodes if not (set(self.parents.get(n, ())) & sub_nodes)}

        # Sanity: component nodes should remain included; if not, your LCA routine is inconsistent.
        missing = comp_set - sub_nodes
        if missing:
            raise ValueError(
                f"LCA set does not reach all component nodes. Missing from subgraph: {sorted(missing)[:10]}"
            )

        # Optional cleanup: ensure roots are actual roots *within the subgraph*
        roots = {r for r in lcas if not (set(self.parents.get(r, ())) & sub_nodes)}

        # ---- 3) Build internal edges and external-child flags ----
        edges: Dict[str, Set[str]] = {}
        has_external_children: Dict[str, bool] = {}

        for n in sub_nodes:
            children = set(self.children.get(n, ()))
            internal = children & sub_nodes
            edges[n] = internal
            has_external_children[n] = bool(children - sub_nodes)

        return RefactorSubgraph(
            nodes=sub_nodes,
            roots=roots,
            edges=edges,
            has_external_children=has_external_children,
        )

    # --- BFS-based neighborhood extraction ---

    def _bfs_neighborhood(
        self,
        seeds: Set[str],
        adjacency: Dict[str, Set[str]],
        max_depth: int,
    ) -> Dict[str, int]:
        """BFS from seeds up to max_depth hops. Returns {node: min_depth}."""
        depth_map: Dict[str, int] = {}
        frontier = list(seeds)
        for s in frontier:
            depth_map[s] = 0
        for d in range(1, max_depth + 1):
            next_frontier = []
            for u in frontier:
                for v in adjacency.get(u, ()):
                    if v not in depth_map:
                        depth_map[v] = d
                        next_frontier.append(v)
            frontier = next_frontier
        return depth_map

    def extract_neighborhood_subgraph(
        self,
        component_uids: Iterable[str],
        upstream_depth: int = 1,
        downstream_depth: int = 1,
        include_siblings: bool = True,
        max_nodes: int = 60,
    ) -> RefactorSubgraph:
        """
        Extract a subgraph by BFS expansion from component nodes.

        Unlike extract_refactor_subgraph (LCA-based), this method:
        - Expands both upstream AND downstream from the component nodes
        - Optionally includes sibling models (other children of component parents)
        - Has a max_nodes knob for size control

        Parameters
        ----------
        component_uids:
            The target nodes to refactor.
        upstream_depth:
            How many hops upstream (via parent edges) to include.
        downstream_depth:
            How many hops downstream (via child edges) to include.
        include_siblings:
            If True, include other children of component nodes' direct parents.
        max_nodes:
            Hard cap on total subgraph size. Prunes furthest nodes first.
        """
        core = set(component_uids)
        if not core:
            return RefactorSubgraph(
                nodes=set(), roots=set(), edges={},
                has_external_children={}, component_nodes=set(),
            )

        # BFS upstream
        up_depths = self._bfs_neighborhood(core, self.parents, upstream_depth)
        # BFS downstream
        down_depths = self._bfs_neighborhood(core, self.children, downstream_depth)

        # Merge depth maps (min distance from any core node in either direction)
        all_depths: Dict[str, int] = {}
        for node, d in up_depths.items():
            all_depths[node] = d
        for node, d in down_depths.items():
            if node not in all_depths or d < all_depths[node]:
                all_depths[node] = d

        # Sibling inclusion: for each direct parent of a core node,
        # add that parent's other children (depth-1 only)
        sibling_nodes: Set[str] = set()
        if include_siblings:
            for c in core:
                for p in self.parents.get(c, ()):
                    if p in all_depths:  # parent is already in subgraph
                        for sib in self.children.get(p, ()):
                            if sib not in all_depths:
                                sibling_nodes.add(sib)
            # Assign siblings a synthetic depth (after direct neighbors)
            for sib in sibling_nodes:
                all_depths[sib] = 2  # same priority as depth-2 nodes

        sub_nodes = set(all_depths.keys())

        # Skip if over budget (caller should handle None)
        if max_nodes and len(sub_nodes) > max_nodes:
            return None

        # Compute roots: nodes in subgraph with no parents in subgraph
        roots = {n for n in sub_nodes if not (set(self.parents.get(n, ())) & sub_nodes)}

        # Build internal edges and external-child flags
        # has_external_children = True means "immutable": the node has children
        # outside the subgraph AND is not a leaf in the subgraph.
        # Leaf nodes are always mutable (LLM preserves leaf semantics).
        edges: Dict[str, Set[str]] = {}
        has_external_children: Dict[str, bool] = {}
        for n in sub_nodes:
            children = set(self.children.get(n, ()))
            internal = children & sub_nodes
            edges[n] = internal
            has_external_children[n] = bool(children - sub_nodes) and bool(internal)

        return RefactorSubgraph(
            nodes=sub_nodes,
            roots=roots,
            edges=edges,
            has_external_children=has_external_children,
            component_nodes=core,
        )

    # --- overlap helper ---

    @staticmethod
    def subgraphs_overlap(g1: RefactorSubgraph, g2: RefactorSubgraph) -> bool:
        """
        Return True if two subgraphs share any node (i.e., cannot be
        refactored independently).
        """
        return bool(g1.nodes & g2.nodes)
    
    
