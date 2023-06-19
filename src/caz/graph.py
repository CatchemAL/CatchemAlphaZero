from copy import copy

from graphviz import Digraph

from .solvers.mcts import Node
from .states import State, TMove


def visualize_tree(state: State[TMove], root: Node) -> Digraph:
    dot = Digraph(comment="Game tree")

    def add_node(node: Node, state: State[TMove]):
        node_id = f"{id(node)}"

        # add current node
        dot.node(node_id, label=node.html(state), shape="rectangle")

        # add edges to children
        for child in node.children:
            child_id = f"{id(child)}"
            dot.edge(node_id, child_id)

            # recursively add nodes
            child_state = copy(state)
            child_state.play_move(child.move)
            add_node(child, child_state)

    # start with root node
    add_node(root, state)

    # render graph
    return dot
