import logging
import queue
from nltk.tree import Tree


logger = logging.getLogger(__name__)


def tree_str_post_process(tree_string):
    tree_string = tree_string.replace('-LRB- (', '-LRB- -LRB-').replace('-RRB- )', '-RRB- -RRB-')
    tree_string = tree_string.replace('TRUNC (', 'TRUNC -LRB-').replace('TRUNC )', 'TRUNC -RRB-')
    return tree_string


def tree_from_string(tree_string):
    try:
        s = tree_str_post_process(tree_string)
        tree = Tree.fromstring(s)
    except Exception as e:
        try:
            tree = Tree.fromstring(tree_string)
        except Exception as e:
            logger.error(f'Unable to parse the tree: {tree_string}')
            raise e

    return tree


def padding_leaves_wnum(leaves, tree):
    for i in range(len(leaves)):
        tree[tree.leaf_treeposition(i)] = f'{i}'


def tree_to_leave_pos_node_span_collapse_v3(tree):
    leaves = tree.leaves()
    padding_leaves_wnum(leaves, tree)
    pos_tags = []
    tree_node_lst = []
    spans = []
    queue_tree = queue.Queue()
    queue_tree.put(tree)
    while not queue_tree.empty():
        node = queue_tree.get()
        while len(node) == 1 and isinstance(node[0], Tree):
            node.set_label(node[0].label())
            node[0:] = [c for c in node[0]]
        if len(node) == 1 and isinstance(node[0], str):
            pos_tags.append(node.label())
            continue
        internal_leaves = node.leaves()
        tree_node_lst.append(node)
        _span = [int(internal_leaves[0]), int(internal_leaves[-1])]
        spans.append(_span)

        for c in node:
            if isinstance(c, Tree):
                queue_tree.put(c)
    del queue_tree
    nodes = [x.label() for x in tree_node_lst]
    if len(nodes) == 0:
        nodes = [tree.label()]
        spans = [[0, len(leaves) - 1]]

    return leaves, pos_tags, nodes, spans, tree_node_lst


def leaves2span(in_leaves, leaves):
    # FIXME: this will cause wrong if the phrase repeat!
    query = in_leaves[0]
    start_idx = [i for (y, i) in zip(leaves, range(len(leaves))) if query == y]
    for idx in start_idx:
        if ' '.join(in_leaves) == ' '.join(leaves[idx:idx + len(in_leaves)]):
            return [idx, idx + len(in_leaves) - 1]

    raise ValueError(f'Not found: {in_leaves}\nIN: {leaves}')


def tree_to_leave_pos_node_span(tree):
    leaves = tree.leaves()
    pos_tags = []
    tree_node_lst = []
    spans = []
    queue_tree = queue.Queue()
    queue_tree.put(tree)
    while not queue_tree.empty():
        node = queue_tree.get()
        if len(node) <= 0:
            logger.warn("[bft]: len(node) <= 0!! will cause error later")

        if len(node) == 1 and isinstance(node[0], str):
            pos_tags.append(node.label())
            continue

        tree_node_lst.append(node)

        # create the spans
        internal_leaves = node.leaves()
        spans.append(leaves2span(internal_leaves, leaves))
        for i in range(len(node)):
            child = node[i]
            if isinstance(child, Tree):
                queue_tree.put(child)
    nodes = [x.label() for x in tree_node_lst]
    return leaves, pos_tags, nodes, spans, tree_node_lst


def tree_string_to_leave_pos_node_span(tree_string, no_collapse = False):
    tree = tree_from_string(tree_string)
    if not no_collapse:
        leaves, pos_tags, nodes, spans, _ = tree_to_leave_pos_node_span_collapse_v3(tree)
    else:
        leaves, pos_tags, nodes, spans, _ = tree_to_leave_pos_node_span(tree)

    return leaves, pos_tags, nodes, spans


def tree_string_to_symbols(tree_string):
    tree = tree_from_string(tree_string)
    leaves = tree.leaves()
    labels = []
    queue_tree = queue.Queue()
    queue_tree.put(tree)
    while not queue_tree.empty():
        node = queue_tree.get()
        labels.append(node.label())

        if len(node) == 1 and isinstance(node[0], str):
            continue

        for i in range(len(node)):
            child = node[i]
            if isinstance(child, Tree):
                queue_tree.put(child)

    tokens = leaves + labels
    return tokens
