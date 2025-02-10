def ucb1(parent, child, exploration_weight):
    if child.visits == 0:
        return float('inf')

    exploitation = child.value
    exploration = 1 * ((2 * np.log(parent.visits) / child.visits) ** 0.5)
    return exploitation + exploration


def ucb1_tuned(parent, child, exploration_weight):
    if child.visits == 0:
        return float('inf')

    exploitation = child.value 

    V = var_child = child.std**2 + ((2 * np.log(parent.visits) / child.visits) ** 0.5)
    exploration = ((np.log(parent.visits) / child.visits) * min(0.25, V)) ** 0.5
    return exploitation + exploration


def sp_mcts(parent, child, exploration_weight):
    if child.visits == 0:
        return float('inf')

    D = 10000
    C = 0.5

    var_child = child.std**2
    
    exploitation = child.value
    exploration = C * ((2 * np.log(parent.visits) / child.visits) ** 0.5)
    sp_value = (var_child+(D/child.visits))**0.5
    return exploitation + exploration + sp_value


def uct(parent, child, exploration_weight):
    if child.visits == 0:
        return float('inf')

    exploitation = child.value
    exploration = exploration_weight * ((2 * np.log(parent.visits) / child.visits) ** 0.5)
    return exploitation + exploration


def dfs_uct(parent, child, exploration_weight, s, dfs_factor):
    if child.visits == 0:
        return float('inf')

    exploitation = child.value / child.visits
    exploration = exploration_weight * ((2 * np.log(parent.visits) / child.visits) ** 0.5)
    dfs_bonus = dfs_factor / (child.visits + s)
    return exploitation + exploration + dfs_bonus


