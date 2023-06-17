import copy
import sys
import time

import numpy as np


def read_input(input_path=None):
    input_cost_lines = [
        "1 1 1 1 1 1 1 1 1 1",
        "1 1 1 1 1 1 1 1 1 1",
        "1 5 5 5 1 5 5 5 1 1",
        "1 5 5 5 1 5 5 5 1 1",
        "1 5 5 5 1 5 5 5 1 1",
        "1 0 0 0 0 0 0 0 0 1",
    ]

    input_cost = []
    if input_path is None:
        lines = input_cost_lines
    else:
        with open(input_path) as f:
            lines = f.readlines()
    for line in lines:
        input_cost.append([int(s) for s in line.split(" ")])
    return input_cost


def print_matrix(matrix):
    m = len(matrix)
    n = len(matrix[0])
    print(m, 'x', n)
    fill = '0'
    align = '>'
    width = 1
    for r in range(m):
        for c in range(n):
            print(f'{matrix[r][c]:{fill}{align}{width}}', end=' ')
        print()


def get_directions(change_order=True):
    if not change_order:
        # Up, Down, Right, Left
        dx = [-1, +0, +1, +0]
        dy = [+0, +1, +0, -1]
    else:
        # Right, Down, Left, Up
        dx = [+0, +1, +0, -1]
        dy = [+1, +0, -1, +0]
    return dx, dy


def manhattan_distance(curr, goal):
    return abs(curr[0] - goal[0]) + abs(curr[1] - goal[1])


def euclidean_distance(curr, goal):
    return ((curr[0] - goal[0]) ** 2 + (curr[1] - goal[1]) ** 2) ** 0.5


def evaluationFunction(blue_curr, blue_goal, red_curr):
    bx, by = blue_curr
    man_dist = manhattan_distance(blue_curr, blue_goal)
    if bx == 5:
        man_dist += 2
    rx, ry = red_curr
    vertical_dist = abs(ry - rx)
    return -1 * man_dist + vertical_dist


def minimax_search(matrix, blue_curr, blue_goal, red_curr, agent_idx, depth):
    if depth == 1 or (blue_curr == blue_goal):
        ret = evaluationFunction(blue_curr, blue_goal, red_curr), None
    elif agent_idx == 0:
        ret = maximizer(matrix, blue_curr, blue_goal, red_curr, agent_idx, depth)
    else:
        ret = minimizer(matrix, blue_curr, blue_goal, red_curr, agent_idx, depth)
    return ret


def minimizer(matrix, blue_curr, blue_goal, red_curr, agent_idx, depth):
    dx, dy = get_directions()
    m, n = matrix.shape
    if agent_idx == 1:
        next_agent = 0
    else:
        next_agent = agent_idx + 1
    next_depth = depth - 1
    x, y = red_curr
    min_score = sys.maxsize
    min_action = None
    for d in range(4):
        nx = x + dx[d]
        ny = y + dy[d]
        if nx < 0 or m <= nx or ny < 0 or n <= ny or matrix[nx][ny] == 0:
            continue
        red_next = (nx, ny)
        if blue_curr == red_next:
            continue
        new_score = minimax_search(matrix, blue_curr, blue_goal, red_next, next_agent, next_depth)[0]
        if new_score < min_score:
            min_score, min_action = new_score, (dx[d], dy[d])
    return min_score, min_action


def maximizer(matrix, blue_curr, blue_goal, red_curr, agent_idx, depth):
    dx, dy = get_directions()
    m, n = matrix.shape
    if agent_idx == 1:
        next_agent = 0
    else:
        next_agent = agent_idx + 1
    next_depth = depth - 1
    x, y = blue_curr
    max_score = -sys.maxsize
    max_action = None
    for d in range(4):
        nx = x + dx[d]
        ny = y + dy[d]
        if nx < 0 or m <= nx or ny < 0 or n <= ny or matrix[nx][ny] == 0:
            continue
        blue_next = (nx, ny)
        if red_curr == blue_next:
            continue
        new_score = minimax_search(matrix, blue_next, blue_goal, red_curr, next_agent, next_depth)[0]
        if new_score > max_score:
            max_score, max_action = new_score, (dx[d], dy[d])
    return max_score, max_action


def find_cliff_walking(matrix, blue_curr, blue_goal, red_curr, depth):
    path = []
    blue_path = []
    red_path = []
    agent_idx = 0
    blue_score = 0
    blue_path.append((blue_curr, 0))
    red_path.append((red_curr, 0))
    while len(blue_path) < max_length and blue_curr != blue_goal and blue_curr != red_curr:
        score, action = minimax_search(matrix, blue_curr, blue_goal, red_curr, agent_idx, depth)
        if agent_idx == 0:
            blue_curr = update_state(blue_curr, action)
            blue_score += matrix[blue_curr]
            path.append((blue_curr, agent_idx, score))
            blue_path.append((blue_curr, score))
            agent_idx += 1
        else:
            red_curr = update_state(red_curr, action)
            path.append((red_curr, agent_idx, score))
            red_path.append((red_curr, score))
            agent_idx = 0

    return blue_path, red_path, blue_score


def update_state(curr, action):
    if action is None:
        return curr
    return (curr[0] + action[0]), (curr[1] + action[1])


def minimax_alphabeta_search(matrix, blue_curr, blue_goal, red_curr, agent_idx, depth, alpha, beta):
    if depth == 1 or (blue_curr == blue_goal) or (blue_curr == red_curr):
        ret = evaluationFunction(blue_curr, blue_goal, red_curr), None
    elif agent_idx == 0:
        ret = alpha_maximizer(matrix, blue_curr, blue_goal, red_curr, agent_idx, depth, alpha, beta)
    else:
        ret = beta_minimizer(matrix, blue_curr, blue_goal, red_curr, agent_idx, depth, alpha, beta)
    return ret


def alpha_maximizer(matrix, blue_curr, blue_goal, red_curr, agent_idx, depth, alpha, beta):
    dx, dy = get_directions()
    m, n = matrix.shape
    if agent_idx == 1:
        next_agent = 0
    else:
        next_agent = agent_idx + 1
    next_depth = depth - 1
    x, y = blue_curr
    max_score = -sys.maxsize
    max_action = None
    for d in range(4):
        nx = x + dx[d]
        ny = y + dy[d]
        if nx < 0 or m <= nx or ny < 0 or n <= ny or matrix[nx][ny] == 0:
            continue
        blue_next = (nx, ny)
        if red_curr == blue_next:
            continue
        new_score = minimax_alphabeta_search(matrix, blue_next, blue_goal, red_curr,
                                             next_agent, next_depth, alpha, beta)[0]
        if new_score > max_score:
            max_score, max_action = new_score, (dx[d], dy[d])
        if new_score > beta:
            return new_score, (dx[d], dy[d])
        alpha = max(alpha, max_score)
    return max_score, max_action


def beta_minimizer(matrix, blue_curr, blue_goal, red_curr, agent_idx, depth, alpha, beta):
    dx, dy = get_directions()
    m, n = matrix.shape
    if agent_idx == 1:
        next_agent = 0
    else:
        next_agent = agent_idx + 1
    next_depth = depth - 1
    x, y = red_curr
    min_score = sys.maxsize
    min_action = None
    for d in range(4):
        nx = x + dx[d]
        ny = y + dy[d]
        if nx < 0 or m <= nx or ny < 0 or n <= ny or matrix[nx][ny] == 0:
            continue
        red_next = (nx, ny)
        if blue_curr == red_next:
            continue
        new_score = minimax_alphabeta_search(matrix, blue_curr, blue_goal, red_next,
                                             next_agent, next_depth, alpha, beta)[0]
        if new_score < min_score:
            min_score, min_action = new_score, (dx[d], dy[d])
        if new_score < alpha:
            return new_score, (dx[d], dy[d])
        beta = min(beta, min_score)
    return min_score, min_action


def find_cliff_walking_alpha_beta(matrix, blue_curr, blue_goal, red_curr, depth, alpha, beta):
    path = []
    blue_path = []
    red_path = []
    agent_idx = 0
    blue_score = 0
    blue_path.append((blue_curr, 0))
    red_path.append((red_curr, 0))
    while len(blue_path) < max_length and blue_curr != blue_goal:
        score, action = minimax_alphabeta_search(matrix, blue_curr, blue_goal, red_curr, agent_idx, depth, alpha, beta)
        if agent_idx == 0:
            blue_curr = update_state(blue_curr, action)
            blue_score += matrix[blue_curr]
            path.append((blue_curr, agent_idx, score))
            blue_path.append((blue_curr, score))
            agent_idx += 1
        else:
            red_curr = update_state(red_curr, action)
            path.append((red_curr, agent_idx, score))
            red_path.append((red_curr, score))
            agent_idx = 0

    return blue_path, red_path, blue_score


def plot_path(bp, rp, cost_matrix, name):
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    cmap = (mpl.colors.ListedColormap(['royalblue', 'white', 'green', 'orange'])
            .with_extremes(over='red', under='blue'))

    bounds = [0.0, 0.5, 2.0, 6.0, 20.0]
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    # bp, rp = path
    extreme_val = +100
    blue_val = -100
    red_val = +10

    blue_start = (5, 0)
    red_start = (0, 4)
    matrix = copy.deepcopy(cost_matrix)
    matrix[blue_start] = blue_val
    matrix[red_start] = red_val
    for i in range(1, 9):
        matrix[5, i] = extreme_val
    fig, ax = plt.subplots()
    matrice = ax.matshow(matrix, cmap=cmap, norm=norm, aspect='equal')
    cb = plt.colorbar(matrice)
    cb.remove()

    # plt.grid(color='black')
    plt.xticks(np.arange(0, 9 + 1, 1.0))
    plt.yticks(np.arange(0, 5 + 1, 1.0))
    plt.gca().set_xticks([x - 0.5 for x in plt.gca().get_xticks()][1:], minor='true')
    plt.gca().set_yticks([y - 0.5 for y in plt.gca().get_yticks()][1:], minor='true')
    plt.grid(which='minor', color='black')

    matrix[blue_start] = cost_matrix[blue_start]
    matrix[red_start] = cost_matrix[red_start]

    def animate(frame_idx):
        prev_idx = int(frame_idx / 2) - 1
        curr_idx = int(frame_idx / 2)
        if frame_idx == 0:
            matrix[bp[frame_idx][0]] = blue_val
        elif frame_idx == 1:
            matrix[rp[frame_idx - 1][0]] = red_val
        elif frame_idx % 2 == 0 and curr_idx < len(bp):
            matrix[bp[prev_idx][0]] = cost_matrix[bp[prev_idx][0]]
            matrix[bp[curr_idx][0]] = blue_val
        elif frame_idx % 2 == 1 and curr_idx < len(rp):
            matrix[rp[prev_idx][0]] = cost_matrix[rp[prev_idx][0]]
            matrix[rp[curr_idx][0]] = red_val

        matrice.set_array(matrix)
        return matrice

    anim = FuncAnimation(fig, animate, frames=30, interval=300)
    # plt.show()
    anim.save(f'{name}.gif', writer='pillow')
    # matrix = np.zeros((6, 10))
    #
    # for bp, rp in path:
    #     matrix[bp[0]] = 10
    #     matrix[rp[0]] = 20
    #     time.sleep(0.2)
    #     plt.imshow(matrix)
    #     matrix[bp[0]] = 0
    #     matrix[rp[0]] = 0
    #     plt.show()


def print_ret(ret, cost_matrix, name):
    bp, rp, bs = ret
    print(f"Blue Path : {bp}")
    print(f"Red  Path : {rp}")
    print(f"Blue Cost : {bs}")
    # plot_path(zip(bp, rp))
    plot_path(bp, rp, cost_matrix, name)


def run_minimax_search(cost_matrix, depth=2, alpha_beta_prune=False):
    st = time.time()
    with_text = 'without'
    if alpha_beta_prune:
        with_text = 'with'
    name = f"Minimax search method ({with_text} alpha-beta pruning) " \
           f"with the depth limited to {depth} layers (d={depth})"
    print(name)
    if alpha_beta_prune:
        ret = find_cliff_walking_alpha_beta(cost_matrix, blue_start_state, blue_goal_state, red_start_state,
                                            depth=depth, alpha=-sys.maxsize, beta=sys.maxsize)
    else:
        ret = find_cliff_walking(cost_matrix, blue_start_state, blue_goal_state, red_start_state, depth=depth)
    print_ret(ret, cost_matrix, name)
    elapsed_time = time.time() - st
    print('Execution time:', elapsed_time, 'seconds')


def main():
    # input_path = "input.txt"
    cost_matrix = np.array(read_input(input_path=None))
    print_matrix(cost_matrix)

    run_minimax_search(cost_matrix, depth=2)
    run_minimax_search(cost_matrix, depth=6)
    run_minimax_search(cost_matrix, depth=6, alpha_beta_prune=True)


if __name__ == '__main__':
    max_length = 100
    blue_start_state = (5, 0)
    blue_goal_state = (5, 9)
    red_start_state = (0, 4)
    main()
