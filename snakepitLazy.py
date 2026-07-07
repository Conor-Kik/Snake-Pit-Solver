from gurobipy import *
from problems import get_problem
from board_plotting import plot_board_lazy

PROBLEM = 6

TRIALS = 1

data = get_problem(PROBLEM)

grid = data.grid
circle_squares = data.circle_squares
x_squares = data.x_squares

T = data.T

def get_orth_neighbours(pos):
    i, j = pos
    rows, cols = len(grid), len(grid[0])
    return [
        (ni, nj)
        for ni, nj in (
            (i - 1, j),
            (i + 1, j),
            (i, j - 1),
            (i, j + 1),
        )
        if 0 <= ni < rows and 0 <= nj < cols
    ]


n = len(grid)
N = range(n)
S = [(i, j) for i in N for j in N]

avg_lazy = 0 
avg_time = 0
for seed in range(TRIALS):
    lazy_count = 0 
    m = Model()
    X = {(s, t): m.addVar(vtype=GRB.BINARY) for s in S for t in T}

    # Set seeds and 1 number per square
    for s in S:
        i, j = s
        if grid[i][j] is not None:
            m.addConstr(X[s, grid[i][j]] == 1)
        m.addConstr(quicksum(X[s, t] for t in T) == 1)

    for s in S:
        # Upper bound for orthogonal neighbours
        deg = len(get_orth_neighbours(s))

        for t in T:
            if s in circle_squares or t == 2:
                k = 1
            else:
                k = 2

            if s in x_squares:
                b = 2
            else:
                b = 1

            # Force x squares to have two neighbours
            m.addConstr(
                b * X[s, t] <= quicksum(X[ss, t] for ss in get_orth_neighbours(s))
            )

            # Force circles to have one neighbour
            m.addConstr(
                quicksum(X[ss, t] for ss in get_orth_neighbours(s))
                <= k + (deg - k) * (1 - X[s, t])
            )


    def get_paths_by_number(XV):
        """
        XV: dict mapping (i,j) -> value (1..9)
        grid: square board (only used for size)

        Returns:
            dict[int, list[list[(int,int)]]]
            number -> list of orthogonally connected paths
        """
        n = len(grid)
        visited = set()
        paths_by_num = {}

        for i in range(n):
            for j in range(n):
                s = (i, j)
                if s in visited:
                    continue

                v = XV[s]
                stack = [s]
                visited.add(s)
                comp = []

                while stack:
                    cur = stack.pop()
                    comp.append(cur)
                    for nb in get_orth_neighbours(cur):
                        if nb not in visited and XV[nb] == v:
                            visited.add(nb)
                            stack.append(nb)

                paths_by_num.setdefault(v, []).append(comp)

        return paths_by_num

    def Callback(model, where):
        if where == GRB.Callback.MIPSOL:
            global lazy_count
            XV = model.cbGetSolution(X)
            s_map = {}
            for i in N:
                for j in N:
                    for t in T:
                        if XV[(i, j), t] > 0.5:
                            s_map[(i, j)] = t
                            break
            paths = get_paths_by_number(s_map)

            for t in T:
                if t not in paths:
                    continue
                for path in paths[t]:
                    # Cut if path doesn't have two ends
                    ends = sum(
                        1
                        for s in path
                        if sum((nb in path) for nb in get_orth_neighbours(s)) == 1
                    )
                    if ends <= 1:
                        lazy_count += 1
                        model.cbLazy(quicksum(X[s, t] for s in path) <= len(path) - 1)
                    if len(path) == t:
                        continue
                    # Cut if path is too long
                    if len(path) > t:
                        lazy_count += 1
                        model.cbLazy(quicksum(X[ss, t] for ss in path) <= len(path) - 1)
                    # Force path to grow by one if too short
                    if len(path) < t:
                        boundary = {
                            sss
                            for ss in path
                            for sss in get_orth_neighbours(ss)
                            if sss not in path
                        }

                        lazy_count += 1
                        model.cbLazy(
                            quicksum(X[sss, t] for sss in boundary)
                            >= 1 - len(path) + quicksum(X[ss, t] for ss in path)
                        )
    #m.Params.Threads = 8
    m.Params.LazyConstraints = 1
    m.Params.Seed = seed

    m.optimize(Callback)
    avg_time += m.Runtime
    avg_lazy += lazy_count
    print("Seed", seed)
    print("Lazy Constraints", lazy_count)
    print()


print(f"Average runtime {avg_time/TRIALS:.2f} with {TRIALS} Trials")
print("Average Lazy Constraints Added", round(avg_lazy/TRIALS, 2))
print("Constraints", m.NumConstrs)
print("Variables", m.NumVars)


plot_board_lazy(
    m,
    X,
    grid,
    T,
    circle_squares,
    x_squares,
)