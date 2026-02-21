from gurobipy import *
from problems import get_problem


PROBLEM = 2

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
                        lazy_count += 1
                        model.cbLazy(
                            quicksum(
                                X[sss, t]
                                for ss in path
                                for sss in get_orth_neighbours(ss)
                                if sss not in path
                            )
                            >= 1 - len(path) + quicksum(X[ss, t] for ss in path)
                        )
    m.Params.Threads = 8
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


def plot_board_matplotlib():
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import matplotlib.colors as mcolors

    if T is None:
        raise ValueError("Pass T (e.g., T=range(1,14))")

    rows, cols = len(grid), len(grid[0])
    circle_set = set(circle_squares)
    blocked_set = set(x_squares or [])

    if m.SolCount == 0:
        print("No solution in model.")
        return

    # Extract chosen digit per cell
    sol = {}
    for i in range(rows):
        for j in range(cols):
            sol[(i, j)] = max(T, key=lambda t: X[((i, j), t)].X)

    # --- Digit colors ---
    cmap = plt.get_cmap("tab20")
    denom = max(1, (len(T) - 1))
    digit_color = {t: mcolors.to_hex(cmap(k / denom)) for k, t in enumerate(sorted(T))}

    fig, ax = plt.subplots()
    ax.set_aspect("equal")
    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.invert_yaxis()

    # Grid
    ax.set_xticks(range(cols + 1))
    ax.set_yticks(range(rows + 1))
    ax.grid(True)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(length=0)

    def draw_faint_x(ax, x, y, size=0.18, lw=1.1, alpha=0.35):
        """Draw a faint centered X inside the unit square at (x, y)."""
        cx, cy = x + 0.5, y + 0.5
        s = size
        ax.plot(
            [cx - s, cx + s],
            [cy - s, cy + s],
            color="black",
            linewidth=lw,
            alpha=alpha,
            solid_capstyle="round",
        )
        ax.plot(
            [cx - s, cx + s],
            [cy + s, cy - s],
            color="black",
            linewidth=lw,
            alpha=alpha,
            solid_capstyle="round",
        )

    for i in range(rows):
        for j in range(cols):
            x, y = j, i
            d = sol[(i, j)]

            # Base colored square (by digit)
            ax.add_patch(
                patches.Rectangle(
                    (x, y),
                    1,
                    1,
                    facecolor=digit_color[d],
                    edgecolor="none",
                    alpha=0.30,
                )
            )

            # X-square marker (faint X only)
            if (i, j) in blocked_set:
                draw_faint_x(ax, x, y)

            # Circle marker
            if (i, j) in circle_set:
                ax.add_patch(
                    patches.Circle(
                        (x + 0.5, y + 0.5),
                        0.40,
                        fill=False,
                        linewidth=2,
                        edgecolor="black",
                    )
                )

            # Given marker (seeded squares)
            if grid[i][j] is not None:
                ax.text(
                    x + 0.15,
                    y + 0.25,
                    "•",
                    ha="center",
                    va="center",
                    fontsize=14,
                    color="black",
                )

            # Digit
            ax.text(
                x + 0.5,
                y + 0.55,
                str(d),
                ha="center",
                va="center",
                fontsize=14,
                fontweight="bold",
                color="black",
            )
    plt.show()


plot_board_matplotlib()