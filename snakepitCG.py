from gurobipy import *
from problems import get_problem
import time

TRIALS = 1
PROBLEM = 1
data = get_problem(PROBLEM)

grid = data.grid
circle_squares = data.circle_squares
x_squares = data.x_squares




def get_orth_neighbours(pos):
    i, j = pos
    rows, cols = len(grid), len(grid[0])
    return {
        (ni, nj)
        for ni, nj in (
            (i - 1, j),
            (i + 1, j),
            (i, j - 1),
            (i, j + 1),
        )
        if 0 <= ni < rows and 0 <= nj < cols
    }


n = len(grid)
N = range(n)
S = [(i, j) for i in N for j in N]


def plot_board_matplotlib(
    sol, grid, circle_squares, blocked_squares=None, T=None, title=None
):
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import matplotlib.colors as mcolors

    if T is None:
        raise ValueError("Pass T (e.g., T=range(1,14))")

    rows, cols = len(grid), len(grid[0])
    circle_set = set(circle_squares)
    blocked_set = set(blocked_squares or [])

    # --- Digit colors (same mapping as other plots) ---
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

    return fig, ax


# CG approach
start_time = time.time()

T = data.T
Cols = set()
Seed = {2: {(s, ss) for s in S for ss in get_orth_neighbours(s) if ss > s}}

avg_time = 0
for t in T:
    for c in Seed[t]:
        if (
            c[0] not in x_squares
            and c[-1] not in x_squares
            and all(grid[s[0]][s[1]] in (None, t) for s in c)
            and not any(
                grid[ss[0]][ss[1]] == t
                for s in c
                for ss in get_orth_neighbours(s)
                if ss not in c
            )
        ):
            Cols.add(c)

    print(t, len(Seed[t]), len(Cols))

    if len(Seed[t]) == 0 or t + 1 not in T:
        break

    Seed[t + 1] = set()
    for c in Seed[t]:
        if any(grid[s[0]][s[1]] == t for s in c):
            continue
        if c[0] not in circle_squares:
            for s in get_orth_neighbours(c[0]):
                if s in c:
                    continue
                if any(s in get_orth_neighbours(ss) for ss in c[1:]):
                    continue
                if grid[s[0]][s[1]] and grid[s[0]][s[1]] <= t:
                    continue
                if s < c[-1]:
                    Seed[t + 1].add((s,) + c)
                else:
                    Seed[t + 1].add(tuple(reversed(c)) + (s,))
        if c[-1] not in circle_squares:
            for s in get_orth_neighbours(c[-1]):
                if s in c:
                    continue
                if any(s in get_orth_neighbours(ss) for ss in c[:-1]):
                    continue
                if grid[s[0]][s[1]] and grid[s[0]][s[1]] <= t:
                    continue
                if s > c[0]:
                    Seed[t + 1].add(c + (s,))
                else:
                    Seed[t + 1].add((s,) + tuple(reversed(c)))

end_time = time.time()
col_gen_time = end_time-start_time
avg_lazy = 0

for seed in range(TRIALS):
    m = Model()
    lazy_count = 0 

    Z = {p: m.addVar(vtype=GRB.BINARY) for p in Cols}



    Cover = {s: m.addConstr(quicksum(Z[p] for p in Z if s in p) == 1) for s in S}

    def Callback(model, where):
        if where == GRB.Callback.MIPSOL:
            global lazy_count
            ZV = model.cbGetSolution(Z)
            sol = {}
            pSet = set()
            for p, val in ZV.items():
                if val > 0.5:
                    pSet.add(p)
                    t = len(p)
                    for s in p:
                        sol[s] = t
            for p in pSet:
                t = len(p)
                for s in p:
                    for ss in get_orth_neighbours(s):
                        if ss > s and ss not in p and sol.get(ss) == t:
                            lazy_count += 1
                            model.cbLazy(
                                quicksum(
                                    Z[pp]
                                    for pp in Z
                                    if len(pp) == t and s in pp and ss not in pp
                                )
                                + quicksum(
                                    Z[pp]
                                    for pp in Z
                                    if len(pp) == t and ss in pp and s not in pp
                                )
                                <= 1
                            )
    m.Params.LazyConstraints = 1
    m.Params.Seed = seed
    m.Params.Threads = 8
    m.optimize(Callback)
    avg_lazy += lazy_count
    avg_time += m.Runtime
    print("Seed", seed)
    print("Lazy Constraints", lazy_count)
    print()

print(f"Average Solve runtime {(avg_time/TRIALS):.2f} with {TRIALS} Trials")
print(f"Average total runtime {col_gen_time + (avg_time/TRIALS):.2f} with {TRIALS} Trials")
print("Average Lazy Constraints", round(avg_lazy/TRIALS, 2))
print("Constraints", m.NumConstrs)
print("Variables", m.NumVars)

if m.SolCount > 0:
    sol = {}
    for p in Z:
        if round(Z[p].x) == 1:
            for s in p:
                sol[s] = len(p)
    plot_board_matplotlib(sol, grid, circle_squares, blocked_squares=x_squares, T=T, title="Solution")



