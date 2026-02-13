from gurobipy import *
from problems import get_problem


FIND_MAX_LENGTH = False #Set true for looped exploration of max feasible length

STARTING_MAX = 10 # Starting max feasible length

PROBLEM = 2
data = get_problem(PROBLEM)

grid = data.grid
circle_squares = data.circle_squares
x_squares = data.x_squares

if FIND_MAX_LENGTH:
    values = [cell for row in grid for cell in row if cell is not None]
    max_length = max(values) if values else 9
    max_length = max(max_length, STARTING_MAX)
else:
    max_length = max(data.T)

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


A = [ (pos, nbr)
        for pos in S
        for nbr in get_orth_neighbours(pos) ]

while True:
    T = range(2, max_length + 1)
    print(f"Trying with max length of {max_length}")
    bigM = max(T)-1
    m = Model()

    # Flow on directed orthogonal arcs, indexed by time/value t
    F = {(a, t): m.addVar(vtype=GRB.BINARY) for a in A for t in T}

    # MTZ order variable (used only for t >= 3)
    M = {s: m.addVar(vtype=GRB.INTEGER, lb=0, ub=bigM) for s in S}

    # Cell membership: X[s,t] = 1 if cell s is of type t
    X = {(s, t): m.addVar(vtype=GRB.BINARY) for s in S for t in T}

    # Head and tail indicators
    H = {(s, t): m.addVar(vtype=GRB.BINARY) for s in S for t in T}
    E = {(s, t): m.addVar(vtype=GRB.BINARY) for s in S for t in T}

    # ------------------------------------------------------------------
    # Head/tail exclusivity + MTZ bounds tied to head/tail choice
    # ------------------------------------------------------------------
    for s in S:
        m.addConstr(quicksum(H[s,t] for t in T) + quicksum(E[s,t] for t in T) <= 1)

        for t in T:
            if t >= 2:
                # tail of length t -> label at least (t-1)
                m.addConstr(M[s] >= (t-1) * E[s,t])

                # tail of length t -> label at most (t-1); otherwise free up to bigM
                m.addConstr(M[s] <= (t-1) + (bigM-1) * (1 - E[s,t]))

                # head -> M[s] = 0
                m.addConstr(M[s] <= bigM * (1 - H[s,t]))

    # ------------------------------------------------------------------
    # Force 2-cycles for t = 2 (edge pairing)
    # ------------------------------------------------------------------
    for s in S:
        for neigh in get_orth_neighbours(s):
            # A t=2 arc must be paired in the opposite direction
            m.addConstr(
                F[(s, neigh), 2] == F[(neigh, s), 2]
            )

    # ------------------------------------------------------------------
    # X-squares cannot be heads or tails and 2's can't be on X squares
    # ------------------------------------------------------------------
    for s in x_squares:
        m.addConstr(
            quicksum(H[s, t] for t in T) +
            quicksum(E[s, t] for t in T)
            == 0
        )
        m.addConstr(X[s, 2] == 0)


    # ------------------------------------------------------------------
    # Circle squares: endpoint for t >= 3 unless used by t = 2
    # ------------------------------------------------------------------
    for c in circle_squares:
        m.addConstr(
            quicksum(H[c, t] + E[c, t] for t in T if t >= 3)
            ==
            1 - X[c,2] #- quicksum(F[(c, neigh), 2] for neigh in get_orth_neighbours(c))
        )
    # ------------------------------------------------------------------
    # 6) Degree caps across ALL t + coverage constraint
    # ------------------------------------------------------------------
    for s in S:
        # At most one outgoing arc across all t
        m.addConstr(
            quicksum(F[(s, neigh), t]
                    for neigh in get_orth_neighbours(s)
                    for t in T)
            <= 1
        )

        # At most one incoming arc across all t
        m.addConstr(
            quicksum(F[(neigh, s), t]
                    for neigh in get_orth_neighbours(s)
                    for t in T)
            <= 1
        )



    # ------------------------------------------------------------------
    # Define X[s,t] from head/tail/incidence of flow
    # ------------------------------------------------------------------
    for s in S:
        m.addConstr(quicksum(X[s, t] for t in T) == 1)

        for t in T:
            m.addConstr(
                2*X[s, t] >=
                quicksum(F[(s, neigh), t] + F[(neigh, s), t] for neigh in get_orth_neighbours(s))
            )


            m.addConstr(
                X[s, t] <=
                        quicksum(F[(neigh, s), t]
                    for neigh in get_orth_neighbours(s))
            +
            quicksum(F[(s, neigh), t]
                    for neigh in get_orth_neighbours(s))
            )




    # ------------------------------------------------------------------
    # Adjacent cells cannot share the same t unless connected by a t-arc
    # ------------------------------------------------------------------
    for s in S:
        for nbh in get_orth_neighbours(s):
            if s < nbh:
                for t in T:
                    m.addConstr(
                        X[s, t] + X[nbh, t]
                        <=
                        1 + F[(s, nbh), t] + F[(nbh, s), t]
                    )


    # ------------------------------------------------------------------
    # Seed squares: all incident arcs must match the given value
    # ------------------------------------------------------------------
    for s in S:
        i, j = s
        if grid[i][j] is not None:
            value = grid[i][j]
            m.addConstr(X[s, value] == 1)



    # ------------------------------------------------------------------
    # Flow balance: in + head = out + tail (per node, per t)
    # ------------------------------------------------------------------
    for s in S:
        for t in T:
            m.addConstr(
                quicksum(F[(neigh, s), t] for neigh in get_orth_neighbours(s))
                + H[s, t]
                ==
                quicksum(F[(s, neigh), t] for neigh in get_orth_neighbours(s))
                + E[s, t]
            )


    # ------------------------------------------------------------------
    # MTZ exact-step constraints for arcs chosen with t >= 3
    # ------------------------------------------------------------------
    for s in S:
        for neigh in get_orth_neighbours(s):
            y = quicksum(F[(s, neigh), t] for t in T if t >= 3)  

            m.addConstr(M[neigh] >= M[s] + 1 - (bigM + 1) * (1 - y))

            m.addConstr(M[neigh] <= M[s] + 1 + (bigM -1)* (1 - y))


    m.optimize()
    if m.Status == GRB.OPTIMAL:
        break
    else:
        max_length += 1


print("Constraints", m.NumConstrs)
print("Variables", m.NumVars)



def plot_flow_matplotlib_colored_by_cell_label(
    m, F, grid, T,
    title="Flow Formulation Solution",
    tol=0.5,
    arrow_shrinkA=14,
    arrow_shrinkB=14,
):
    """
    - Cells are colored by inferred label t
    - Digits are drawn ONCE per cell in board-style position (center)
    - Flow arrows are drawn but do NOT carry numbers
    - Blocked/X squares use the SAME faint X marker as plot_board_matplotlib
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import matplotlib.colors as mcolors

    if m.SolCount == 0:
        print("No solution in model.")
        return

    rows, cols = len(grid), len(grid[0])
    circle_set = set(circle_squares or [])
    blocked_set = set(x_squares or [])
    T_list = list(T)

    def get_orth_neighbours(pos):
        i, j = pos
        return [
            (ni, nj)
            for ni, nj in ((i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1))
            if 0 <= ni < rows and 0 <= nj < cols
        ]

    # --- SAME faint X as board plot ---
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

    # --- color map for labels t ---
    cmap = plt.get_cmap("tab20")
    denom = max(1, (len(T_list) - 1))
    t_color = {t: mcolors.to_hex(cmap(k / denom)) for k, t in enumerate(sorted(T_list))}

    # --- infer a label t for every cell from incident flow ---
    cell_label = {}
    for i in range(rows):
        for j in range(cols):
            s = (i, j)
            scores = {}
            for t in T_list:
                outd = sum(F[((s, neigh), t)].X for neigh in get_orth_neighbours(s))
                ind  = sum(F[((neigh, s), t)].X for neigh in get_orth_neighbours(s))
                scores[t] = outd + ind
            best_t = max(T_list, key=lambda tt: scores[tt])
            cell_label[s] = best_t if scores[best_t] > tol else None

    # --- collect active arcs per t for drawing ---
    active = {t: [] for t in T_list}
    for (a, t), var in F.items():
        if var.X > tol:
            u, v = a
            active[t].append((u, v))

    # --- figure ---
    fig, ax = plt.subplots()
    ax.set_aspect("equal")
    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.invert_yaxis()

    ax.set_xticks(range(cols + 1))
    ax.set_yticks(range(rows + 1))
    ax.grid(True)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(length=0)

    # --- draw colored cells + markers + numbers ---
    for i in range(rows):
        for j in range(cols):
            x, y = j, i
            s = (i, j)
            t_s = cell_label[s]

            if t_s is not None:
                ax.add_patch(
                    patches.Rectangle(
                        (x, y), 1, 1,
                        facecolor=t_color[t_s],
                        edgecolor="none",
                        alpha=0.25
                    )
                )

            # Blocked/X-square marker (FAINT X ONLY)
            if (i, j) in blocked_set:
                draw_faint_x(ax, x, y)

            if (i, j) in circle_set:
                ax.add_patch(
                    patches.Circle((x + 0.5, y + 0.5), 0.40, fill=False, linewidth=2, edgecolor="black")
                )

            # Given marker
            if grid[i][j] is not None:
                ax.text(x + 0.15, y + 0.25, "•", ha="center", va="center", fontsize=14)

            # Single number per cell
            if t_s is not None:
                ax.text(
                    x + 0.5, y + 0.55, str(t_s),
                    ha="center", va="center",
                    fontsize=14,
                    fontweight="bold",
                    color="black"
                )

    # --- draw arrows (pure flow, no numbers) ---
    denom_t = max(1, (len(T_list) - 1))
    for k, t in enumerate(sorted(T_list)):
        arcs_t = active.get(t, [])
        if not arcs_t:
            continue

        off = (k - denom_t / 2) * (0.06 / max(1, denom_t))

        for u, v in arcs_t:
            (ui, uj), (vi, vj) = u, v
            x0, y0 = uj + 0.5, ui + 0.5
            x1, y1 = vj + 0.5, vi + 0.5
            dx, dy = (x1 - x0), (y1 - y0)

            if dx == 0 and dy == 0:
                continue
            if dx == 0:
                px, py = 1.0, 0.0
            elif dy == 0:
                px, py = 0.0, 1.0
            else:
                norm = (dx * dx + dy * dy) ** 0.5
                px, py = (-dy / norm, dx / norm)

            x0o, y0o = x0 + off * px, y0 + off * py
            x1o, y1o = x1 + off * px, y1 + off * py

            ax.annotate(
                "",
                xy=(x1o, y1o),
                xytext=(x0o, y0o),
                arrowprops=dict(
                    arrowstyle="->",
                    lw=2.0,
                    shrinkA=arrow_shrinkA,
                    shrinkB=arrow_shrinkB,
                ),
            )

    ax.set_title(title)
    plt.show()

plot_flow_matplotlib_colored_by_cell_label(
    m, F, grid, T)