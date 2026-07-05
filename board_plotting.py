import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors

def plot_board_path_gen(
    sol, grid, circle_squares, blocked_squares=None, T=None, title=None
):


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


def plot_board_flow(
    m,
    F,
    grid,
    T,
    circle_squares,
    x_squares,
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
                ind = sum(F[((neigh, s), t)].X for neigh in get_orth_neighbours(s))
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
                        (x, y),
                        1,
                        1,
                        facecolor=t_color[t_s],
                        edgecolor="none",
                        alpha=0.25,
                    )
                )

            # Blocked/X-square marker (FAINT X ONLY)
            if (i, j) in blocked_set:
                draw_faint_x(ax, x, y)

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

            # Given marker
            if grid[i][j] is not None:
                ax.text(x + 0.15, y + 0.25, "•", ha="center", va="center", fontsize=14)

            # Single number per cell
            if t_s is not None:
                ax.text(
                    x + 0.5,
                    y + 0.55,
                    str(t_s),
                    ha="center",
                    va="center",
                    fontsize=14,
                    fontweight="bold",
                    color="black",
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


def plot_board_lazy(
    m,
    X,
    grid,
    T,
    circle_squares,
    x_squares,
):

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

