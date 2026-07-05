def plot_board(
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
            if (i,j) in sol:
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

    plt.show()

    return fig, ax

