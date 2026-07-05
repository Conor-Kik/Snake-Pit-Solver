from gurobipy import *
from problems import get_problem
from board_plotting import plot_board_flow

TRIALS = 10
PROBLEM = 5
data = get_problem(PROBLEM)

grid = data.grid
circle_squares = data.circle_squares
x_squares = data.x_squares


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


A = [(pos, nbr) for pos in S for nbr in get_orth_neighbours(pos)]

avg_time = 0
T = data.T
bigM = max(T) - 1

for seed in range(TRIALS):
    m = Model()

    # Flow on directed orthogonal arcs, indexed by time/value t
    F = {(a, t): m.addVar(vtype=GRB.BINARY) for a in A for t in T}

    # MTZ order variable 
    M = {s: m.addVar(vtype=GRB.INTEGER, lb=0, ub=bigM) for s in S}

    # Cell membership: X[s,t] = 1 if cell s is of type l
    X = {(s, t): m.addVar(vtype=GRB.BINARY) for s in S for t in T}

    # Head and tail indicators
    H = {(s, t): m.addVar(vtype=GRB.BINARY) for s in S for t in T}
    E = {(s, t): m.addVar(vtype=GRB.BINARY) for s in S for t in T}

    # ------------------------------------------------------------------
    # Head/tail exclusivity + MTZ bounds tied to head/tail choice
    # ------------------------------------------------------------------
    for s in S:
        m.addConstr(quicksum(H[s, t] + E[s, t] for t in T) <= 1)

        for t in T:
            # tail of length t -> label at least (t-1)
            m.addConstr(M[s] >= (t - 1) * E[s, t])

            # tail of length t -> label at most (t-1); otherwise free up to bigM
            m.addConstr(M[s] <= (t - 1) + (bigM - 1) * (1 - E[s, t]))

            # head -> M[s] = 0
            m.addConstr(M[s] <= bigM * (1 - H[s, t]))

    # ------------------------------------------------------------------
    # X-squares cannot be heads or tails can't be on X squares
    # ------------------------------------------------------------------
    for s in x_squares:
        m.addConstr(quicksum(H[s, t] + E[s, t] for t in T) == 0)

    # ------------------------------------------------------------------
    # Circle squares are endpoint
    # ------------------------------------------------------------------
    for c in circle_squares:
        m.addConstr(
            quicksum(H[c, t] + E[c, t] for t in T)
            == 1
        )



    # ------------------------------------------------------------------
    # Degree caps across ALL t 
    # ------------------------------------------------------------------
    for s in S:
        m.addConstr(quicksum(F[(s, neigh), t] for neigh in get_orth_neighbours(s) for t in T) <= 1)
        m.addConstr(quicksum(F[(neigh, s), t] for neigh in get_orth_neighbours(s) for t in T) <= 1)

    # ------------------------------------------------------------------
    # Define X[s,t] from head/tail/incidence of flow
    # ------------------------------------------------------------------
    for s in S:
        m.addConstr(quicksum(X[s, t] for t in T) == 1)
        if s in x_squares:
            b = 2
        else:
            b = 1
        for t in T:
            if s in circle_squares or t == 2:
                k = 1
            else:
                k = 2
            m.addConstr(
                k * X[s, t] 
                >= quicksum(
                    F[(s, neigh), t] + F[(neigh, s), t]
                    for neigh in get_orth_neighbours(s)
                )
            )

            m.addConstr(
                b*X[s, t]
                <= quicksum(F[(neigh, s), t] for neigh in get_orth_neighbours(s))
                + quicksum(F[(s, neigh), t] for neigh in get_orth_neighbours(s))
            )

    # ------------------------------------------------------------------
    # Adjacent cells cannot share the same t unless connected by a t-arc
    # ------------------------------------------------------------------
    for s in S:
        for nbh in get_orth_neighbours(s):
            if s < nbh:
                for t in T:
                    m.addConstr(
                        X[s, t] + X[nbh, t] <= 1 + F[(s, nbh), t] + F[(nbh, s), t]
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
                quicksum(F[(neigh, s), t] for neigh in get_orth_neighbours(s)) + H[s, t]
                == quicksum(F[(s, neigh), t] for neigh in get_orth_neighbours(s))
                + E[s, t]
            )

    # ------------------------------------------------------------------
    # MTZ exact-step constraints for arcs chosen
    # ------------------------------------------------------------------
    for s in S:
        for neigh in get_orth_neighbours(s):
            y = quicksum(F[(s, neigh), t] for t in T)

            m.addConstr(M[neigh] >= M[s] + 1 - (bigM + 1) * (1 - y))

            m.addConstr(M[neigh] <= M[s] + 1 + (bigM - 1) * (1 - y))

    #m.Params.Threads = 8
    m.Params.Seed = seed
    

    m.optimize()
    avg_time += m.Runtime

    runtime = round(m.Runtime, 2)
    
    print("----------------------------------")
    print(f"Seed {seed}: {runtime}s")
    print(f"Average so far: {avg_time/(seed + 1):.2f}")
    print("----------------------------------")


print(f"Average runtime {avg_time/TRIALS:.2f} with {TRIALS} Trials")
print("Constraints", m.NumConstrs)
print("Variables", m.NumVars)

plot_board_flow(m, F, grid, T, circle_squares, x_squares)
