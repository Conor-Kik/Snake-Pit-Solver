from gurobipy import *
from problems import get_problem
import time
from board_plotting import plot_board_path_gen

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


# Path Gen approach
start_time = time.time()

T = data.T

n = len(grid)
N = range(n)
S = [(i, j) for i in N for j in N]
Cols = set()
Seed = {2: {(s, ss) for s in S for ss in get_orth_neighbours(s) if ss > s}}

avg_time = 0


Neigh = {s: get_orth_neighbours(s) for s in S}

ogrid = grid
grid = {s: grid[s[0]][s[1]] for s in S}

# All snakes to use
Paths = set()

def Compatible(n1,n2):
    return not n1 or not n2 or n1==n2

start_time = time.time()

# Growing snakes of length t
TPath = {t: set() for t in T}
# Length 2, if two numbers, both the same
TPath[2] = {(s1,s2) for s1 in S for s2 in Neigh[s1] if s1<s2
            and Compatible(grid[s1], grid[s2])}

for t in T:
    for p in TPath[t]:
        # Is it OK to add to Paths
        # Doesn't start or end at an x_square
        # Doesn't contain a number which is not t
        # Doesn't have a neighbour which is t
        if p[0] not in x_squares and p[-1] not in x_squares and \
            all(Compatible(grid[s],t) for s in p) and \
                all(grid[s]!=t for s in {sn for ss in p for sn in Neigh[ss]}-set(p)):
                Paths.add(p)
        # Is it OK to extend to the next size
        # Don't grow beyond max size, and don't grow if it has a t in it
        if t==T[-1] or any(grid[s]==t for s in p):
            continue
        # Grow at the front
        if p[0] not in circle_squares:
            for s in Neigh[p[0]]:
                if grid[s] and grid[s] <=t:
                    continue
                # Not in the path and not a new number
                # Not neighbouring anything else in the path
                if s not in p and (not grid[s] or all(Compatible(grid[s],grid[s2]) for s2 in p)) \
                    and all(s not in Neigh[s2] for s2 in p[1:]):
                    if s < p[-1]:
                        TPath[t+1].add((s,)+p)
                    else:
                        TPath[t+1].add(p[::-1]+(s,))
        # Grow at the end
        if p[-1] not in circle_squares:
            for s in Neigh[p[-1]]:
                if grid[s] and grid[s] <=t:
                    continue
                # Not in the path and not a new number
                # Not neighbouring anything else in the path
                if s not in p and (not grid[s] or all(Compatible(grid[s],grid[s2]) for s2 in p)) \
                    and all(s not in Neigh[s2] for s2 in p[:-1]):
                    if s > p[0]:
                        TPath[t+1].add(p+(s,))
                    else:
                        TPath[t+1].add((s,)+p[::-1])
    print(t, len(TPath[t]), len(Paths))


PSet = {s: set() for s in S}

for p in Paths:
    for s in p:
        PSet[s].add(p)
end_time = time.time()
col_gen_time = end_time-start_time
avg_lazy = 0

for seed in range(TRIALS):
    m = Model()
    lazy_count = 0
    Z = {p: m.addVar(vtype=GRB.BINARY) for p in Paths}

    Cover = {s: 
        m.addConstr(quicksum(Z[p] for p in PSet[s])==1)
        for s in S}
        
    def Callback(model,where):
        if where==GRB.Callback.MIPSOL:
            global lazy_count
            sol = {}
            Snakes = set()
            ZV = model.cbGetSolution(Z)
            for p in ZV:
                if round(ZV[p])==1:
                    Snakes.add(p)
                    for s in p:
                        sol[s] = len(p)
            
            for p in Snakes:
                for s1 in p:
                    for s2 in Neigh[s1]:
                        if s2>s1 and sol[s1]==sol[s2] and s2 not in p:
                            lazy_count += 1
                            model.cbLazy(
                                quicksum(
                                    Z[pp] for pp in PSet[s1]^PSet[s2] if len(pp)==len(p))<=1)

    m.Params.LazyConstraints = 1
    m.Params.LazyConstraints = 1
    m.Params.Seed = seed
    #m.Params.Threads = 8
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
    plot_board_path_gen(sol, ogrid, circle_squares, blocked_squares=x_squares, T=T, title="Solution")



