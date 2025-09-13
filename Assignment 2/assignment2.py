import random

# Hill Climbing Algorithm
def hill_climbing(func, bounds, step_size, start=None):
    # choose starting point (randomized if not provided)
    current_x = start if start is not None else random.uniform(bounds[0], bounds[1])
    # snap to nearest step size
    current_x = round(current_x / step_size) * step_size
    # starting point is within bounds given
    current_x = max(bounds[0], min(bounds[1], current_x))

   
    # print(f"  [Start point chosen: x={round(current_x,4)}]")

    current_val = func(current_x)

    while True:
        neighbors = []
        # generates neighboring points by step_size
        for step in [-step_size, step_size]:
            neighbor_x = current_x + step
            neighbor_x = round(neighbor_x / step_size) * step_size
            if bounds[0] <= neighbor_x <= bounds[1]:
                neighbors.append(neighbor_x)

        # edge case, stops if no neighbors exist
        if not neighbors:
            break

        # find the best neighbor 
        next_x = None
        next_val = float('-inf')
        for nx in neighbors:
            nv = func(nx)
            if nv > next_val:
                next_val = nv
                next_x = nx

        # stop if no improvement
        if next_val <= current_val:  
            break
        
        # better neighbor
        current_x, current_val = next_x, next_val

    return round(current_x, 4), round(current_val, 4)


# Random Restart Hill Climbing
def random_restart_hill_climbing(func, bounds, step_size, num_restarts):
    best_x = None
    best_val = float('-inf')

    for i in range(num_restarts):
        # random starting point
        start_x = random.uniform(bounds[0], bounds[1])
        start_x = round(start_x / step_size) * step_size
        start_x = max(bounds[0], min(bounds[1], start_x))
        # print(f"  Restart {i+1}: starting at x={round(start_x,4)}")

        # run hill-climbing from the starting point generated above
        x, val = hill_climbing(func, bounds, step_size, start_x)
        
        # keep track of best result across all restarts
        if val > best_val:
            best_val = val
            best_x = x

    return round(best_x, 4), round(best_val, 4)


def f(x):
    return 2 - x**2

def g(x):
    return (0.0051 * x**5) - (0.1367 * x**4) + (1.24 * x**3) - (4.456 * x**2) + (5.66 * x) - 0.287


# Part 1: Hill-climbing on f(x)
print("Part 1a: Hill-climbing for f(x) with step-size 0.5")
x1, y1 = hill_climbing(f, [-5, 5], 0.5)
print(f"  Maximum found at (x={x1}, y={y1})\n")

print("Part 1b: Hill-climbing for f(x) with step-size 0.01")
x2, y2 = hill_climbing(f, [-5, 5], 0.01)
print(f"  Maximum found at (x={x2}, y={y2})\n")


# Part 2: Hill-climbing on g(x)
print("Part 2a: Random-restart hill-climbing for g(x) with step-size 0.5 and 20 restarts")
x3, y3 = random_restart_hill_climbing(g, [0, 10], 0.5, 20)
print(f"  Best maximum found at (x={x3}, y={y3})\n")

print("Part 2b: Simple hill-climbing for g(x) with step-size 0.5")
x4, y4 = hill_climbing(g, [0, 10], 0.5)
print(f"  Maximum found at (x={x4}, y={y4})\n")
