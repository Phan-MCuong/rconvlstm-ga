# ga_run.py
import random, math, json
from copy import deepcopy
from src.ga_eval import evaluate_individual

# 1) không gian tham số
FILTER_CHOICES = [(16,32), (32,64), (32,96), (64,64)]
KERNEL_CHOICES = [(3,3), (5,5)]
BATCH_CHOICES  = [4, 8, 16]
T_IN_CHOICES   = [4,5,6,7]

def random_genome():
    return {
        "T_in": random.choice(T_IN_CHOICES),
        "filters": random.choice(FILTER_CHOICES),
        "kernels": random.choice(KERNEL_CHOICES),
        "dropout": round(random.uniform(0.0, 0.3), 3),
        "use_bn": random.choice([True, False]),
        "relu_cap": 1.0,
        "lr": 10 ** random.uniform(-4.0, -3.0),   # 1e-4 .. 1e-3
        "batch": random.choice(BATCH_CHOICES),
        "epochs_eval": random.choice([4,5,6])     # proxy train
    }

def mutate(g, rate=0.2):
    h = deepcopy(g)
    if random.random() < rate: h["T_in"]   = random.choice(T_IN_CHOICES)
    if random.random() < rate: h["filters"]= random.choice(FILTER_CHOICES)
    if random.random() < rate: h["kernels"]= random.choice(KERNEL_CHOICES)
    if random.random() < rate: h["dropout"]= round(max(0.0, min(0.5, h["dropout"] + random.uniform(-0.1, 0.1))), 3)
    if random.random() < rate: h["use_bn"] = not h["use_bn"]
    if random.random() < rate: h["lr"]     = 10 ** random.uniform(-4.5, -3.0)
    if random.random() < rate: h["batch"]  = random.choice(BATCH_CHOICES)
    if random.random() < rate: h["epochs_eval"] = random.choice([4,5,6])
    return h

def crossover(a, b):
    c = {}
    for k in a.keys():
        c[k] = random.choice([a[k], b[k]])
    return c

def tournament_select(pop, scores, k, t=3):
    selected = []
    idxs = list(range(len(pop)))
    for _ in range(k):
        cand = random.sample(idxs, min(t, len(idxs)))
        best = min(cand, key=lambda i: scores[i])
        selected.append(deepcopy(pop[best]))
    return selected

def run_ga(generations=5, pop_size=12, elitism=2):
    # init
    population = [random_genome() for _ in range(pop_size)]
    history = []

    for gen in range(generations):
        results = []
        for i, g in enumerate(population):
            fit, aux = evaluate_individual(g)  # <<— chấm điểm cá thể
            results.append((fit, aux))
            print(f"[Gen {gen:02d} | Ind {i:02d}] fit={fit:.6f} aux={aux} g={g}")

        scores = [r[0] for r in results]
        # log best
        best_i = min(range(len(population)), key=lambda i: scores[i])
        history.append({"gen": gen, "best_fit": scores[best_i], "best_genome": population[best_i]})
        print(f"===> Gen {gen} BEST fit={scores[best_i]:.6f} genome={population[best_i]}")

        # next population
        elite_idx = sorted(range(len(population)), key=lambda i: scores[i])[:elitism]
        elites = [deepcopy(population[i]) for i in elite_idx]

        parents = tournament_select(population, scores, k=(pop_size - elitism))
        children = []
        for i in range(0, len(parents), 2):
            if i+1 < len(parents):
                ch = crossover(parents[i], parents[i+1])
            else:
                ch = deepcopy(parents[i])
            ch = mutate(ch, rate=0.2)
            children.append(ch)

        population = elites + children
        population = population[:pop_size]

    # final best
    final_scores = []
    for g in population:
        fit, aux = evaluate_individual(g)
        final_scores.append((fit, g, aux))
    final_best = min(final_scores, key=lambda x: x[0])
    print(f"\nFINAL BEST fit={final_best[0]:.6f}\n  genome={final_best[1]}\n  aux={final_best[2]}")
    return final_best, history

if __name__ == "__main__":
    best, hist = run_ga(generations=3, pop_size=8, elitism=2)
