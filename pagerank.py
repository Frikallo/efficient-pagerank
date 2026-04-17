"""
PageRank Graph Representation Experiments

Compares two graph representations for PageRank computation:
  1. Adjacency Matrix  (O(N^2) space)
  2. Adjacency List    (O(N + E) space)

Measurements:
  - Runtime vs graph size (with error bars over multiple trials)
  - Normalized runtime (validates theoretical scaling)
  - Runtime vs graph density
  - Peak resident memory via tracemalloc (byte-level measurement)
"""

import random
import time
import tracemalloc
import numpy as np
import matplotlib.pyplot as plt


# Graph Generation

def generate_sparse_graph(n, avg_degree=3, seed=None):
    """Generate a random directed graph with approximately `avg_degree` out-edges per node."""
    rng = random.Random(seed)
    graph = {i: set() for i in range(n)}
    for i in range(n):
        for _ in range(avg_degree):
            j = rng.randint(0, n - 1)
            if j != i:
                graph[i].add(j)
    return graph


def generate_dense_graph(n, density=0.3, seed=None):
    """Generate a random directed graph where each possible edge exists with probability `density`."""
    rng = random.Random(seed)
    graph = {i: set() for i in range(n)}
    for i in range(n):
        for j in range(n):
            if i != j and rng.random() < density:
                graph[i].add(j)
    return graph


# Conversions

def to_adjacency_matrix(graph, n):
    """Convert dict-of-sets to an N x N list-of-lists adjacency matrix."""
    matrix = [[0] * n for _ in range(n)]
    for i in graph:
        for j in graph[i]:
            matrix[i][j] = 1
    return matrix


# PageRank Implementations

def pagerank_list(graph, n, d=0.85, iterations=50):
    """PageRank over dict-of-sets adjacency list."""
    pr = [1 / n] * n
    for _ in range(iterations):
        new_pr = [(1 - d) / n] * n
        for i in range(n):
            out = graph[i]
            if not out:
                continue
            share = d * pr[i] / len(out)
            for j in out:
                new_pr[j] += share
        pr = new_pr
    return pr


def pagerank_matrix(matrix, n, d=0.85, iterations=50):
    """PageRank over dense N x N adjacency matrix."""
    pr = [1 / n] * n
    outdeg = [sum(row) for row in matrix]

    for _ in range(iterations):
        new_pr = [(1 - d) / n] * n
        for i in range(n):
            if outdeg[i] == 0:
                continue
            share = d * pr[i] / outdeg[i]
            row = matrix[i]
            for j in range(n):
                if row[j]:
                    new_pr[j] += share
        pr = new_pr
    return pr

### Utility Functions

# Correctness Check

def verify_implementations_agree(n=100, tol=1e-9):
    graph = generate_sparse_graph(n, seed=42)
    matrix = to_adjacency_matrix(graph, n)

    pr_list = pagerank_list(graph, n)
    pr_mat = pagerank_matrix(matrix, n)

    max_diff = max(abs(a - b) for a, b in zip(pr_list, pr_mat))
    assert max_diff < tol, f"List vs Matrix mismatch: {max_diff}"
    print(f"[verify] Both implementations agree (max diff < {tol}) (success)")


# Timing Utility

def timed_run(func, *args, trials=5):
    times = []
    for _ in range(trials):
        start = time.perf_counter()
        func(*args)
        times.append(time.perf_counter() - start)
    return float(np.mean(times)), float(np.std(times))


# Memory Measurement (tracemalloc)

def measure_build_memory(build_fn, *args):
    """
    Measure peak bytes allocated while constructing a data structure.
    Returns (peak_bytes, constructed_object).
    """
    tracemalloc.start()
    tracemalloc.clear_traces()
    obj = build_fn(*args)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak, obj


# Experiment 1: Runtime vs Graph Size

def experiment_runtime_vs_size():
    print("\n[exp1] Runtime vs graph size (sparse graphs)")
    sizes = [100, 300, 600, 1000, 2000]

    list_times, list_std = [], []
    matrix_times, matrix_std = [], []
    list_norm, matrix_norm = [], []

    for n in sizes:
        graph = generate_sparse_graph(n, seed=n)
        matrix = to_adjacency_matrix(graph, n)

        lt, ls = timed_run(pagerank_list, graph, n)
        mt, ms = timed_run(pagerank_matrix, matrix, n)

        list_times.append(lt);    list_std.append(ls)
        matrix_times.append(mt);  matrix_std.append(ms)

        list_norm.append(lt / n)
        matrix_norm.append(mt / (n * n))

        print(f"  n={n:5d}  list={lt:.4f}s  matrix={mt:.4f}s")

    # Raw runtime plot with error bars
    plt.figure()
    plt.errorbar(sizes, list_times,   yerr=list_std,   marker='o', label='Adjacency List',   capsize=3)
    plt.errorbar(sizes, matrix_times, yerr=matrix_std, marker='s', label='Adjacency Matrix', capsize=3)
    plt.xlabel('Number of Nodes (n)')
    plt.ylabel('Runtime (seconds)')
    plt.title('Runtime vs Graph Size (Sparse Graphs)')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('figures/runtime_vs_size.png', dpi=300)
    plt.close()

    # Normalized runtime plot
    plt.figure()
    plt.plot(sizes, list_norm,   marker='o', label='List (time / n)')
    plt.plot(sizes, matrix_norm, marker='s', label='Matrix (time / n²)')
    plt.xlabel('Number of Nodes (n)')
    plt.ylabel('Normalized Runtime (seconds)')
    plt.title('Normalized Runtime Comparison')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('figures/runtime_normalized.png', dpi=300)
    plt.close()


# Experiment 2: Runtime vs Graph Density

def experiment_runtime_vs_density():
    print("\n[exp2] Runtime vs graph density (n = 1000)")
    densities = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5]
    n = 1000

    list_times, matrix_times = [], []

    for d in densities:
        graph = generate_dense_graph(n, d, seed=int(d * 1000))
        matrix = to_adjacency_matrix(graph, n)

        lt, _ = timed_run(pagerank_list, graph, n, trials=3)
        mt, _ = timed_run(pagerank_matrix, matrix, n, trials=3)

        list_times.append(lt)
        matrix_times.append(mt)

        print(f"  density={d:.2f}  list={lt:.4f}s  matrix={mt:.4f}s")

    plt.figure()
    plt.plot(densities, list_times,   marker='o', label='Adjacency List')
    plt.plot(densities, matrix_times, marker='s', label='Adjacency Matrix')
    plt.xlabel('Graph Density')
    plt.ylabel('Runtime (seconds)')
    plt.title('Runtime vs Graph Density (n = 1000)')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('figures/runtime_vs_density.png', dpi=300)
    plt.close()


# Experiment 3: Memory Usage

def experiment_memory():
    print("\n[exp3] Peak memory usage (tracemalloc, bytes)")
    sizes = [100, 300, 600, 1000, 2000]

    list_mem, matrix_mem = [], []

    for n in sizes:
        list_peak,   graph = measure_build_memory(generate_sparse_graph, n, 3, n)
        matrix_peak, _     = measure_build_memory(to_adjacency_matrix, graph, n)

        list_mem.append(list_peak)
        matrix_mem.append(matrix_peak)

        print(f"  n={n:5d}  list={list_peak:>10,}B  matrix={matrix_peak:>12,}B")

    plt.figure()
    plt.plot(sizes, list_mem,   marker='o', label='Adjacency List')
    plt.plot(sizes, matrix_mem, marker='s', label='Adjacency Matrix')
    plt.yscale('log')
    plt.xlabel('Number of Nodes (n)')
    plt.ylabel('Peak Memory (bytes, log scale)')
    plt.title('Memory Usage Comparison')
    plt.legend()
    plt.grid(alpha=0.3, which='both')
    plt.tight_layout()
    plt.savefig('figures/memory_comparison_log.png', dpi=300)
    plt.close()

if __name__ == "__main__":
    verify_implementations_agree()
    experiment_runtime_vs_size()
    experiment_runtime_vs_density()
    experiment_memory()
    print("\nAll experiments complete.")