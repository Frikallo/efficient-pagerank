import random
import time
import matplotlib.pyplot as plt
import numpy as np

# Graph Generation

def generate_sparse_graph(n, avg_degree=3):
    graph = {i: set() for i in range(n)}
    for i in range(n):
        for _ in range(avg_degree):
            j = random.randint(0, n-1)
            if j != i:
                graph[i].add(j)
    return graph


def generate_dense_graph(n, density=0.3):
    graph = {i: set() for i in range(n)}
    for i in range(n):
        for j in range(n):
            if i != j and random.random() < density:
                graph[i].add(j)
    return graph

# Conversions

def to_adjacency_matrix(graph, n):
    matrix = [[0]*n for _ in range(n)]
    for i in graph:
        for j in graph[i]:
            matrix[i][j] = 1
    return matrix

# PageRank Implementations

def pagerank_list(graph, n, d=0.85, iterations=50):
    pr = [1/n]*n
    for _ in range(iterations):
        new_pr = [(1-d)/n]*n
        for i in range(n):
            if len(graph[i]) == 0:
                continue
            share = pr[i] / len(graph[i])
            for j in graph[i]:
                new_pr[j] += d * share
        pr = new_pr
    return pr


def pagerank_matrix(matrix, n, d=0.85, iterations=50):
    pr = [1/n]*n
    outdeg = [sum(row) for row in matrix]

    for _ in range(iterations):
        new_pr = [(1-d)/n]*n
        for i in range(n):
            if outdeg[i] == 0:
                continue
            for j in range(n):
                if matrix[i][j] == 1:
                    new_pr[j] += d * pr[i] / outdeg[i]
        pr = new_pr
    return pr

# Utilities

def timed_run(func, *args, trials=5):
    times = []
    for _ in range(trials):
        start = time.perf_counter()
        func(*args)
        times.append(time.perf_counter() - start)
    return np.mean(times), np.std(times)

# Experiment 1: Runtime vs Size + Normalization + Error Bars

def experiment_runtime_vs_size():
    sizes = [100, 300, 600, 1000, 2000]
    list_times, matrix_times = [], []
    list_std, matrix_std = [], []
    list_norm, matrix_norm = [], []

    for n in sizes:
        graph = generate_sparse_graph(n)
        matrix = to_adjacency_matrix(graph, n)

        lt, ls = timed_run(pagerank_list, graph, n)
        mt, ms = timed_run(pagerank_matrix, matrix, n)

        list_times.append(lt)
        matrix_times.append(mt)
        list_std.append(ls)
        matrix_std.append(ms)

        list_norm.append(lt / n)
        matrix_norm.append(mt / (n*n))

    # Raw runtime plot
    plt.figure()
    plt.errorbar(sizes, list_times, yerr=list_std, marker='o', label='Adjacency List')
    plt.errorbar(sizes, matrix_times, yerr=matrix_std, marker='o', label='Adjacency Matrix')
    plt.xlabel('Number of Nodes (n)')
    plt.ylabel('Runtime (seconds)')
    plt.title('Runtime vs Graph Size (Sparse Graphs)')
    plt.legend()
    plt.grid()
    plt.savefig('runtime_vs_size.png', dpi=300)
    plt.close()

    # Normalized runtime plot
    plt.figure()
    plt.plot(sizes, list_norm, marker='o', label='List (per node)')
    plt.plot(sizes, matrix_norm, marker='o', label='Matrix (per n^2)')
    plt.xlabel('Number of Nodes (n)')
    plt.ylabel('Normalized Runtime')
    plt.title('Normalized Runtime Comparison')
    plt.legend()
    plt.grid()
    plt.savefig('runtime_normalized.png', dpi=300)
    plt.close()

# Experiment 2: Runtime vs Density

def experiment_runtime_vs_density():
    densities = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5]
    n = 1000
    list_times, matrix_times = [], []

    for d in densities:
        graph = generate_dense_graph(n, d)
        matrix = to_adjacency_matrix(graph, n)

        lt, _ = timed_run(pagerank_list, graph, n)
        mt, _ = timed_run(pagerank_matrix, matrix, n)

        list_times.append(lt)
        matrix_times.append(mt)

    plt.figure()
    plt.plot(densities, list_times, marker='o', label='Adjacency List')
    plt.plot(densities, matrix_times, marker='o', label='Adjacency Matrix')
    plt.xlabel('Graph Density')
    plt.ylabel('Runtime (seconds)')
    plt.title('Runtime vs Graph Density (n = 1000)')
    plt.legend()
    plt.grid()
    plt.savefig('runtime_vs_density.png', dpi=300)
    plt.close()

# Experiment 3: Memory Comparison (Log Scale)

def estimate_memory(graph, matrix, n):
    list_memory = sum(len(v) for v in graph.values())
    matrix_memory = n * n
    return list_memory, matrix_memory


def experiment_memory():
    sizes = [100, 300, 600, 1000, 2000]
    list_memories, matrix_memories = [], []

    for n in sizes:
        graph = generate_sparse_graph(n)
        matrix = to_adjacency_matrix(graph, n)

        list_mem, matrix_mem = estimate_memory(graph, matrix, n)
        list_memories.append(list_mem)
        matrix_memories.append(matrix_mem)

    plt.figure()
    plt.plot(sizes, list_memories, marker='o', label='Adjacency List')
    plt.plot(sizes, matrix_memories, marker='o', label='Adjacency Matrix')
    plt.yscale('log')
    plt.xlabel('Number of Nodes (n)')
    plt.ylabel('Memory (log scale)')
    plt.title('Memory Usage Comparison (Log Scale)')
    plt.legend()
    plt.grid()
    plt.savefig('memory_comparison_log.png', dpi=300)
    plt.close()

# Run All

if __name__ == "__main__":
    experiment_runtime_vs_size()
    experiment_runtime_vs_density()
    experiment_memory()
