#ifndef GRAPH_H
#define GRAPH_H

#include "bench.h"
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <vector>

const std::chrono::nanoseconds BASELINE_GRAPH{5000000};
inline constexpr double NAIVE_SPEEDUP_LOWER_BOUND_GRAPH{2.50};

struct Edge {
    int to;
    Edge* next;
};

struct Node {
    Edge* edges;
};

struct Graph {
    int n;
    Node* nodes;
};

// CSR (Compressed Sparse Row) format for better cache locality
struct GraphCSR {
    int n;  // number of nodes
    std::vector<int> offsets;  // offsets[i] = start index in to array for node i
    std::vector<int> to;       // all destination nodes
};

struct graph_args {
    Graph graph;
    GraphCSR graph_csr;  // CSR representation for optimized access
    std::vector<Node> nodes;
    std::vector<Edge> edge_storage;
    std::uint64_t out;
    double epsilon;
    // TODO: You may want to add new params at the end...

    explicit graph_args(double epsilon_in = 1e-6)
        : graph{0, nullptr}, out{0}, epsilon{epsilon_in} {}
};

void naive_graph(std::uint64_t& out, const Graph& graph);

// Convert graph from adjacency list to CSR format (not included in timing)
void convert_graph_to_csr(GraphCSR& csr, const Graph& graph);

// Optimized version using CSR format
void stu_graph(std::uint64_t& out, const GraphCSR& graph_csr);

void naive_graph_wrapper(void* ctx);
void stu_graph_wrapper(void* ctx);

void initialize_graph(graph_args* args,
                       std::size_t node_count,
                       int avg_degree,
                       std::uint_fast64_t seed);

bool graph_check(void* stu_ctx, void* ref_ctx, lab_test_func naive_func);

#endif
