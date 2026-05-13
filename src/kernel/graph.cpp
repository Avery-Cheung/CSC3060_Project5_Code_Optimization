#include "graph.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>


void initialize_graph(graph_args* args,
                       std::size_t node_count,
                       int avg_degree,
                       std::uint_fast64_t seed) {
    if (!args) {
        return;
    }

    std::mt19937_64 gen(seed);
    std::uniform_int_distribution<int> dist(0, static_cast<int>(node_count) - 1);

    args->nodes.assign(node_count, Node{nullptr});
    args->edge_storage.clear();
    args->edge_storage.resize(node_count * static_cast<std::size_t>(avg_degree));

    args->graph.n = static_cast<int>(node_count);
    args->graph.nodes = args->nodes.data();

    std::size_t edge_pos = 0;

    for (std::size_t u = 0; u < node_count; ++u) {
        std::vector<int> neighbors;
        neighbors.reserve(avg_degree);

        for (int k = 0; k < avg_degree; ++k) {
            neighbors.push_back(dist(gen));
        }

        Edge* head = nullptr;
        for (int k = avg_degree - 1; k >= 0; --k) {
            Edge& e = args->edge_storage[edge_pos + static_cast<std::size_t>(k)];
            e.to = neighbors[static_cast<std::size_t>(k)];
            e.next = head;
            head = &e;
        }

        args->nodes[u].edges = head;
        edge_pos += static_cast<std::size_t>(avg_degree);
    }

    args->out = 0;
}

void naive_graph(std::uint64_t& out, const Graph& graph) {
    std::uint64_t checksum = 0;
    for (int u = 0; u < graph.n; ++u) {
        const Edge* e = graph.nodes[u].edges;
        while (e) {
            checksum += static_cast<std::uint64_t>(e->to);
            e = e->next;
        }
    }
    out = checksum;
}

// Convert from adjacency list to CSR format
void convert_graph_to_csr(GraphCSR& csr, const Graph& graph) {
    csr.n = graph.n;
    csr.offsets.resize(graph.n + 1);
    csr.to.clear();

    // First pass: count edges per node to build offsets
    std::vector<int> degree(graph.n, 0);
    for (int u = 0; u < graph.n; ++u) {
        const Edge* e = graph.nodes[u].edges;
        while (e) {
            degree[u]++;
            e = e->next;
        }
    }

    // Build offsets array
    csr.offsets[0] = 0;
    for (int u = 0; u < graph.n; ++u) {
        csr.offsets[u + 1] = csr.offsets[u] + degree[u];
    }

    // Reserve space for destinations
    csr.to.resize(csr.offsets[graph.n]);

    // Second pass: fill the to array
    std::vector<int> curr_offset = csr.offsets;
    for (int u = 0; u < graph.n; ++u) {
        const Edge* e = graph.nodes[u].edges;
        while (e) {
            csr.to[curr_offset[u]] = e->to;
            curr_offset[u]++;
            e = e->next;
        }
    }
}

void stu_graph(std::uint64_t& out, const GraphCSR& graph_csr) {
    std::uint64_t checksum = 0;
    for (int u = 0; u < graph_csr.n; ++u) {
        const int start = graph_csr.offsets[u];
        const int end = graph_csr.offsets[u + 1];
        for (int i = start; i < end; ++i) {
            checksum += static_cast<std::uint64_t>(graph_csr.to[i]);
        }
    }
    out = checksum;
}

void naive_graph_wrapper(void* ctx) {
    auto& args = *static_cast<graph_args*>(ctx);
    naive_graph(args.out, args.graph);
}

void stu_graph_wrapper(void* ctx) {
    auto& args = *static_cast<graph_args*>(ctx);
    stu_graph(args.out, args.graph_csr);
}

bool graph_check(void* stu_ctx, void* ref_ctx, lab_test_func naive_func) {
    naive_func(ref_ctx);

    auto& stu_args = *static_cast<graph_args*>(stu_ctx);
    auto& ref_args = *static_cast<graph_args*>(ref_ctx);
    const auto eps = ref_args.epsilon;

    const double s = static_cast<double>(stu_args.out);
    const double r = static_cast<double>(ref_args.out);
    const double err = std::abs(s - r);
    const double atol = 0.0;
    const double rel = (std::abs(r) > 1e-12) ? err / std::abs(r) : err;

    debug_log("\tDEBUG: graph stu={} ref={} err={} rel={}\n",
              stu_args.out,
              ref_args.out,
              err,
              rel);

    return err <= (atol + eps * std::abs(r));
}
