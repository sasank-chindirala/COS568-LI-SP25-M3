#include "benchmarks/benchmark_pgm.h"

#include "benchmark.h"
#include "benchmarks/common.h"
#include "competitors/pgm_index.h"

template <typename Searcher>
void benchmark_64_pgm(tli::Benchmark<uint64_t>& benchmark, 
                      bool pareto, const std::vector<int>& params) {
  if (!pareto){
    util::fail("PGM's hyperparameter cannot be set");
  }
  else {
    benchmark.template Run<PGM<uint64_t, Searcher, 4>>();
    benchmark.template Run<PGM<uint64_t, Searcher, 8>>();
    benchmark.template Run<PGM<uint64_t, Searcher, 16>>();
    benchmark.template Run<PGM<uint64_t, Searcher, 32>>();
    benchmark.template Run<PGM<uint64_t, Searcher, 64>>();
    benchmark.template Run<PGM<uint64_t, Searcher, 128>>();
    benchmark.template Run<PGM<uint64_t, Searcher, 256>>();
  }
}

template <int record>
void benchmark_64_pgm(tli::Benchmark<uint64_t>& benchmark, const std::string& filename) {
  if (filename.find("books_100M") != std::string::npos) {
    benchmark.template Run<PGM<uint64_t, LinearSearch<record>,32>>();
    benchmark.template Run<PGM<uint64_t, BranchingBinarySearch<record>,64>>();
    benchmark.template Run<PGM<uint64_t, BranchingBinarySearch<record>,256>>();
  }
  if (filename.find("fb_100M") != std::string::npos) {
    benchmark.template Run<PGM<uint64_t, LinearSearch<record>,16>>();
    benchmark.template Run<PGM<uint64_t, LinearSearch<record>,8>>();
    benchmark.template Run<PGM<uint64_t, LinearSearch<record>,32>>();
  }
  if (filename.find("osmc_100M") != std::string::npos) {
    benchmark.template Run<PGM<uint64_t, BranchingBinarySearch<record>,64>>();
    benchmark.template Run<PGM<uint64_t, BranchingBinarySearch<record>,128>>();
    benchmark.template Run<PGM<uint64_t, LinearSearch<record>,64>>();
  }
  if (filename.find("wiki_100M") != std::string::npos) {
    benchmark.template Run<PGM<uint64_t, LinearSearch<record>,32>>();
    benchmark.template Run<PGM<uint64_t, ExponentialSearch<record>,8>>();
    benchmark.template Run<PGM<uint64_t, BranchingBinarySearch<record>,64>>();
  }
}

INSTANTIATE_TEMPLATES_MULTITHREAD(benchmark_64_pgm, uint64_t);