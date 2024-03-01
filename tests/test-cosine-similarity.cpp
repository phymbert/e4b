
#include "e4b.h"
#include <tuple>
#include <vector>
#include <cassert>
#include <cstdio>
#include <cmath>

int main() {
    e4b_similarity_fn_t fn = e4b_cosine_similarity;

    std::vector<std::pair<float, std::pair<std::vector<float>, std::vector<float>>>> tests = {
            {1.000000, {{0, 1, 2, 3}, {0, 1, 2, 3}}},
            {0.956183, {{0, 1, 2, 3}, {0, 0, 1, 2}}}
    };

    for (auto &test: tests) {
        auto expected_similarity = test.first;
        auto embd_a = test.second.first;
        auto embd_b = test.second.second;

        auto similarity = fn(embd_a.data(), embd_b.data(), 4);

        auto d = std::fabs(expected_similarity - similarity);
        auto precision = 1e-3;

        // pretty print error message before asserting
        if (d > precision) {
            fprintf(stderr, "expected_similarity: %f\n", expected_similarity);
            fprintf(stderr, "actual_similarity: %f\n", similarity);
        }
        assert(std::fabs(expected_similarity - similarity) <= precision);
    }
}