#include <cmath>
#include <map>
#include <string>
#include <vector>
#include <complex>
#include <random>
#include <algorithm>
#include "e4b.h"

//
// E4B implementation.
//

struct e4b_context_params e4b_default_context_params(void) {
    struct e4b_context_params result = {
            /*.n_embd                    =*/ DEFAULT_N_EMBD,
            /*.n_max                     =*/ uint32_t(DEFAULT_N_MAX),
            /*.similarity_search_target  =*/ DEFAULT_SIMILARITY_SEARCH,
            /*.fn_similarity             =*/ e4b_cosine_similarity,
            /*.n_hash               =*/ uint32_t(-1),
            /*.n_buckets_width           =*/ uint32_t(-1),
            /*.fn_hash                   =*/ e4b_lsh,
            /*.n_init_capacity           =*/ DEFAULT_INIT_CAPACITY,
            /*.grow_ratio                =*/ DEFAULT_GROW_RATIO,
            /*.use_persistent_storage    =*/ false,
            /*.persistent_storage_folder =*/ nullptr,
    };

    result.n_hash = (uint32_t) std::log(result.n_max);
    result.n_buckets_width = 2 * std::acos(result.similarity_search_target);

    return result;
}

// LSH function context.
struct e4b_lsh_context {
    e4b_lsh_context(uint32_t n_hash, uint32_t n_embd)
            : projections(n_hash, std::vector<float>(n_embd)), offsets(n_embd) {}

    // random projection vectors
    std::vector<std::vector<float>> projections;

    // random offsets
    std::vector<float> offsets;
};

//
// e4b context, holds the state of the db.
//
struct e4b_context {
    e4b_context(const e4b_context_params &cparams)
            : cparams(cparams), lsh_ctx(nullptr), entries(nullptr), size(0), capacity(0) {}

    ~e4b_context() {
        if (!cparams.use_persistent_storage) {
            if (lsh_ctx != nullptr) {
                delete lsh_ctx;
            }
            delete entries;
        }
    }

    const struct e4b_context_params cparams;

    std::map<std::string, std::vector<int>> index;

    e4b_lsh_context *lsh_ctx;

    e4b_entry *entries;
    uint32_t size;
    uint32_t capacity;
};

bool e4b_internal_grow(e4b_context *ctx);

struct e4b_context *e4b_init(struct e4b_context_params cparams) {
    auto ctx = new e4b_context(cparams);

    ctx->capacity = cparams.n_init_capacity;
    ctx->entries = new e4b_entry[ctx->capacity];
    ctx->lsh_ctx = new e4b_lsh_context(cparams.n_hash, cparams.n_embd);

    // Initialize the random projection vectors and offsets
    std::random_device rd; // FIXME only the first time
    std::mt19937 gen(rd());
    std::normal_distribution<double> nd(0.0, 1.0);
    std::uniform_real_distribution<double> ud(0.0, cparams.n_hash);
    for (uint32_t i = 0; i < cparams.n_hash; i++) {
        for (uint32_t j = 0; j < cparams.n_embd; j++) {
            ctx->lsh_ctx->projections[i][j] = (float) nd(gen);
        }
        ctx->lsh_ctx->offsets[i] = (float) ud(gen);
    }

    return ctx;
}

bool e4b_start(struct e4b_context *ctx) {
    return true;
}

bool e4b_stop(struct e4b_context *ctx) {
    delete ctx;
}

std::string e4b_internal_hash_to_key(const int *hash, uint32_t n_hash) {
    std::string key("0", n_hash);
    for (uint32_t i = 0; i < n_hash; ++i) {
        if (hash[i] > 0) {
            key[i] = '1';
        }
    }
    return key;
}

bool e4b_internal_grow(e4b_context *ctx) {
    if (ctx->cparams.use_persistent_storage) {
        // FIXME grow mmap
        return false;
    }

    ctx->capacity = (uint32_t) ctx->cparams.grow_ratio * ctx->capacity;
    auto *entries = new e4b_entry[ctx->capacity];

    for (uint32_t i = 0; i < ctx->size; ++i) {
        entries[i] = ctx->entries[i];
    }

    delete ctx->entries;
    ctx->entries = entries;

    return true;
}

uint32_t e4b_index(struct e4b_context *ctx, const char *text, uint32_t n_text, const float *embd) {
    // hash and get the bucket key index
    const auto &hash = ctx->cparams.fn_hash(ctx, embd, ctx->cparams.n_embd);
    const auto &key = e4b_internal_hash_to_key(hash, ctx->cparams.n_hash);
    delete hash;

    // grow if necessary
    if (ctx->capacity == ctx->size
        && !e4b_internal_grow(ctx)) {
        return -1;
    }

    // next index
    uint32_t idx = ctx->size++;

    // store the new entry
    ctx->entries[idx] = e4b_entry{
            text,
            embd,
            n_text
    };

    // index the entry
    ctx->index[key].push_back(idx);

    return idx;
}

struct e4b_query_results e4b_query(const struct e4b_context *ctx,
                                   const float *q_embd,
                                   uint32_t n_q_embd,
                                   uint32_t top_n,
                                   float similarity) {
    // hash and get the bucket key index
    const auto &q_hash = ctx->cparams.fn_hash(ctx, q_embd, n_q_embd);
    const auto &key = e4b_internal_hash_to_key(q_hash, ctx->cparams.n_hash);

    const auto bucket = ctx->index.at(key);
    if (bucket.empty()) {
        return e4b_query_results{
                /*.results   =*/nullptr,
                /*.n_results =*/0,
                /*.total     =*/0,
        };
    }

    const auto &entries = ctx->entries;

    std::vector<e4b_query_result_entry *> v_results(bucket.size());
    std::transform(bucket.begin(), bucket.end(), v_results.begin(),
                   [&](uint32_t idx) {
                       const auto &entry = ctx->entries[idx];
                       auto similarity = ctx->cparams.fn_similarity(q_embd,
                                                                    entries->embd,
                                                                    ctx->cparams.n_embd);
                       auto score = 1 / (1 + similarity);
                       return new e4b_query_result_entry{
                               /*.idx   =*/idx,
                               /*.entry =*/entry,
                               /*.score =*/score,
                       };
                   });
    std::sort(v_results.begin(), v_results.end(), [](const e4b_query_result_entry *a, const e4b_query_result_entry *b) {
        return a->score > b->score;
    });

    auto total = (uint32_t) v_results.size();
    auto capacity = std::min(top_n, total);
    uint32_t n_results = 0;
    auto results = new e4b_query_result_entry *[capacity];
    for (uint32_t i = 0; i < capacity; ++i) {
        const auto &result = v_results[i];
        if (result->score >= similarity) {
            results[n_results++] = result;
        }
    }

    delete q_hash;
    return e4b_query_results{
            *results,
            n_results,
            total
    };
}

const int *e4b_lsh(const e4b_context *ctx, const float *embd, uint32_t n_embd) {
    auto n_hash = ctx->cparams.n_hash;
    int *hash = new int[n_hash];
    for (uint32_t i = 0; i < n_hash; i++) {
        float dot = 0.0;
        for (uint32_t j = 0; j < n_embd; j++) {
            dot += embd[j] * ctx->lsh_ctx->projections[i][j];
        }
        int hi = (int) std::abs(std::floor((dot + ctx->lsh_ctx->offsets[i]) / (float) n_hash));
        hash[i] = hi % 2 == 1;
    }
    return hash;
}

const void *e4b_lsh_ctx(const e4b_context *ctx) {
    return ctx->lsh_ctx;
}

float e4b_cosine_similarity(const float *embd_a, const float *embd_b, uint32_t n_embd) {
    float dot = 0;
    float norm1 = 0;
    float norm2 = 0;
    for (uint32_t i = 0; i < n_embd; i++) {
        dot += embd_a[i] * embd_b[i];
        norm1 += embd_a[i] * embd_a[i];
        norm2 += embd_b[i] * embd_b[i];
    }
    norm1 = std::sqrt(norm1);
    norm2 = std::sqrt(norm2);
    return dot / (norm1 * norm2);
}