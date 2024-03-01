//
// e4b: Embeddings lightweight database library.
//
// License: MIT
// Author: @phymbert - Pierrick HYMBERT
//
#ifndef E4B_H
#define E4B_H

#include <stdint.h>
#include <stdbool.h>

#ifdef E4B_SHARED
#    if defined(_WIN32) && !defined(__MINGW32__)
#        ifdef E4B_BUILD
#            define E4B_API __declspec(dllexport)
#        else
#            define E4B_API __declspec(dllimport)
#        endif
#    else
#        define E4B_API __attribute__ ((visibility ("default")))
#    endif
#else
#    define E4B_API
#endif

#ifdef __GNUC__
#    define DEPRECATED(func, hint) func __attribute__((deprecated(hint)))
#elif defined(_MSC_VER)
#    define DEPRECATED(func, hint) __declspec(deprecated(hint)) func
#else
#    define DEPRECATED(func, hint) func
#endif

#define DEFAULT_N_MAX             1e6
#define DEFAULT_N_EMBD            1024
#define DEFAULT_SIMILARITY_SEARCH 0.8
#define DEFAULT_INIT_CAPACITY     1000
#define DEFAULT_GROW_RATIO        2
#define DEFAULT_TOP_N             10

#define E4B_DEFAULT_SEED 0xFFFFFFFF

#define E4B_SESSION_MAGIC   E4B_FILE_MAGIC_GGSN
#define E4B_SESSION_VERSION 4

#ifdef __cplusplus
extern "C" {
#endif

struct e4b_context;

// Type of pointer to the hash function.
typedef const int *(*e4b_hash_fn_t)(const struct e4b_context *ctx, const float *embd, uint32_t n_embd);

// Type of pointer to the hash function context provider.
typedef const void *(*e4b_hash_ctx_fn_t)(const struct e4b_context *ctx);

// Type of pointer to the similarity function.
typedef float (*e4b_similarity_fn_t)(const float *embd_a, const float *embd_b, uint32_t n_embd);

//
// embedding database params.
//
// Choosing the parameters n_hash and n_buckets_width for LSH depends on the data distribution [3],
// the similarity measure, and the desired trade-off between accuracy and efficiency.
// There is no definitive answer, but some general guidelines are:
//
//      - n_hash should be large enough to reduce the number of false positives,
//          but not too large to increase the number of false negatives.
//          A common choice is n_hash= log(n_max), where n_max is the number of vectors in the dataset[1].
//      - n_buckets_width should be proportional to the average distance between similar vectors,
//          so that they are more likely to fall into the same bucket.
//          A common choice is n_buckets_width = 2r, where r is the radius of the similarity_search_target [2].
// [1] https://link.springer.com/chapter/10.1007/978-3-319-22174-8_6
// [2] https://www.pinecone.io/learn/series/faiss/locality-sensitive-hashing/
// [3] http://madscience.ucsd.edu/2020/notes/lec13.pdf
struct e4b_context_params {
    // 1-dimensional vector size of embeddings returned by the model.
    // shape: [n_embd].
    uint32_t n_embd;

    // Max number of embeddings in the dataset,
    // can exceed the DB size but the hash functions may not work as expected.
    uint32_t n_max;

    // Similarity search target.
    float similarity_search_target;

    // Similarity function.
    e4b_similarity_fn_t fn_similarity;

    // The number of hash bits.
    uint32_t n_hash;

    // The width of the buckets.
    uint32_t n_buckets_width;

    // Hash function.
    e4b_hash_fn_t fn_hash;

    // Initial capacity of the database.
    int n_init_capacity;

    // Grow ratio if size reached capacity
    float grow_ratio;

    bool use_persistent_storage;           // If the DB is persisted to disk.
    const char *persistent_storage_folder; // Folder path where the DB data, config and files will be stored.
};

// Embedding DB entry
typedef struct e4b_entry {
    const char *text;
    const float *embd;

    uint32_t n_text;
} e4b_entry;

// query result with entry and search score.
struct e4b_query_result_entry {
    const uint32_t idx;
    const struct e4b_entry entry;
    const float score;
};

// query results.
struct e4b_query_results {
    const struct e4b_query_result_entry *results;
    const uint32_t n_results;
    const uint32_t total;
};

// Helpers for getting default embedding database parameters
struct e4b_context_params e4b_default_context_params(void);

// Initialize the embedding database context.
E4B_API struct e4b_context *e4b_init(struct e4b_context_params cparams);

// Start the embedding database,
E4B_API bool e4b_start(struct e4b_context *ctx);

// Stop the embedding database, release underlying resources.
E4B_API bool e4b_stop(struct e4b_context *ctx);

// Index the embedding and store the text.
E4B_API uint32_t e4b_index(struct e4b_context *ctx, const char *text, uint32_t n_text, const float *embd);

// Query the database against the provided input string.
E4B_API struct e4b_query_results e4b_query(const struct e4b_context *ctx,
                                          const float *q_embd,
                                          uint32_t n_q_embd,
                                          uint32_t top_n,
                                          float similarity);

// Free query results underlying results.
E4B_API void e4b_free_query_results(struct e4b_query_results);

// Locality Sensitive Hashing function.
E4B_API const int *e4b_lsh(const struct e4b_context *ctx, const float *embd, uint32_t n_embd);

// Cosine similarity function.
E4B_API float e4b_cosine_similarity(const float *embd_a, const float *embd_b, uint32_t n_embd);


#ifdef __cplusplus
}
#endif

// Internal API to be implemented by e4b.cpp and used by tests/benchmarks only
#ifdef E4B_API_INTERNAL

#include <vector>
#include <string>

#endif // E4B_API_INTERNAL

#endif // E4B_H