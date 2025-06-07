#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>

#define BLOCK_SIZE 256
#define VECTOR_DIM 768

__global__ void compute_cosine_similarity(
    const float* query_vectors,
    const float* database_vectors,
    float* similarities,
    int num_queries,
    int num_database,
    int dim
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= num_queries * num_database) {
        return;
    }

    int query_idx = tid / num_database;
    int db_idx = tid % num_database;

    // Compute dot product
    float dot_product = 0.0f;
    float query_norm = 0.0f;
    float db_norm = 0.0f;

    for (int i = 0; i < dim; i++) {
        float q_val = query_vectors[query_idx * dim + i];
        float d_val = database_vectors[db_idx * dim + i];

        dot_product += q_val * d_val;
        query_norm += q_val * q_val;
        db_norm += d_val * d_val;
    }

    // Compute cosine similarity
    float similarity = dot_product / (sqrtf(query_norm) * sqrtf(db_norm) + 1e-8f);
    similarities[tid] = similarity;
}

class GPUSimilaritySearch {
private:
    cublasHandle_t cublas_handle;
    cudaStream_t stream;

    // Device memory pools
    float* d_database_vectors;
    float* d_query_vectors;
    float* d_similarities;
    int* d_indices;

    size_t database_size;
    size_t max_batch_size;

public:
    GPUSimilaritySearch(size_t db_size, size_t batch_size) 
        : database_size(db_size), max_batch_size(batch_size) {

        // Initialize CUDA
        cublasCreate(&cublas_handle);
        cudaStreamCreate(&stream);

        // Allocate device memory
        size_t db_bytes = database_size * VECTOR_DIM * sizeof(float);
        size_t query_bytes = max_batch_size * VECTOR_DIM * sizeof(float);
        size_t sim_bytes = max_batch_size * database_size * sizeof(float);
        size_t idx_bytes = max_batch_size * database_size * sizeof(int);

        cudaMalloc(&d_database_vectors, db_bytes);
        cudaMalloc(&d_query_vectors, query_bytes);
        cudaMalloc(&d_similarities, sim_bytes);
        cudaMalloc(&d_indices, idx_bytes);
    }

    void load_database(const std::vector<float>& vectors) {
        cudaMemcpyAsync(
            d_database_vectors,
            vectors.data(),
            vectors.size() * sizeof(float),
            cudaMemcpyHostToDevice,
            stream
        );
    }

    std::vector<SearchResult> search_batch(
        const std::vector<float>& queries,
        int top_k
    ) {
        int num_queries = queries.size() / VECTOR_DIM;

        // Copy queries to device
        cudaMemcpyAsync(
            d_query_vectors,
            queries.data(),
            queries.size() * sizeof(float),
            cudaMemcpyHostToDevice,
            stream
        );

        // Compute similarities
        int total_comparisons = num_queries * database_size;
        int num_blocks = (total_comparisons + BLOCK_SIZE - 1) / BLOCK_SIZE;

        compute_cosine_similarity<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            d_query_vectors,
            d_database_vectors,
            d_similarities,
            num_queries,
            database_size,
            VECTOR_DIM
        );

        // Find top-k for each query
        std::vector<SearchResult> results(num_queries);

        for (int q = 0; q < num_queries; q++) {
            // Use thrust to sort similarities for this query
            thrust::device_ptr<float> sim_ptr(d_similarities + q * database_size);
            thrust::device_ptr<int> idx_ptr(d_indices + q * database_size);

            // Initialize indices
            thrust::sequence(idx_ptr, idx_ptr + database_size);

            // Sort by similarity (descending)
            thrust::sort_by_key(
                sim_ptr,
                sim_ptr + database_size,
                idx_ptr,
                thrust::greater<float>()
            );

            // Copy top-k results
            std::vector<float> top_similarities(top_k);
            std::vector<int> top_indices(top_k);

            cudaMemcpy(
                top_similarities.data(),
                d_similarities + q * database_size,
                top_k * sizeof(float),
                cudaMemcpyDeviceToHost
            );

            cudaMemcpy(
                top_indices.data(),
                d_indices + q * database_size,
                top_k * sizeof(int),
                cudaMemcpyDeviceToHost
            );

            results[q] = SearchResult{
                query_id: q,
                indices: top_indices,
                similarities: top_similarities
            };
        }

        cudaStreamSynchronize(stream);
        return results;
    }
};
