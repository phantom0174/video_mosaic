#include <turbojpeg.h> 

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#ifdef __NVCC__
#pragma nv_diag_suppress 170 

#endif

#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize2.h"

#ifdef __NVCC__
#pragma nv_diag_default 170
#endif

#include "common.h"
#include "timer.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <thread>
#include <mutex>
#include <atomic>
#include <algorithm>
#include <queue>
#include <condition_variable>
#include <cfloat>
#include <cstdlib>
#include <cstdio>
#include <cstring>

namespace fs = std::filesystem;

#define CHECK_CUDA(call) { \
    const cudaError_t error = call; \
    if (error != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(error) << " at line " << __LINE__ << std::endl; \
        exit(1); \
    } \
}

#define CHECK_CUBLAS(call) { \
    if (call != CUBLAS_STATUS_SUCCESS) { \
        std::cerr << "CUBLAS Error at line " << __LINE__ << std::endl; \
        exit(1); \
    } \
}

const int NUM_STREAMS = 4;
const int NUM_WRITER_THREADS = 16;
const int JPEG_QUALITY = 85;

int g_rank = 0;
int g_world_size = 1;
int g_local_rank = 0;
std::mutex print_mutex;

std::vector<unsigned char*> g_pinned_pool;
std::queue<int> g_free_buffer_ids; 
std::mutex g_buffer_mutex;
std::condition_variable g_buffer_cv;

struct WriteTask {
    unsigned char* raw_ptr; 

    int buffer_id;          

    std::string filename;
};

std::queue<WriteTask> g_write_queue;
std::mutex g_queue_mutex;
std::condition_variable g_queue_cv;
std::atomic<bool> g_finished(false);
std::atomic<int> g_pending_writes(0);

void init_pinned_pool(size_t img_bytes) {

    int pool_size = NUM_WRITER_THREADS + NUM_STREAMS + 8;
    for (int i = 0; i < pool_size; ++i) {
        unsigned char* ptr;

        CHECK_CUDA(cudaMallocHost(&ptr, img_bytes));
        g_pinned_pool.push_back(ptr);
        g_free_buffer_ids.push(i);
    }
}

void free_pinned_pool() {

    for (auto* ptr : g_pinned_pool) cudaFreeHost(ptr);
}

void turbo_save_worker() {

    tjhandle compressor = tjInitCompress();
    if (!compressor) {
        std::cerr << "TurboJPEG init failed!" << std::endl;
        return;
    }

    while (true) {
        WriteTask task;
        {
            std::unique_lock<std::mutex> lock(g_queue_mutex);
            g_queue_cv.wait(lock, []{ return !g_write_queue.empty() || g_finished; });

            if (g_write_queue.empty() && g_finished) break;

            task = std::move(g_write_queue.front());
            g_write_queue.pop();
        }

        unsigned char* jpegBuf = nullptr; 

        unsigned long jpegSize = 0;

        int ret = tjCompress2(compressor, task.raw_ptr, Cfg::FINAL_W, 0, Cfg::FINAL_H, TJPF_RGB,
            &jpegBuf, &jpegSize, TJSAMP_420, JPEG_QUALITY, TJFLAG_FASTDCT);

        if (ret == 0) {
            FILE* file = fopen(task.filename.c_str(), "wb");
            if (file) {
                fwrite(jpegBuf, 1, jpegSize, file);
                fclose(file);
            }
            tjFree(jpegBuf); 

        } else {
            std::cerr << "JPEG Compress Error: " << tjGetErrorStr2(compressor) << std::endl;
        }

        {
            std::lock_guard<std::mutex> lock(g_buffer_mutex);
            g_free_buffer_ids.push(task.buffer_id);
            g_pending_writes--;
        }
        g_buffer_cv.notify_one();
    }
    tjDestroy(compressor);
}

struct FrameResources {
    cudaStream_t stream; 

    unsigned char* d_raw_input_img;
    float* d_input_blocks;
    float* d_gemm_result;
    int* d_best_indices;
    unsigned char* d_output_img;

    unsigned char* h_pinned_input; 
    unsigned char* h_pinned_output; 

    void allocate(size_t in_bytes, size_t out_bytes, int num_blocks, int pixel_dim, int loaded_tiles) {
        CHECK_CUDA(cudaStreamCreate(&stream));
        CHECK_CUDA(cudaMallocHost(&h_pinned_input, in_bytes));
        CHECK_CUDA(cudaMallocHost(&h_pinned_output, out_bytes));

        CHECK_CUDA(cudaMalloc(&d_raw_input_img, in_bytes));
        CHECK_CUDA(cudaMalloc(&d_input_blocks, num_blocks * pixel_dim * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_gemm_result, loaded_tiles * num_blocks * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_best_indices, num_blocks * sizeof(int)));
        CHECK_CUDA(cudaMalloc(&d_output_img, out_bytes));
    }

    void free() {
        cudaFreeHost(h_pinned_input); cudaFreeHost(h_pinned_output);
        cudaFree(d_raw_input_img); cudaFree(d_input_blocks);
        cudaFree(d_gemm_result); cudaFree(d_best_indices); cudaFree(d_output_img);
        cudaStreamDestroy(stream);
    }
};

struct LibraryCacheHeader {
    int num_tiles; int pixel_dim; int plot_tile_size; unsigned int magic;
};

class LibraryCacheManager {
public:
    bool load(int& out_count, std::vector<float>& features, std::vector<float>& sq_sums, std::vector<unsigned char>& pixels) {
        if (!fs::exists(Cfg::CACHE_FILE)) return false;
        FILE* fp = fopen(Cfg::CACHE_FILE.c_str(), "rb");
        if (!fp) return false;
        LibraryCacheHeader h;
        if (fread(&h, sizeof(h), 1, fp) != 1) { fclose(fp); return false; }

        if (h.magic != Cfg::MAGIC || h.pixel_dim != Cfg::PIXEL_DIM) { 
            std::cerr << "Cache mismatch! Found dim=" << h.pixel_dim << ", Expected=" << Cfg::PIXEL_DIM << std::endl;
            fclose(fp); return false; 
        }

        out_count = h.num_tiles;
        features.resize(h.num_tiles * Cfg::PIXEL_DIM);
        sq_sums.resize(h.num_tiles);
        pixels.resize(h.num_tiles * Cfg::PLOT_SIZE);

        size_t r1 = fread(features.data(), sizeof(float), features.size(), fp);
        size_t r2 = fread(sq_sums.data(), sizeof(float), sq_sums.size(), fp);
        size_t r3 = fread(pixels.data(), sizeof(unsigned char), pixels.size(), fp);

        fclose(fp);

        if (r1 != features.size() || r2 != sq_sums.size() || r3 != pixels.size()) {
            std::cerr << "Error: Cache file read incomplete!" << std::endl;
            return false;
        }

        return true;
    }
};

struct PreloadedTarget {
    std::string serial;
    std::vector<unsigned char> raw_pixels;
    bool valid = false;
};

__global__ void preprocess_target_kernel(
    const unsigned char* __restrict__ input_img, float* __restrict__ output_blocks, 
    int img_w, int thumb_w, int thumb_h, int w_size, int pixel_dim)
{
    int bx = blockIdx.x; int by = blockIdx.y; int block_id = by * w_size + bx;
    int start_x = bx * thumb_w; int start_y = by * thumb_h; int num_pixels = thumb_w * thumb_h;

    int tid = threadIdx.x;

    for (int i = tid; i < num_pixels; i += blockDim.x) {
        int py = i / thumb_w; int px = i % thumb_w;
        int src_idx = ((start_y + py) * img_w + (start_x + px)) * 3;
        float r = (float)input_img[src_idx + 0];
        float g = (float)input_img[src_idx + 1];
        float b = (float)input_img[src_idx + 2];
        int dst_idx = (block_id * pixel_dim) + (i * 3);
        output_blocks[dst_idx + 0] = r;
        output_blocks[dst_idx + 1] = g;
        output_blocks[dst_idx + 2] = b;
    }
}

__global__ void find_best_match_reduction_v4_unroll(
    const float* __restrict__ gemm_results, 
    const float* __restrict__ tile_sq_sums, 
    int* __restrict__ output_indices, 
    int num_tiles,   
    int num_blocks) 
{
    int block_id = blockIdx.x;
    if (block_id >= num_blocks) return;

    const float* current_gemm_col = gemm_results + (block_id * num_tiles);

    int tid = threadIdx.x;

    float local_min_val = FLT_MAX;
    int local_best_idx = 0;

    int i = tid;
    while (i + 3 * blockDim.x < num_tiles) {

        int idx0 = i;
        int idx1 = i + blockDim.x;
        int idx2 = i + 2 * blockDim.x;
        int idx3 = i + 3 * blockDim.x;

        float g0 = current_gemm_col[idx0];
        float t0 = tile_sq_sums[idx0];

        float g1 = current_gemm_col[idx1];
        float t1 = tile_sq_sums[idx1];

        float g2 = current_gemm_col[idx2];
        float t2 = tile_sq_sums[idx2];

        float g3 = current_gemm_col[idx3];
        float t3 = tile_sq_sums[idx3];

        float val0 = t0 + g0;
        float val1 = t1 + g1;
        float val2 = t2 + g2;
        float val3 = t3 + g3;

        if (val0 < local_min_val) { local_min_val = val0; local_best_idx = idx0; }
        if (val1 < local_min_val) { local_min_val = val1; local_best_idx = idx1; }
        if (val2 < local_min_val) { local_min_val = val2; local_best_idx = idx2; }
        if (val3 < local_min_val) { local_min_val = val3; local_best_idx = idx3; }

        i += 4 * blockDim.x;
    }

    while (i < num_tiles) {
        float val = tile_sq_sums[i] + current_gemm_col[i];
        if (val < local_min_val) {
            local_min_val = val;
            local_best_idx = i;
        }
        i += blockDim.x;
    }

    extern __shared__ char smem[];
    float* s_val = (float*)smem;
    int* s_idx = (int*)&s_val[blockDim.x];

    s_val[tid] = local_min_val;
    s_idx[tid] = local_best_idx;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (s_val[tid + s] < s_val[tid]) {
                s_val[tid] = s_val[tid + s];
                s_idx[tid] = s_idx[tid + s];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        output_indices[block_id] = s_idx[0];
    }
}

__global__ void stitch_mosaic_kernel(
    const int* __restrict__ indices, const unsigned char* __restrict__ tile_pixels, unsigned char* __restrict__ output_img, 
    int grid_w, int tile_w, int tile_h, int final_w) 
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= final_w || y >= (grid_w * tile_h)) return; 

    int bx = x / tile_w;
    int by = y / tile_h;
    int lx = x % tile_w;
    int ly = y % tile_h; 

    int block_idx = by * grid_w + bx;
    int tile_idx = indices[block_idx];

    int src_offset = (tile_idx * (tile_w * tile_h * 3)) + (ly * tile_w + lx) * 3;
    int dst_offset = (y * final_w + x) * 3;

    output_img[dst_offset + 0] = tile_pixels[src_offset + 0];
    output_img[dst_offset + 1] = tile_pixels[src_offset + 1];
    output_img[dst_offset + 2] = tile_pixels[src_offset + 2];
}

inline int h_ceil(int a, int b) { return (a + b - 1) / b; }

struct FileEntry { int id; std::string path; std::string serial; };

inline bool has_dot_and_all_is_number(const std::string& s) {
    size_t dot_pos = s.find_last_of('.');
    if (dot_pos == std::string::npos) return false;
    return std::all_of(s.begin(), s.begin()+dot_pos, ::isdigit);
}

std::vector<FileEntry> scan_directory_sharded(const std::string& dir_path, int min_id, int max_id) {
    if (!fs::exists(dir_path)) return {};
    std::vector<FileEntry> results;

    for (const auto& entry : fs::directory_iterator(dir_path)) {
        if (!entry.is_regular_file()) continue;

        std::string filename = entry.path().filename().string();
        try {
            size_t dot_pos = filename.find_last_of('.');
            if (!has_dot_and_all_is_number(filename)) continue;

            int id = std::stoi(filename.substr(0, dot_pos));
            if (id < min_id || id > max_id) continue;
            if (id % g_world_size != g_rank) continue;

            results.push_back({id, entry.path().string(), filename.substr(0, dot_pos)});
        } catch (...) {}
    }
    std::sort(
        results.begin(), results.end(),
        [](const FileEntry& a, const FileEntry& b) { return a.id < b.id; });

    return results;
}

int main() {
    if (const char* env_rank = std::getenv("SLURM_PROCID")) g_rank = std::stoi(env_rank);
    if (const char* env_size = std::getenv("SLURM_NTASKS")) g_world_size = std::stoi(env_size);
    if (const char* env_local = std::getenv("SLURM_LOCALID")) g_local_rank = std::stoi(env_local);

    int num_gpus = 0;
    CHECK_CUDA(cudaGetDeviceCount(&num_gpus));
    CHECK_CUDA(cudaSetDevice(g_local_rank % num_gpus));
    if (g_rank == 0) std::cout << ">>> Hybrid Mosaic Generator (TurboJPEG + CUDA + cuBLAS) <<<" << std::endl;

    if (!fs::exists(Cfg::OUTPUT_DIR)) fs::create_directories(Cfg::OUTPUT_DIR);

    size_t in_img_bytes = Cfg::W * Cfg::H * 3;
    size_t out_img_bytes = Cfg::FINAL_W * Cfg::FINAL_H * 3;

    if (g_rank == 0) std::cout << "[Phase 1] Loading Cache..." << std::endl;

    std::vector<float> h_features, h_tile_sq_sums;
    std::vector<unsigned char> h_pixels;
    int loaded_count = 0;
    LibraryCacheManager cache_mgr;
    if (!cache_mgr.load(loaded_count, h_features, h_tile_sq_sums, h_pixels)) {
        if (g_rank == 0) std::cout << "Cache miss or invalid. Please run make_cache first." << std::endl; return 1;
    }

    cudaStream_t mem_stream; CHECK_CUDA(cudaStreamCreate(&mem_stream)); 

    cublasHandle_t blas_handle; CHECK_CUBLAS(cublasCreate(&blas_handle));

    float *d_features, *d_tile_sq_sums; unsigned char *d_pixels;
    CHECK_CUDA(cudaMalloc(&d_features, h_features.size() * sizeof(float)));
    CHECK_CUDA(cudaMemcpyAsync(d_features, h_features.data(), h_features.size() * sizeof(float), cudaMemcpyHostToDevice, mem_stream));
    CHECK_CUDA(cudaMalloc(&d_tile_sq_sums, h_tile_sq_sums.size() * sizeof(float)));
    CHECK_CUDA(cudaMemcpyAsync(d_tile_sq_sums, h_tile_sq_sums.data(), h_tile_sq_sums.size() * sizeof(float), cudaMemcpyHostToDevice, mem_stream));
    CHECK_CUDA(cudaMalloc(&d_pixels, h_pixels.size() * sizeof(unsigned char)));
    CHECK_CUDA(cudaMemcpyAsync(d_pixels, h_pixels.data(), h_pixels.size() * sizeof(unsigned char), cudaMemcpyHostToDevice, mem_stream));

    init_pinned_pool(out_img_bytes);
    std::vector<std::thread> thread_pool;
    for(int i = 0; i < NUM_WRITER_THREADS; ++i) thread_pool.emplace_back(turbo_save_worker);

    std::vector<FileEntry> work_items = scan_directory_sharded(Cfg::FIT_DIR, 1, 6571);
    size_t my_tasks = work_items.size();
    if (g_rank == 0) std::cout << "[Phase 2] Pre-loading " << my_tasks << " targets..." << std::endl;

    std::vector<PreloadedTarget> preloaded_targets(my_tasks);
    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < my_tasks; ++i) {
        int w, h, c;
        unsigned char* data = stbi_load(work_items[i].path.c_str(), &w, &h, &c, 3);
        if (!data) continue; 

        preloaded_targets[i].serial = work_items[i].serial;
        preloaded_targets[i].raw_pixels.resize(in_img_bytes);

        if (w == Cfg::W || h == Cfg::H) {
            memcpy(preloaded_targets[i].raw_pixels.data(), data, in_img_bytes);
        } else {
            stbir_resize_uint8_linear(data, w, h, 0, preloaded_targets[i].raw_pixels.data(), Cfg::W, Cfg::H, 0, (stbir_pixel_layout)3);
        }
        preloaded_targets[i].valid = true;
        stbi_image_free(data);
    }

    CHECK_CUDA(cudaStreamSynchronize(mem_stream));

    std::vector<FrameResources> streams(NUM_STREAMS);
    for (int i = 0; i < NUM_STREAMS; ++i) streams[i].allocate(in_img_bytes, out_img_bytes, Cfg::NUM_BLOCKS, Cfg::PIXEL_DIM, loaded_count);

    dim3 preGrid(Cfg::W_SIZE, Cfg::H_SIZE); 
    int preBlock = 256;

    int M = loaded_count; 
    int N = Cfg::NUM_BLOCKS; 
    int K = Cfg::PIXEL_DIM;

    float alpha = -2.0f; float beta = 0.0f;

    dim3 stitchBlock(64, 4);
    dim3 stitchGrid(h_ceil(Cfg::FINAL_W, stitchBlock.x), h_ceil(Cfg::FINAL_H, stitchBlock.y));

    dim3 reduceGrid(Cfg::NUM_BLOCKS); 
    dim3 reduceBlock(256);
    size_t reduceSharedMem = reduceBlock.x * (sizeof(float) + sizeof(int));

    Timer timer;
    timer.start();

    if (g_rank == 0) std::cout << "[Phase 3] Pipelined Execution with TurboJPEG (CUDA)..." << std::endl;

    int processed = 0;
    static std::vector<std::string> pending_serials(NUM_STREAMS);
    static std::vector<bool> has_pending(NUM_STREAMS, false);

    for (size_t i = 0; i < preloaded_targets.size(); ++i) {
        if (!preloaded_targets[i].valid) continue;

        int sid = i % NUM_STREAMS;
        FrameResources& res = streams[sid];

        CHECK_CUDA(cudaStreamSynchronize(res.stream));

        if (has_pending[sid]) {
            int buffer_id = -1;
            {
                std::unique_lock<std::mutex> lock(g_buffer_mutex);
                g_buffer_cv.wait(lock, []{ return !g_free_buffer_ids.empty(); });
                buffer_id = g_free_buffer_ids.front();
                g_free_buffer_ids.pop();
                g_pending_writes++;
            }

            memcpy(g_pinned_pool[buffer_id], res.h_pinned_output, out_img_bytes);

            {
                std::lock_guard<std::mutex> lock(g_queue_mutex);
                g_write_queue.push({g_pinned_pool[buffer_id], buffer_id, Cfg::OUTPUT_DIR + "/" + pending_serials[sid] + ".jpg"});
                g_queue_cv.notify_one();
            }
        }

        memcpy(res.h_pinned_input, preloaded_targets[i].raw_pixels.data(), in_img_bytes);

        CHECK_CUDA(cudaMemcpyAsync(res.d_raw_input_img, res.h_pinned_input, in_img_bytes, cudaMemcpyHostToDevice, res.stream));

        preprocess_target_kernel<<<preGrid, preBlock, 0, res.stream>>>(
            res.d_raw_input_img, res.d_input_blocks, Cfg::W, Cfg::THUMB_W, Cfg::THUMB_H, Cfg::W_SIZE, Cfg::PIXEL_DIM);

        CHECK_CUBLAS(cublasSetStream(blas_handle, res.stream));

        CHECK_CUBLAS(cublasSgemm(
            blas_handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, K, &alpha, d_features, K, res.d_input_blocks, K, &beta, res.d_gemm_result, M));

        find_best_match_reduction_v4_unroll<<<reduceGrid, reduceBlock, reduceSharedMem, res.stream>>>(
            res.d_gemm_result, 
            d_tile_sq_sums,
            res.d_best_indices,
            M, 
            N
        );

        stitch_mosaic_kernel<<<stitchGrid, stitchBlock, 0, res.stream>>>(
            res.d_best_indices, d_pixels, res.d_output_img, Cfg::W_SIZE, Cfg::PLOT_W, Cfg::PLOT_H, Cfg::FINAL_W);

        CHECK_CUDA(cudaMemcpyAsync(res.h_pinned_output, res.d_output_img, out_img_bytes, cudaMemcpyDeviceToHost, res.stream));

        pending_serials[sid] = preloaded_targets[i].serial;
        has_pending[sid] = true;

        if (++processed % 100 == 0) {
            std::lock_guard<std::mutex> lock(print_mutex);
            std::cout << "[Rank " << g_rank << "] Processed " << processed << " / " << my_tasks << std::endl;
        }
    }

    for (int i = 0; i < NUM_STREAMS; ++i) {
        CHECK_CUDA(cudaStreamSynchronize(streams[i].stream));
        if (!has_pending[i]) continue;
        int buffer_id = -1;

        {
            std::unique_lock<std::mutex> lock(g_buffer_mutex);
            g_buffer_cv.wait(lock, []{ return !g_free_buffer_ids.empty(); });
            buffer_id = g_free_buffer_ids.front();
            g_free_buffer_ids.pop();
            g_pending_writes++;
        }

        memcpy(g_pinned_pool[buffer_id], streams[i].h_pinned_output, out_img_bytes);

        {
            std::lock_guard<std::mutex> lock(g_queue_mutex);
            g_write_queue.push({g_pinned_pool[buffer_id], buffer_id, Cfg::OUTPUT_DIR + "/" + pending_serials[i] + ".jpg"});
            g_queue_cv.notify_one();
        }
    }

    {
        std::lock_guard<std::mutex> lock(g_queue_mutex);
        g_finished = true;
        g_queue_cv.notify_all();
    }
    for (auto& t : thread_pool) t.join();

    free_pinned_pool();
    for (int i = 0; i < NUM_STREAMS; ++i) streams[i].free();
    CHECK_CUDA(cudaFree(d_features)); CHECK_CUDA(cudaFree(d_tile_sq_sums)); CHECK_CUDA(cudaFree(d_pixels));
    cublasDestroy(blas_handle); cudaStreamDestroy(mem_stream);

    std::cout << "[Rank " << g_rank << "] All Done." << std::endl;

    timer.stop();
    std::cout << "[Rank " << g_rank << "] Phase 3 Time: " << timer.milliseconds() << " ms" << std::endl;

    return 0;
}