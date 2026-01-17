#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <random>
#include <iomanip>
#include <cmath>

#define CUDA_CHECK(call) do {                                   \
  cudaError_t err = (call);                                     \
  if (err != cudaSuccess) {                                     \
    std::cerr << "CUDA error: " << cudaGetErrorString(err)      \
              << " at " << __FILE__ << ":" << __LINE__ << "\n"; \
    std::exit(1);                                               \
  }                                                             \
} while(0)

__global__ void mul_global(const float* __restrict__ in,
                           float* __restrict__ out,
                           float k, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = in[i] * k;
}

// Демонстрация shared: для такой простой операции обычно не ускоряет,
// но корректно показывает загрузку/синхронизацию/выгрузку.
__global__ void mul_shared(const float* __restrict__ in,
                           float* __restrict__ out,
                           float k, int n) {
  extern __shared__ float sh[];
  int tid = threadIdx.x;
  int i = blockIdx.x * blockDim.x + tid;

  if (i < n) sh[tid] = in[i];
  __syncthreads();

  if (i < n) out[i] = sh[tid] * k;
}

static float time_ms_mul(bool use_shared,
                         const float* d_in, float* d_out,
                         float k, int n,
                         int block) {
  int grid = (n + block - 1) / block;

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  // warmup
  if (!use_shared) {
    mul_global<<<grid, block>>>(d_in, d_out, k, n);
  } else {
    size_t shmem = block * sizeof(float);
    mul_shared<<<grid, block, shmem>>>(d_in, d_out, k, n);
  }
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaEventRecord(start));
  if (!use_shared) {
    mul_global<<<grid, block>>>(d_in, d_out, k, n);
  } else {
    size_t shmem = block * sizeof(float);
    mul_shared<<<grid, block, shmem>>>(d_in, d_out, k, n);
  }
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return ms;
}

int main() {
  const int N = 1'000'000;
  const float K = 2.5f;
  const int block = 256;

  std::vector<float> h_in(N), h_out(N);

  std::mt19937 rng(123);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (int i = 0; i < N; ++i) h_in[i] = dist(rng);

  float *d_in=nullptr, *d_out=nullptr;
  CUDA_CHECK(cudaMalloc(&d_in, N*sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_out, N*sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), N*sizeof(float), cudaMemcpyHostToDevice));

  float ms_global = time_ms_mul(false, d_in, d_out, K, N, block);
  float ms_shared = time_ms_mul(true,  d_in, d_out, K, N, block);

  CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, N*sizeof(float), cudaMemcpyDeviceToHost));

  bool ok = true;
  for (int i = 0; i < 20; ++i) {
    float ref = h_in[i] * K;
    if (std::fabs(h_out[i] - ref) > 1e-5f) ok = false;
  }

  std::cout << std::fixed << std::setprecision(3);
  std::cout << "[Task1] N=" << N << ", block=" << block << "\n";
  std::cout << "Global kernel: " << ms_global << " ms\n";
  std::cout << "Shared kernel: " << ms_shared << " ms\n";
  std::cout << "Check: " << (ok ? "OK" : "FAIL") << "\n";

  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_out));
  return ok ? 0 : 2;
}
