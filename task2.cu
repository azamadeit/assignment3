#include <cuda_runtime.h>        // CUDA runtime
#include <iostream>              // ввод/вывод
#include <vector>                // std::vector
#include <random>                // генератор случайных чисел
#include <iomanip>               // форматирование вывода
#include <cmath>                 // fabs

#define CUDA_CHECK(call) do {                                   \
  cudaError_t err = (call);                                     /* вызов CUDA */ \
  if (err != cudaSuccess) {                                     /* проверка ошибки */ \
    std::cerr << "CUDA error: " << cudaGetErrorString(err)      /* вывод ошибки */ \
              << " at " << __FILE__ << ":" << __LINE__ << "\n"; \
    std::exit(1);                                               /* выход */ \
  }                                                             \
} while(0)

__global__ void add_kernel(const float* __restrict__ a,         // входной массив a
                           const float* __restrict__ b,         // входной массив b
                           float* __restrict__ c,               // выходной массив c
                           int n) {                              // размер массива
  int i = blockIdx.x * blockDim.x + threadIdx.x;                 // глобальный индекс
  if (i < n) c[i] = a[i] + b[i];                                 // сложение элементов
}

static float time_ms_add_repeated(const float* d_a,              // массив a на GPU
                                  const float* d_b,              // массив b на GPU
                                  float* d_c,                    // массив c на GPU
                                  int n,                          // размер
                                  int block,                      // размер блока
                                  int repeats) {                  // число повторов
  int grid = (n + block - 1) / block;                            // число блоков

  cudaEvent_t start, stop;                                       // события CUDA
  CUDA_CHECK(cudaEventCreate(&start));                           // создать start
  CUDA_CHECK(cudaEventCreate(&stop));                            // создать stop

  add_kernel<<<grid, block>>>(d_a, d_b, d_c, n);                 // прогрев
  CUDA_CHECK(cudaGetLastError());                                 // проверка
  CUDA_CHECK(cudaDeviceSynchronize());                            // синхронизация

  CUDA_CHECK(cudaEventRecord(start));                             // старт таймера
  for (int r = 0; r < repeats; ++r) {                             // цикл повторов
    add_kernel<<<grid, block>>>(d_a, d_b, d_c, n);                // запуск ядра
  }
  CUDA_CHECK(cudaGetLastError());                                 // проверка
  CUDA_CHECK(cudaEventRecord(stop));                              // стоп таймера
  CUDA_CHECK(cudaEventSynchronize(stop));                         // ожидание

  float ms = 0.0f;                                                // время
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));             // вычисление времени

  CUDA_CHECK(cudaEventDestroy(start));                            // удалить start
  CUDA_CHECK(cudaEventDestroy(stop));                             // удалить stop
  return ms;                                                      // вернуть время
}

int main() {
  const int N = 1'000'000;                                        // размер массивов
  const int REPEATS = 200000;                                     // число повторов

  const int blocks_to_test[] = {128, 256, 512};                   // размеры блоков

  std::vector<float> h_a(N), h_b(N), h_c(N);                     // массивы CPU

  std::mt19937 rng(42);                                          // ГСЧ
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);      // распределение
  for (int i = 0; i < N; ++i) {                                  // заполнение
    h_a[i] = dist(rng);                                          // a[i]
    h_b[i] = dist(rng);                                          // b[i]
  }

  float *d_a=nullptr, *d_b=nullptr, *d_c=nullptr;                // указатели GPU
  CUDA_CHECK(cudaMalloc(&d_a, N*sizeof(float)));                 // malloc a
  CUDA_CHECK(cudaMalloc(&d_b, N*sizeof(float)));                 // malloc b
  CUDA_CHECK(cudaMalloc(&d_c, N*sizeof(float)));                 // malloc c

  CUDA_CHECK(cudaMemcpy(d_a, h_a.data(),                         // копирование a
                        N*sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b, h_b.data(),                         // копирование b
                        N*sizeof(float),
                        cudaMemcpyHostToDevice));

  std::cout << std::fixed << std::setprecision(4);               // формат вывода
  std::cout << "[Task2] N=" << N                                 // вывод заголовка
            << ", REPEATS=" << REPEATS << "\n";

  for (int block : blocks_to_test) {                              // цикл по блокам
    float total_ms = time_ms_add_repeated(d_a, d_b, d_c,         // замер времени
                                          N, block, REPEATS);
    std::cout << "Размер блока: " << block                       // вывод блока
              << ", время выполнения: " << total_ms << " мс\n"; // вывод времени
  }

  int block = 256;                                               // блок для проверки
  int grid = (N + block - 1) / block;                            // grid
  add_kernel<<<grid, block>>>(d_a, d_b, d_c, N);                 // запуск ядра
  CUDA_CHECK(cudaGetLastError());                                 // проверка
  CUDA_CHECK(cudaDeviceSynchronize());                            // ожидание
  CUDA_CHECK(cudaMemcpy(h_c.data(), d_c,                         // копирование
                        N*sizeof(float),
                        cudaMemcpyDeviceToHost));

  bool ok = true;                                                // флаг проверки
  for (int i = 0; i < 20; ++i) {                                 // проверка значений
    float ref = h_a[i] + h_b[i];                                 // эталон
    if (std::fabs(h_c[i] - ref) > 1e-5f) ok = false;             // сравнение
  }
  std::cout << "Check: " << (ok ? "OK" : "FAIL") << "\n";        // результат

  CUDA_CHECK(cudaFree(d_a));                                     // free a
  CUDA_CHECK(cudaFree(d_b));                                     // free b
  CUDA_CHECK(cudaFree(d_c));                                     // free c
  return ok ? 0 : 2;                                             // код возврата
}
