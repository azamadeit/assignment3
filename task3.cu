#include <cuda_runtime.h>   // CUDA runtime API
#include <iostream>         // стандартный ввод/вывод
#include <vector>           // контейнер std::vector
#include <random>           // генерация случайных чисел
#include <iomanip>          // форматирование вывода
#include <cmath>            // математические функции (fabs)

// Макрос для проверки ошибок CUDA
#define CUDA_CHECK(call) do {                                   \
  cudaError_t err = (call);                                     /* вызов CUDA-функции */ \
  if (err != cudaSuccess) {                                     /* если произошла ошибка */ \
    std::cerr << "CUDA error: "                                 \
              << cudaGetErrorString(err)                        /* текст ошибки */ \
              << " at " << __FILE__ << ":" << __LINE__ << "\n"; /* файл и строка */ \
    std::exit(1);                                               /* аварийный выход */ \
  }                                                             \
} while(0)

// CUDA-ядро с коалесцированным доступом к памяти
// Каждый поток обрабатывает элемент с тем же индексом
__global__ void coalesced_kernel(const float* __restrict__ in,  // входной массив
                                 float* __restrict__ out,       // выходной массив
                                 int n) {                        // размер массива
  int i = blockIdx.x * blockDim.x + threadIdx.x;                // глобальный индекс потока
  if (i < n)                                                    // проверка выхода за границы
    out[i] = in[i] * 1.001f + 0.123f;                           // простая арифметическая операция
}

// CUDA-ядро с некоалесцированным доступом к памяти
// Потоки обращаются к разрозненным элементам массива
__global__ void uncoalesced_kernel(const float* __restrict__ in, // входной массив
                                   float* __restrict__ out,      // выходной массив
                                   int n,                         // размер массива
                                   int stride) {                  // шаг доступа
  int i = blockIdx.x * blockDim.x + threadIdx.x;                 // глобальный индекс потока
  if (i < n) {                                                   // проверка границ
    int j = (i * stride) % n;                                    // вычисление разреженного индекса
    out[j] = in[j] * 1.001f + 0.123f;                            // арифметическая операция
  }
}

// Функция измерения времени выполнения CUDA-ядра с повторами
static float time_ms_repeated(bool uncoalesced,                 // выбор типа доступа
                              const float* d_in,                // входной массив на GPU
                              float* d_out,                     // выходной массив на GPU
                              int n,                             // размер массива
                              int block,                         // размер блока
                              int repeats,                       // число повторов
                              int stride) {                      // шаг доступа
  int grid = (n + block - 1) / block;                            // вычисление числа блоков

  cudaEvent_t start, stop;                                       // CUDA-события
  CUDA_CHECK(cudaEventCreate(&start));                           // создание start-события
  CUDA_CHECK(cudaEventCreate(&stop));                            // создание stop-события

  // Прогрев GPU (один запуск ядра)
  if (!uncoalesced)
    coalesced_kernel<<<grid, block>>>(d_in, d_out, n);
  else
    uncoalesced_kernel<<<grid, block>>>(d_in, d_out, n, stride);

  CUDA_CHECK(cudaGetLastError());                                // проверка ошибок ядра
  CUDA_CHECK(cudaDeviceSynchronize());                           // ожидание завершения

  CUDA_CHECK(cudaEventRecord(start));                            // запуск таймера

  for (int r = 0; r < repeats; ++r) {                            // цикл повторов
    if (!uncoalesced)
      coalesced_kernel<<<grid, block>>>(d_in, d_out, n);
    else
      uncoalesced_kernel<<<grid, block>>>(d_in, d_out, n, stride);
  }

  CUDA_CHECK(cudaGetLastError());                                // проверка ошибок
  CUDA_CHECK(cudaEventRecord(stop));                             // остановка таймера
  CUDA_CHECK(cudaEventSynchronize(stop));                        // ожидание окончания

  float ms = 0.0f;                                               // переменная для времени
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));            // вычисление времени в мс

  CUDA_CHECK(cudaEventDestroy(start));                           // удаление события start
  CUDA_CHECK(cudaEventDestroy(stop));                            // удаление события stop
  return ms;                                                     // возврат времени
}

int main() {
  const int N = 1'000'000;                                       // размер массива
  const int block = 256;                                        // размер блока

  // Шаг доступа (не кратен 32 для ухудшения коалесцинга)
  const int stride = 97;

  // Количество повторов для увеличения времени измерения
  const int REPEATS = 50000;

  std::vector<float> h_in(N), h_out(N);                          // массивы на CPU

  std::mt19937 rng(7);                                          // генератор случайных чисел
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);     // равномерное распределение

  for (int i = 0; i < N; ++i)                                   // заполнение массива
    h_in[i] = dist(rng);

  float *d_in = nullptr, *d_out = nullptr;                      // указатели на GPU
  CUDA_CHECK(cudaMalloc(&d_in, N * sizeof(float)));             // выделение памяти для входа
  CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(float)));            // выделение памяти для выхода

  CUDA_CHECK(cudaMemcpy(d_in, h_in.data(),                      // копирование входных данных
                        N * sizeof(float),
                        cudaMemcpyHostToDevice));

  float ms_coal = time_ms_repeated(false, d_in, d_out,          // замер коалесцированного доступа
                                   N, block, REPEATS, stride);

  float ms_uncoal = time_ms_repeated(true, d_in, d_out,         // замер некоалесцированного доступа
                                     N, block, REPEATS, stride);

  std::cout << std::fixed << std::setprecision(4);              // формат вывода
  std::cout << "[Task3] N=" << N                                // вывод параметров
            << ", block=" << block
            << ", REPEATS=" << REPEATS
            << ", stride=" << stride << "\n";

  std::cout << "Коалесцированный доступ:   " << ms_coal   << " мс\n";   // вывод времени
  std::cout << "Некоалесцированный доступ: " << ms_uncoal << " мс\n";   // вывод времени

  CUDA_CHECK(cudaMemcpy(h_out.data(), d_out,                    // копирование результата
                        N * sizeof(float),
                        cudaMemcpyDeviceToHost));

  std::cout << "Пример out[0]: " << h_out[0] << "\n";           // пример результата

  CUDA_CHECK(cudaFree(d_in));                                   // освобождение памяти
  CUDA_CHECK(cudaFree(d_out));                                  // освобождение памяти
  return 0;                                                     // завершение программы
}
