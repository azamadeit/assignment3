#include <cuda_runtime.h>                          // подключаем CUDA Runtime API
#include <iostream>                                // подключаем стандартный вывод
#include <vector>                                  // подключаем std::vector
#include <random>                                  // подключаем генератор случайных чисел
#include <iomanip>                                 // подключаем форматирование вывода
#include <limits>                                  // подключаем numeric_limits для infinity
#include <cmath>                                   // подключаем fabs для проверки

#define CUDA_CHECK(call) do {                      /* макрос для проверки вызовов CUDA */ \
  cudaError_t err = (call);                        /* выполняем CUDA-вызов и получаем код ошибки */ \
  if (err != cudaSuccess) {                        /* если код ошибки не успех */ \
    std::cerr << "CUDA error: "                    /* печатаем префикс */ \
              << cudaGetErrorString(err)           /* печатаем текст ошибки */ \
              << " at " << __FILE__               /* печатаем имя файла */ \
              << ":" << __LINE__ << "\n";         /* печатаем номер строки */ \
    std::exit(1);                                  /* завершаем программу */ \
  }                                                /* конец проверки */ \
} while(0)                                         /* оборачиваем в do-while для корректности */

// CUDA-ядро: поэлементное сложение массивов
__global__ void add_kernel(const float* __restrict__ a,  // входной массив a в глобальной памяти
                           const float* __restrict__ b,  // входной массив b в глобальной памяти
                           float* __restrict__ c,        // выходной массив c в глобальной памяти
                           int n) {                      // размер массивов
  int i = blockIdx.x * blockDim.x + threadIdx.x;         // вычисляем глобальный индекс элемента
  if (i < n)                                             // проверяем, что не вышли за границы
    c[i] = a[i] + b[i];                                  // выполняем сложение
}

// Функция замера времени: прогрев + repeats запусков + измерение CUDA events
static float time_ms_add_repeated(const float* d_a,       // указатель на a на GPU
                                  const float* d_b,       // указатель на b на GPU
                                  float* d_c,             // указатель на c на GPU
                                  int n,                  // размер массивов
                                  int block,              // размер блока (threads per block)
                                  int repeats) {           // число повторов ядра для стабильного измерения
  int grid = (n + block - 1) / block;                     // вычисляем число блоков (grid size)

  cudaEvent_t start, stop;                                // создаём CUDA события для измерения времени
  CUDA_CHECK(cudaEventCreate(&start));                    // создаём событие start
  CUDA_CHECK(cudaEventCreate(&stop));                     // создаём событие stop

  add_kernel<<<grid, block>>>(d_a, d_b, d_c, n);          // прогревочный запуск (warm-up)
  CUDA_CHECK(cudaGetLastError());                         // проверяем ошибки запуска ядра
  CUDA_CHECK(cudaDeviceSynchronize());                    // ждём завершения прогрева

  CUDA_CHECK(cudaEventRecord(start));                     // записываем старт таймера
  for (int r = 0; r < repeats; ++r) {                     // повторяем запуск ядра repeats раз
    add_kernel<<<grid, block>>>(d_a, d_b, d_c, n);        // запуск ядра
  }                                                       // конец цикла повторов
  CUDA_CHECK(cudaGetLastError());                         // проверяем ошибки последнего запуска
  CUDA_CHECK(cudaEventRecord(stop));                      // записываем стоп таймера
  CUDA_CHECK(cudaEventSynchronize(stop));                 // ждём завершения (и ядра, и события stop)

  float ms = 0.0f;                                        // переменная для времени в миллисекундах
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));     // считаем прошедшее время между start и stop

  CUDA_CHECK(cudaEventDestroy(start));                    // удаляем событие start
  CUDA_CHECK(cudaEventDestroy(stop));                     // удаляем событие stop
  return ms;                                              // возвращаем измеренное время
}

int main() {                                              // точка входа программы
  const int N = 1'000'000;                                // размер массивов (1 000 000 элементов)
  const int REPEATS = 5000;                               // повторы для измерения (можно 2000..20000)

  std::vector<float> h_a(N), h_b(N), h_c(N);              // создаём массивы на CPU: a, b, c

  std::mt19937 rng(1234);                                 // инициализируем генератор случайных чисел
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);// задаём равномерное распределение

  for (int i = 0; i < N; ++i) {                           // заполняем входные массивы
    h_a[i] = dist(rng);                                   // a[i] случайным числом
    h_b[i] = dist(rng);                                   // b[i] случайным числом
  }                                                       // конец заполнения

  float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;   // объявляем указатели на массивы в GPU памяти

  CUDA_CHECK(cudaMalloc(&d_a, N * sizeof(float)));        // выделяем память на GPU под a
  CUDA_CHECK(cudaMalloc(&d_b, N * sizeof(float)));        // выделяем память на GPU под b
  CUDA_CHECK(cudaMalloc(&d_c, N * sizeof(float)));        // выделяем память на GPU под c

  CUDA_CHECK(cudaMemcpy(d_a, h_a.data(),                  // копируем a с CPU на GPU
                        N * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b, h_b.data(),                  // копируем b с CPU на GPU
                        N * sizeof(float),
                        cudaMemcpyHostToDevice));

  const int bad_block = 64;                               // задаём заведомо "плохой" размер блока
  float bad_ms = time_ms_add_repeated(d_a, d_b, d_c,      // замер времени для плохой конфигурации
                                      N, bad_block, REPEATS);

  const int candidates[] = {64, 128, 192, 256, 320, 384, 512, 768, 1024}; // кандидаты размеров блока

  float best_ms = std::numeric_limits<float>::infinity(); // лучшее (минимальное) время (начинаем с бесконечности)
  int best_block = -1;                                    // лучший размер блока (пока неизвестен)

  for (int block : candidates) {                          // перебираем все кандидаты
    float ms = time_ms_add_repeated(d_a, d_b, d_c,        // замеряем время для текущего block
                                    N, block, REPEATS);
    if (ms < best_ms) {                                   // если текущее время меньше лучшего
      best_ms = ms;                                       // обновляем лучшее время
      best_block = block;                                 // запоминаем лучший размер блока
    }                                                     // конец условия
  }                                                       // конец перебора кандидатов

  std::cout << std::fixed << std::setprecision(4);        // настраиваем формат вывода
  std::cout << "[Task4] N=" << N                          // выводим параметры эксперимента
            << ", REPEATS=" << REPEATS << "\n";

  std::cout << "Плохая конфигурация: block="              // печатаем результат "плохой" конфигурации
            << bad_block << ", время: " << bad_ms << " мс\n";

  std::cout << "Лучшая конфигурация: block="              // печатаем результат "лучшей" конфигурации
            << best_block << ", время: " << best_ms << " мс\n";

  double speedup = static_cast<double>(bad_ms)            // считаем ускорение как отношение времён
                 / static_cast<double>(best_ms);

  std::cout << "Ускорение (плохая/лучшая): "              // выводим ускорение
            << speedup << "x\n";

  int grid = (N + 256 - 1) / 256;                         // считаем grid для проверки корректности (block=256)
  add_kernel<<<grid, 256>>>(d_a, d_b, d_c, N);            // выполняем один запуск для получения результата
  CUDA_CHECK(cudaGetLastError());                         // проверяем ошибки запуска
  CUDA_CHECK(cudaDeviceSynchronize());                    // ждём завершения

  CUDA_CHECK(cudaMemcpy(h_c.data(), d_c,                  // копируем результат c с GPU на CPU
                        N * sizeof(float),
                        cudaMemcpyDeviceToHost));

  bool ok = true;                                         // флаг корректности
  for (int i = 0; i < 20; ++i) {                          // проверяем несколько элементов
    float ref = h_a[i] + h_b[i];                          // эталон на CPU
    if (std::fabs(h_c[i] - ref) > 1e-5f) ok = false;      // сравниваем с допуском
  }                                                       // конец проверки

  std::cout << "Check: "                                  // выводим статус проверки
            << (ok ? "OK" : "FAIL") << "\n";

  CUDA_CHECK(cudaFree(d_a));                              // освобождаем память GPU для a
  CUDA_CHECK(cudaFree(d_b));                              // освобождаем память GPU для b
  CUDA_CHECK(cudaFree(d_c));                              // освобождаем память GPU для c

  return ok ? 0 : 2;                                      // возвращаем код (0 если OK, иначе 2)
}
