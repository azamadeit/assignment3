#include <cuda_runtime.h>                                   // Подключаем CUDA Runtime API (cudaMalloc, cudaMemcpy, cudaEvent и т.д.)
#include <iostream>                                         // Потоки ввода/вывода C++ (cout, cerr)
#include <vector>                                           // Контейнер std::vector для хранения массивов на CPU
#include <random>                                           // Генератор случайных чисел (mt19937, distribution)
#include <iomanip>                                          // Управление форматированием вывода (setprecision, fixed)
#include <cmath>                                            // Математика (fabs)

#define CUDA_CHECK(call) do {                               /* Макрос для проверки ошибок CUDA-вызовов */ \
  cudaError_t err = (call);                                 /* Выполняем вызов и сохраняем код ошибки */ \
  if (err != cudaSuccess) {                                 /* Если код ошибки не "успех" */ \
    std::cerr << "CUDA error: " << cudaGetErrorString(err)  /* Печатаем человекочитаемое сообщение об ошибке */ \
              << " at " << __FILE__ << ":" << __LINE__ << "\n"; /* Добавляем файл и строку, где случилось */ \
    std::exit(1);                                           /* Прерываем программу с кодом 1 */ \
  }                                                         /* Конец if */ \
} while(0)                                                  /* Обёртка do-while(0) для корректного поведения в любом контексте */

__global__ void mul_global(const float* __restrict__ in,     // CUDA-ядро: умножение массива на скаляр; in — входной массив в глобальной памяти
                           float* __restrict__ out,          // out — выходной массив в глобальной памяти
                           float k, int n) {                 // k — множитель, n — число элементов
  int i = blockIdx.x * blockDim.x + threadIdx.x;             // Глобальный индекс элемента: номер блока * размер блока + поток внутри блока
  if (i < n) out[i] = in[i] * k;                             // Проверяем границу и выполняем умножение
}                                                           // Конец ядра mul_global

// Демонстрация shared: для такой простой операции обычно не ускоряет, // Комментарий: shared memory тут как демонстрация, а не оптимизация
// но корректно показывает загрузку/синхронизацию/выгрузку.             // Комментарий: показаны этапы load -> sync -> store
__global__ void mul_shared(const float* __restrict__ in,     // CUDA-ядро с использованием shared memory: вход в глобальной памяти
                           float* __restrict__ out,          // Выход в глобальной памяти
                           float k, int n) {                 // k — множитель, n — число элементов
  extern __shared__ float sh[];                              // Динамически выделяемый shared-массив (размер задаётся при запуске ядра)
  int tid = threadIdx.x;                                     // Локальный индекс потока внутри блока
  int i = blockIdx.x * blockDim.x + tid;                     // Глобальный индекс элемента для данного потока

  if (i < n) sh[tid] = in[i];                                // Загружаем элемент из глобальной памяти в shared (если в диапазоне)
  __syncthreads();                                           // Барьер: ждём, пока все потоки в блоке закончат запись в shared

  if (i < n) out[i] = sh[tid] * k;                           // Читаем из shared и пишем результат в глобальную память
}                                                           // Конец ядра mul_shared

static float time_ms_mul(bool use_shared,                    // Функция измерения времени выполнения ядра: true — shared-версия, false — global
                         const float* d_in, float* d_out,    // Указатели на device-память: вход и выход
                         float k, int n,                     // k — множитель, n — размер данных
                         int block) {                        // block — число потоков в блоке (blockDim.x)
  int grid = (n + block - 1) / block;                        // Число блоков: округление вверх для покрытия всех n элементов

  cudaEvent_t start, stop;                                   // CUDA-события для тайминга на GPU
  CUDA_CHECK(cudaEventCreate(&start));                       // Создаём событие start
  CUDA_CHECK(cudaEventCreate(&stop));                        // Создаём событие stop

  // warmup                                                   // Разогрев: первый запуск часто включает JIT/прогрев кэшей и даёт "грязное" время
  if (!use_shared) {                                         // Если выбрана версия без shared
    mul_global<<<grid, block>>>(d_in, d_out, k, n);           // Запуск ядра mul_global с сеткой grid и блоком block
  } else {                                                   // Иначе shared-версия
    size_t shmem = block * sizeof(float);                    // Размер dynamic shared memory: по float на поток в блоке
    mul_shared<<<grid, block, shmem>>>(d_in, d_out, k, n);    // Запуск mul_shared с указанным объёмом shared-памяти
  }
  CUDA_CHECK(cudaGetLastError());                             // Проверяем ошибку запуска ядра (асинхронная ошибка конфигурации/запуска)
  CUDA_CHECK(cudaDeviceSynchronize());                        // Ждём завершения warmup-запуска, чтобы не смешивать с таймингом

  CUDA_CHECK(cudaEventRecord(start));                         // Записываем событие start в очередь выполнения (начало измерения)
  if (!use_shared) {                                         // Выбор версии для измеряемого запуска
    mul_global<<<grid, block>>>(d_in, d_out, k, n);           // Запускаем глобальную версию
  } else {                                                   // Или shared-версию
    size_t shmem = block * sizeof(float);                    // Снова считаем размер shared-памяти (можно было вынести, но так нагляднее)
    mul_shared<<<grid, block, shmem>>>(d_in, d_out, k, n);    // Запуск shared-версии
  }
  CUDA_CHECK(cudaGetLastError());                             // Проверяем ошибку запуска ядра
  CUDA_CHECK(cudaEventRecord(stop));                          // Ставим событие stop после запуска ядра (останавливаем измерение)
  CUDA_CHECK(cudaEventSynchronize(stop));                     // Ждём, пока событие stop будет достигнуто (ядро завершится)

  float ms = 0.0f;                                            // Переменная для времени в миллисекундах
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));         // Считаем время между start и stop на GPU
  CUDA_CHECK(cudaEventDestroy(start));                        // Освобождаем ресурс события start
  CUDA_CHECK(cudaEventDestroy(stop));                         // Освобождаем ресурс события stop
  return ms;                                                  // Возвращаем измеренное время
}                                                            // Конец time_ms_mul

int main() {                                                  // Точка входа программы
  const int N = 1'000'000;                                    // Размер массива: 1,000,000 элементов (C++14-литерал с разделителем)
  const float K = 2.5f;                                       // Множитель для умножения
  const int block = 256;                                      // Размер блока: 256 потоков (типичное значение)

  std::vector<float> h_in(N), h_out(N);                       // Хост-массивы: вход и выход (в оперативной памяти)

  std::mt19937 rng(123);                                      // Генератор Mersenne Twister с фиксированным seed (воспроизводимость)
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);    // Равномерное распределение по [-1, 1]
  for (int i = 0; i < N; ++i) h_in[i] = dist(rng);            // Заполняем входной массив случайными числами

  float *d_in=nullptr, *d_out=nullptr;                        // Device-указатели (на GPU), инициализируем nullptr
  CUDA_CHECK(cudaMalloc(&d_in, N*sizeof(float)));             // Выделяем на GPU память под входной массив
  CUDA_CHECK(cudaMalloc(&d_out, N*sizeof(float)));            // Выделяем на GPU память под выходной массив
  CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), N*sizeof(float), cudaMemcpyHostToDevice)); // Копируем входные данные CPU->GPU

  float ms_global = time_ms_mul(false, d_in, d_out, K, N, block); // Измеряем время глобальной версии (use_shared=false)
  float ms_shared = time_ms_mul(true,  d_in, d_out, K, N, block); // Измеряем время shared версии (use_shared=true)

  CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, N*sizeof(float), cudaMemcpyDeviceToHost)); // Копируем результат GPU->CPU

  bool ok = true;                                             // Флаг корректности результатов
  for (int i = 0; i < 20; ++i) {                               // Проверяем первые 20 элементов (быстрая sanity-check)
    float ref = h_in[i] * K;                                   // Эталон: умножение на CPU
    if (std::fabs(h_out[i] - ref) > 1e-5f) ok = false;         // Если ошибка больше допуска, считаем провал
  }

  std::cout << std::fixed << std::setprecision(3);             // Настраиваем формат вывода: фиксированная точка и 3 знака после запятой
  std::cout << "[Task1] N=" << N << ", block=" << block << "\n"; // Печатаем параметры эксперимента
  std::cout << "Global kernel: " << ms_global << " ms\n";      // Печатаем время global-версии
  std::cout << "Shared kernel: " << ms_shared << " ms\n";      // Печатаем время shared-версии
  std::cout << "Check: " << (ok ? "OK" : "FAIL") << "\n";      // Печатаем результат проверки корректности

  CUDA_CHECK(cudaFree(d_in));                                  // Освобождаем память GPU для входного массива
  CUDA_CHECK(cudaFree(d_out));                                 // Освобождаем память GPU для выходного массива
  return ok ? 0 : 2;                                           // Возвращаем 0 при успехе, иначе 2 (условный код ошибки)
}                                                            // Конец main
