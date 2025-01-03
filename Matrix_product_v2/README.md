## TODO: Реализация блочного умножения матриц A и B

1. **Входные данные**:
   - Матрица A размера M x K.
   - Матрица B размера K x N.
   - Размеры shared блоков:
     - Shared блок A: Mtile x Ktile
     - Shared блок B: Ktile x Ntile
     - Temporary блок C: Mtile x Ntile

2. **Выходные данные**:
   - Результирующая матрица C размера M x N.

### Шаги реализации

#### 1. Блочная организация матриц
- Разделить матрицы A, B, C на блоки:
  - Для матрицы A: 
    - A00, A01, A10, A11, ...
  - Для матрицы B:
    - B00, B01, B10, B11, ...
  - Для матрицы C:
    - C00, C01, C10, C11, ...

#### 2. Параллельная обработка
- Создать P = (M / Mtile) * (N / Ntile) процессов
- Каждый процесс будет считать соответствующий блок C
- Каждый процесс получает две полосы: 
  - полоса матрицы A = обрабатываемые Mtile строк матрицы A
  - полоса матрицы B = обрабатываемые Ntile столбцов матрицы B
- Создать T = K / Ktile потоков в каждом процессе
- Каждый поток будет умножать блоки, вошедшие в полосы процесса

##### Пример распределения задач:
Пусть Mtile = Ntile = 2, Ktile = 3

Есть матрица A размера 6 x 6
A00 | A01
A10 | A11
A20 | A21

И есть матрица B размера 6 x 4:
B00 | B01
B10 | B11

Тогда результирующая матрица C размера 6 x 4:
C00 | C01
C10 | C11
C20 | C21

- Процесс 0:
  - Инициализация: загрузка матриц A и B, выделение памяти для C
  - Разделение работы: определение блоков для процессов 1-6
  - Запуск процессов: инициирование параллельных вычислений
  - Сбор результатов: агрегирование временных блоков C в результирующую матрицу
  - Очистка ресурсов: освобождение памяти и завершение работы

- Процесс 1:
  - Входные данные: A00, A01, B00, B10
  - Вычисление: C00 = A00 * B00 + A01 * B10

- Процесс 2:
  - Входные данные: A10, A11, B00, B10
  - Вычисление: C10 = A10 * B00 + A11 * B10

- Процесс 3:
  - Входные данные: A20, A21, B00, B10
  - Вычисление: C20 = A20 * B00 + A21 * B10

- Процесс 4:
  - Входные данные: A00, A01, B01, B11
  - Вычисление: C01 = A00 * B01 + A01 * B11

- Процесс 5:
  - Входные данные: A10, A11, B01, B11
  - Вычисление: C11 = A10 * B01 + A11 * B11

- Процесс 6:
  - Входные данные: A20, A21, B01, B11
  - Вычисление: C21 = A20 * B01 + A21 * B11

### Примечания
- Определить оптимальную структуру для хранения матриц A, B и C
- Убедиться в корректности работы алгоритма для различных размеров матриц
- Провести тестирование
  - оценить время выполнения алгоритма в зависимости от размера входных матриц
  - исследовать влияние размеров блоков на производительность
  - оценить масштабируемость алгоритма при увеличении количества процессов и их влияние на время выполнения
  - проанализировать выигрыш в производительности
  - Test Inspector + System Profiler (Visual Studio)

### Ссылки
- [Matrix Multiplication Performance Guide NVIDIA](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html#math-mem)
- [Matrix Multiplication Background User's Guide NVIDIA](https://docs.nvidia.com/deeplearning/performance/pdf/Matrix-Multiplication-Background-User-Guide.pdf)
- [Tiled Matrix Multiplication Notes](https://harmanani.github.io/classes/csc447/Notes/Lecture23-tiled-matrix-multiplication.pdf)
- [How to Tile Matrix Multiplication](https://alvinwan.com/how-to-tile-matrix-multiplication/)
- [Blog on Tiled Matrix Multiplication](https://penny-xu.github.io/blog/tiled-matrix-multiplication)
