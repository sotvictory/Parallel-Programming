#include <algorithm>
#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <chrono>
#include <omp.h>

template <typename T>
class Matrix {
public:
    Matrix(int rows, int cols) : rows_(rows), cols_(cols), data_(rows * cols) {}

    T& operator()(int row, int col) {
        return data_[row * cols_ + col];
    }

    const T& operator()(int row, int col) const {
        return data_[row * cols_ + col];
    }

    int rows() const { return rows_; }
    int cols() const { return cols_; }

private:
    int rows_;
    int cols_;
    std::vector<T> data_;
};

template <typename T>
Matrix<T> generateMatrix(int rows, int cols) {
    Matrix<T> matrix(rows, cols);
    std::random_device rnd_device;
    std::mt19937 gen_engine{rnd_device()};
    std::uniform_real_distribution<T> dist{0.0, 1.0};

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            matrix(i, j) = dist(gen_engine);
        }
    }

    return matrix;
}

template <typename T>
void sequentialMatrixMultiply(
    const Matrix<T>& matrixA, 
    const Matrix<T>& matrixB, 
    Matrix<T>& resultMatrix, 
    int rowsA, 
    int columnsA, 
    int columnsB
) {
    for (int j = 0; j < columnsB; ++j) {
        for (int i = 0; i < rowsA; ++i) {
            T sum = 0;
            for (int k = 0; k < columnsA; ++k) {
                sum += matrixA(i, k) * matrixB(k, j);
            }
            resultMatrix(i, j) = sum;
        }
    }
}

template <typename T>
void parallelMatrixMultiply(
    const Matrix<T>& matrixA, 
    const Matrix<T>& matrixB, 
    Matrix<T>& resultMatrix, 
    int rowsA, 
    int columnsA, 
    int columnsB, 
    int threadCount
) {
    omp_set_num_threads(threadCount);

    #pragma omp parallel for collapse(2)
    for (int j = 0; j < columnsB; ++j) {
        for (int i = 0; i < rowsA; ++i) {
            T sum = 0;
            for (int k = 0; k < columnsA; ++k) {
                sum += matrixA(i, k) * matrixB(k, j);
            }
            resultMatrix(i, j) = sum;
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc != 5) {
        std::cerr << "Usage: " << argv[0] << " M N K P" << std::endl;
        return 1;
    }

    int m = std::atoi(argv[1]);
    int n = std::atoi(argv[2]);
    int k = std::atoi(argv[3]);
    int p = std::atoi(argv[4]);

    Matrix<double> matrixA = generateMatrix<double>(m, n);
    Matrix<double> matrixB = generateMatrix<double>(n, k);
    Matrix<double> resultMatrix(m, k);

    auto startTimeSerial = std::chrono::high_resolution_clock::now();
    sequentialMatrixMultiply<double>(matrixA, matrixB, resultMatrix, m, n, k);
    auto endTimeSerial = std::chrono::high_resolution_clock::now();
    
    auto startTimeParallel = std::chrono::high_resolution_clock::now();
    parallelMatrixMultiply<double>(matrixA, matrixB, resultMatrix, m, n, k, p);
    auto endTimeParallel = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> duration_serial = endTimeSerial - startTimeSerial;
    std::chrono::duration<double> duration_parallel = endTimeParallel - startTimeParallel;

    double speedup = duration_serial.count() / duration_parallel.count();
    long long W = static_cast<long long>(m) * n * k;
    double efficiency = speedup / p;
    double cost_C = p * duration_parallel.count();
    double total_overhead_T0 = p * duration_parallel.count() * duration_serial.count();

    std::ofstream logFile("log.txt", std::ios_base::app);
    if (logFile.is_open()) {
        logFile << "M: " << m << ", N: " << n << ", K: " << k << ", P: " << p << "\n";
        logFile << "Duration Serial: " << duration_serial.count() << " seconds\n";
        logFile << "Duration Parallel: " << duration_parallel.count() << " seconds\n";
        logFile << "Speedup S: " << speedup << "\n";
        logFile << "Computational complexity W: " << W << " operations\n";
        logFile << "Efficiency E: " << efficiency << "\n";
        logFile << "Cost of computations C: " << cost_C << "\n";
        logFile << "Total overhead T0: " << total_overhead_T0 << "\n\n";
        logFile.close();
    } else {
        std::cerr << "Unable to open log file." << std::endl;
    }

    //printMatrix(matrixA);
    //printMatrix(matrixB);
    //printMatrix(resultMatrix);

    return 0;
}