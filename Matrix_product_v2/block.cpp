#include <iostream>
#include <algorithm>
#include <iomanip>
#include <random>
#include <chrono>
#include <vector>
#include <omp.h>
#include <mpi.h>

// mpic++ -fopenmp -o prog matmul.cpp
// mpirun -np <num_processes> ./matmul <method>

void printMatrix(const std::vector<int>& matrix, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << std::setw(5) << matrix[i * cols + j];
        }
        std::cout << "\n";
    }
}

void generateRandom(std::vector<int>& A, std::vector<int>& B, std::vector<int>& C, int N, int M, int K) {
    A.resize(N * M);
    B.resize(M * K);
    C.resize(N * K);

    unsigned seed = 42;
    std::mt19937 gen(seed);
    std::uniform_int_distribution<> dis(-10, 10);

    for (int i = 0; i < N * M; ++i) {
        A[i] = dis(gen);
    }
    
    for (int i = 0; i < M * K; ++i) {
        B[i] = dis(gen);
    }

    std::fill(C.begin(), C.end(), 0);
}

void blockMultiply(const std::vector<int>& A, const std::vector<int>& B, std::vector<int>& C,
                         int M, int K, int N, int Mtile, int Ktile, int Ntile) {

    for (int i = 0; i < M; i += Mtile) {
        for (int j = 0; j < N; j += Ntile) {
            for (int k = 0; k < K; k += Ktile) {
                std::vector<int> temp(Mtile * Ntile, 0);
                
                for (int ii = i; ii < std::min(i + Mtile, M); ++ii) {
                    for (int jj = k; jj < std::min(k + Ktile, K); ++jj) {
                        for (int jj2 = j; jj2 < std::min(j + Ntile, N); ++jj2) {
                            temp[(ii - i) * Ntile + (jj2 - j)] += A[ii * K + jj] * B[jj * N + jj2];
                        }
                    }
                }

                for (int ii = i; ii < std::min(i + Mtile, M); ++ii) {
                    for (int jj2 = j; jj2 < std::min(j + Ntile, N); ++jj2) {
                        C[ii * N + jj2] += temp[(ii - i) * Ntile + (jj2 - j)];
                    }
                }
            }
        }
    }
}

void OMPmethod(const std::vector<int>& A, const std::vector<int>& B, std::vector<int>& C,
               int M, int K, int N, int Mtile, int Ktile, int Ntile) {
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            C[i * N + j] = 0;
        }
    }

    #pragma omp parallel for collapse(3)
    for (int i = 0; i < M; i += Mtile) {
        for (int j = 0; j < N; j += Ntile) {
            for (int k = 0; k < K; k += Ktile) {
                std::vector<int> blockA(Mtile * Ktile, 0);
                std::vector<int> blockB(Ktile * Ntile, 0);
                
                for (int ii = 0; ii < Mtile; ++ii) {
                    for (int jj = 0; jj < Ktile; ++jj) {
                        if (i + ii < M && k + jj < K) {
                            blockA[ii * Ktile + jj] = A[(i + ii) * K + (k + jj)];
                        }
                    }
                }

                for (int jj = 0; jj < Ktile; ++jj) {
                    for (int jj2 = 0; jj2 < Ntile; ++jj2) {
                        if (k + jj < K && j + jj2 < N) {
                            blockB[jj * Ntile + jj2] = B[(k + jj) * N + (j + jj2)];
                        }
                    }
                }

                std::vector<int> temp(Mtile * Ntile, 0);

                for (int ii = 0; ii < Mtile; ++ii) {
                    for (int jj = 0; jj < Ktile; ++jj) {
                        for (int jj2 = 0; jj2 < Ntile; ++jj2) {
                            if ((i + ii) < M && (j + jj2) < N) {
                                temp[ii * Ntile + jj2] += blockA[ii * Ktile + jj] * blockB[jj * Ntile + jj2];
                            }
                        }
                    }
                }

                for (int ii = 0; ii < Mtile; ++ii) {
                    for (int jj2 = 0; jj2 < Ntile; ++jj2) {
                        if ((i + ii) < M && (j + jj2) < N) {
                            #pragma omp atomic
                            C[(i + ii) * N + (j + jj2)] += temp[ii * Ntile + jj2];
                        }
                    }
                }
            }
        }
    }
}

void MPImethod(const std::vector<int>& A, const std::vector<int>& B, std::vector<int>& C,
               int M, int K, int N, int Mtile, int Ktile, int Ntile) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int rows_per_process = M / size;
    int start_row = rank * rows_per_process;
    int end_row = (rank + 1) * rows_per_process;

    std::vector<int> local_C(rows_per_process * N, 0);

    for (int i = start_row; i < end_row; i += Mtile) {
        for (int j = 0; j < N; j += Ntile) {
            for (int k = 0; k < K; k += Ktile) {
                std::vector<int> temp(Mtile * Ntile, 0);
                
                for (int ii = i; ii < std::min(i + Mtile, end_row); ++ii) {
                    for (int jj = k; jj < std::min(k + Ktile, K); ++jj) {
                        for (int jj2 = j; jj2 < std::min(j + Ntile, N); ++jj2) {
                            temp[(ii - i) * Ntile + (jj2 - j)] += A[ii * K + jj] * B[jj * N + jj2];
                        }
                    }
                }

                for (int ii = i; ii < std::min(i + Mtile, end_row); ++ii) {
                    for (int jj2 = j; jj2 < std::min(j + Ntile, N); ++jj2) {
                        local_C[(ii - start_row) * N + jj2] += temp[(ii - i) * Ntile + (jj2 - j)];
                    }
                }
            }
        }
    }

    MPI_Gather(local_C.data(), rows_per_process * N, MPI_INT,
               C.data(), rows_per_process * N, MPI_INT,
               0, MPI_COMM_WORLD);
}

void OMPMPImethod(const std::vector<int>& A, const std::vector<int>& B, std::vector<int>& C,
                  int M, int K, int N, int Mtile, int Ktile, int Ntile) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int rows_per_process = M / size;
    int start_row = rank * rows_per_process;
    int end_row = (rank + 1) * rows_per_process;

    std::vector<int> local_C(rows_per_process * N, 0);

    for (int i = start_row; i < end_row; i += Mtile) {
        for (int j = 0; j < N; j += Ntile) {
            for (int k = 0; k < K; k += Ktile) {
                std::vector<int> temp(Mtile * Ntile, 0);
                
                #pragma omp parallel for
                for (int ii = i; ii < std::min(i + Mtile, end_row); ++ii) {
                    for (int jj2 = j; jj2 < std::min(j + Ntile, N); ++jj2) {
                        int sum = 0;
                        for (int jj = k; jj < std::min(k + Ktile, K); ++jj) {
                            sum += A[ii * K + jj] * B[jj * N + jj2];
                        }
                        temp[(ii - i) * Ntile + (jj2 - j)] += sum;
                    }
                }

                for (int ii = i; ii < std::min(i + Mtile, end_row); ++ii) {
                    for (int jj2 = j; jj2 < std::min(j + Ntile, N); ++jj2) {
                        local_C[(ii - start_row) * N + jj2] += temp[(ii - i) * Ntile + (jj2 - j)];
                    }
                }
            }
        }
    }

    MPI_Gather(local_C.data(), rows_per_process * N, MPI_INT,
               C.data(), rows_per_process * N, MPI_INT,
               0, MPI_COMM_WORLD);
}

int main(int argc, char **argv) {
    const int N = 6; // 512
    const int M = 6;
    const int K = 4;

    const int Mtile = 2; //256
    const int Ktile = 3;
    const int Ntile = 2;

    std::vector<int> A;
    std::vector<int> B;
    std::vector<int> C(N * K); 

    generateRandom(A, B, C, N, M, K);
   
    if(argc != 2) {
        std::cerr << "Usage: mpirun -np <num_procs> ./prog <method>\n";
        std::cerr << "Method: \n";
        std::cerr << "0 - Block\n";
        std::cerr << "1 - OpenMP\n";
        std::cerr << "2 - MPI\n";
        std::cerr << "3 - Hybrid (OpenMP + MPI)\n";
        return EXIT_FAILURE;
    }

    int method_choice = atoi(argv[1]);
    if(method_choice < 0 || method_choice > 3) {
        std::cerr << "Invalid method\n";
        return EXIT_FAILURE;
    }

    std::chrono::high_resolution_clock::time_point start, end;
    std::chrono::duration<double> duration;
    int rank;

    switch (atoi(argv[1])) {
        case 0:
            start = std::chrono::high_resolution_clock::now();
            blockMultiply(A, B, C, N, M, K, Mtile, Ktile, Ntile);
            end = std::chrono::high_resolution_clock::now();
            break;

        case 1:
            start = std::chrono::high_resolution_clock::now();
            OMPmethod(A, B, C, N, M, K, Mtile, Ktile, Ntile);
            end = std::chrono::high_resolution_clock::now();
            break;

        case 2:
            start = std::chrono::high_resolution_clock::now();
            MPI_Init(&argc,&argv);
            MPI_Comm_rank(MPI_COMM_WORLD,&rank);
            MPImethod(A, B, C, N, M, K, Mtile, Ktile, Ntile);
            MPI_Finalize();
            end = std::chrono::high_resolution_clock::now();
            break;

        case 3:
            start = std::chrono::high_resolution_clock::now();
            MPI_Init(&argc,&argv);
            MPI_Comm_rank(MPI_COMM_WORLD,&rank);
            OMPMPImethod(A, B, C, N, M, K, Mtile, Ktile, Ntile);
            MPI_Finalize();
            end = std::chrono::high_resolution_clock::now();
            break;

        default:
            std::cerr << "Invalid method\n";
            break;
   }

    switch (atoi(argv[1])) {
        case 0:
            std::cout << "Time taken by block method: ";
            duration = end - start;
            std::cout << duration.count() << " seconds\n";

            //std::cout << "Result:\n";
            //printMatrix(C, N, K);
            break;

        case 1:
            std::cout << "Time taken by OpenMP method: ";
            duration = end - start;
            std::cout << duration.count() << " seconds\n";

            //std::cout << "Result:\n";
            //printMatrix(C, N, K);
            break;

        case 2:
            if (rank == 0) {
                std::cout << "Time taken by MPI method: ";
                duration = end - start;
                std::cout << duration.count() << " seconds\n";

                //std::cout << "Result:\n";
                //printMatrix(C, N, K);
            }
            break;
        
        case 3:
            if (rank == 0) {
                std::cout << "Time taken by hybrid method: ";
                duration = end - start;
                std::cout << duration.count() << " seconds\n";

                //std::cout << "Result:\n";
                //printMatrix(C, N, K);
            }
            break;
        }
   
    return EXIT_SUCCESS;
}