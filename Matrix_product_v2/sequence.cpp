#include <iostream>
#include <mpi.h>
#include <omp.h>
#include <iomanip>
#include <cmath>
#include <vector>
#include <random>
#include <chrono>

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

void generateRandom(std::vector<int>& M1, std::vector<int>& M2, std::vector<int>& resM, int n, int m, int k) {
    M1.resize(n * m);
    M2.resize(m * k);
    resM.resize(n * k);

    unsigned seed = 42;
    std::mt19937 gen(seed);
    std::uniform_int_distribution<> dis(-10, 10);

    for (int i = 0; i < n * m; ++i) {
        M1[i] = dis(gen);
    }
    
    for (int i = 0; i < m * k; ++i) {
        M2[i] = dis(gen);
    }

    std::fill(resM.begin(), resM.end(), 0);
}

void SEQmethod(const std::vector<int>& M1, const std::vector<int>& M2, std::vector<int>& resM, int n, int m, int k) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < k; ++j) {
            resM[i * k + j] = 0;
            for (int t = 0; t < m; ++t) {
                resM[i * k + j] += M1[i * m + t] * M2[t * k + j];
            }
        }
    }
}

void OMPmethod(const std::vector<int>& M1, const std::vector<int>& M2, std::vector<int>& resM, int n, int m, int k) {
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < k; ++j) {
            resM[i * k + j] = 0;
            for (int t = 0; t < m; ++t) {
                resM[i * k + j] += M1[i * m + t] * M2[t * k + j];
            }
        }
    }
}

void MPImethod(const std::vector<int>& M1, const std::vector<int>& M2, std::vector<int>& resM, int n, int m, int k) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int rows_per_process = n / size;

    std::vector<int> local_M1(rows_per_process * m);
    std::vector<int> local_resM(rows_per_process * k);

    MPI_Scatter(M1.data(), rows_per_process * m, MPI_INT,
                local_M1.data(), rows_per_process * m, MPI_INT,
                0, MPI_COMM_WORLD);

    MPI_Bcast(const_cast<int*>(M2.data()), m * k, MPI_INT, 0, MPI_COMM_WORLD);

    for (int i = 0; i < rows_per_process; ++i) {
        for (int j = 0; j < k; ++j) {
            local_resM[i * k + j] = 0;
            for (int t = 0; t < m; ++t) {
                local_resM[i * k + j] += local_M1[i * m + t] * M2[t * k + j];
            }
        }
    }

    MPI_Gather(local_resM.data(), rows_per_process * k, MPI_INT,
               resM.data(), rows_per_process * k, MPI_INT,
               0, MPI_COMM_WORLD);
}

void OMPMPImethod(const std::vector<int>& M1, const std::vector<int>& M2, std::vector<int>& resM, int n, int m, int k) {
    int rank, size;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    int rows_per_process = n / size;

    std::vector<int> local_M1(rows_per_process * m);
    std::vector<int> local_resM(rows_per_process * k);

    MPI_Scatter(M1.data(), rows_per_process * m, MPI_INT,
                local_M1.data(), rows_per_process * m, MPI_INT,
                0, MPI_COMM_WORLD);

   #pragma omp parallel for collapse(2)
   for (int i = 0; i < rows_per_process; ++i) {
       for (int j = 0; j < k; ++j) {
           local_resM[i * k + j] = 0;
           for (int t = 0; t < m; ++t) {
               local_resM[i * k + j] += local_M1[i * m + t] * M2[t * k + j];
           }
       }
   }

   MPI_Gather(local_resM.data(), rows_per_process * k, MPI_INT,
              resM.data(), rows_per_process * k, MPI_INT,
              0, MPI_COMM_WORLD);
}

int main(int argc, char **argv) {
    const int n = 512;
    const int m = 512;
    const int k = 512;

    std::vector<int> M1;
    std::vector<int> M2;
    std::vector<int> resM(n*k); 

    generateRandom(M1, M2, resM, n, m, k);
   
    if(argc != 2) {
        std::cerr << "Usage: mpirun -np <num_procs> ./prog <method>\n";
        std::cerr << "Method: \n";
        std::cerr << "0 - Sequential\n";
        std::cerr << "1 - OpenMP\n";
        std::cerr << "2 - MPI\n";
        std::cerr << "3 - Hybrid(OpenMP + MPI)\n";
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
            SEQmethod(M1, M2, resM, n, m, k);
            end = std::chrono::high_resolution_clock::now();
            break;

        case 1:
            start = std::chrono::high_resolution_clock::now();
            OMPmethod(M1, M2, resM, n, m, k);
            end = std::chrono::high_resolution_clock::now();
            break;

        case 2:
            start = std::chrono::high_resolution_clock::now();
            MPI_Init(&argc,&argv);
            MPI_Comm_rank(MPI_COMM_WORLD,&rank);
            MPImethod(M1, M2, resM, n, m, k);
            MPI_Finalize();
            end = std::chrono::high_resolution_clock::now();
            break;

        case 3:
            start = std::chrono::high_resolution_clock::now();
            MPI_Init(&argc,&argv);
            MPI_Comm_rank(MPI_COMM_WORLD,&rank);
            OMPMPImethod(M1, M2, resM, n, m, k);
            MPI_Finalize();
            end = std::chrono::high_resolution_clock::now();
            break;

        default:
            std::cerr << "Invalid method\n";
            break;
   }

    switch (atoi(argv[1])) {
        case 0:
            std::cout << "Time taken by sequential method: ";
            duration = end - start;
            std::cout << duration.count() << " seconds\n";

            //std::cout << "Result:\n";
            //printMatrix(resM, n, k);
            break;

        case 1:
            std::cout << "Time taken by OpenMP method: ";
            duration = end - start;
            std::cout << duration.count() << " seconds\n";

            //std::cout << "Result:\n";
            //printMatrix(resM, n, k);
            break;

        case 2:
            if (rank == 0) {
                std::cout << "Time taken by MPI method: ";
                duration = end - start;
                std::cout << duration.count() << " seconds\n";

                //std::cout << "Result:\n";
                //printMatrix(resM, n, k);
            }
            break;
        
        case 3:
            if (rank == 0) {
                std::cout << "Time taken by hybrid method: ";
                duration = end - start;
                std::cout << duration.count() << " seconds\n";

                //std::cout << "Result:\n";
                //printMatrix(resM, n, k);
            }
            break;
        }
   
    return EXIT_SUCCESS;
}