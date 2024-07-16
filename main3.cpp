#include <iostream>
#include <mpi.h>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>

#define NUM_VECTORS 30
#define DIMENSION 3

// Función para calcular la distancia euclidiana
double euclidean_distance(const std::vector<double>& vec1, const std::vector<double>& vec2) {
    double sum = 0.0;
    for (int i = 0; i < DIMENSION; ++i) {
        sum += (vec1[i] - vec2[i]) * (vec1[i] - vec2[i]);
    }
    return sqrt(sum);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    std::vector<std::vector<double>> A(NUM_VECTORS, std::vector<double>(DIMENSION));
    std::vector<std::vector<double>> C(DIMENSION, std::vector<double>(DIMENSION));
    std::vector<int> closest_indices(NUM_VECTORS);

    // Inicializar los vectores aleatoriamente en el proceso raíz
    if (rank == 0) {
        std::srand(std::time(0));
        for (auto& vec : A) {
            for (auto& val : vec) {
                val = 1 + std::rand() % 10;
            }
        }
        for (auto& vec : C) {
            for (auto& val : vec) {
                val = 1 + std::rand() % 10;
            }
        }
    }

    // Transmitir los vectores C a todos los procesos
    for (auto& vec : C) {
        MPI_Bcast(vec.data(), DIMENSION, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    // Distribuir los vectores A entre los procesos
    int vectors_per_proc = NUM_VECTORS / nprocs;
    std::vector<std::vector<double>> local_A(vectors_per_proc, std::vector<double>(DIMENSION));

    MPI_Scatter(A.data(), vectors_per_proc * DIMENSION, MPI_DOUBLE,
                local_A.data(), vectors_per_proc * DIMENSION, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    // Calcular el índice del vector más cercano para los vectores locales
    std::vector<int> local_closest_indices(vectors_per_proc);
    for (int i = 0; i < vectors_per_proc; ++i) {
        double min_distance = euclidean_distance(local_A[i], C[0]);
        int min_index = 0;
        for (int j = 1; j < DIMENSION; ++j) {
            double distance = euclidean_distance(local_A[i], C[j]);
            if (distance < min_distance) {
                min_distance = distance;
                min_index = j;
            }
        }
        local_closest_indices[i] = min_index;
    }

    // Recoger los resultados de todos los procesos
    MPI_Gather(local_closest_indices.data(), vectors_per_proc, MPI_INT,
               closest_indices.data(), vectors_per_proc, MPI_INT,
               0, MPI_COMM_WORLD);

    // Imprimir los resultados en el proceso raíz
    if (rank == 0) {
        std::cout << "Centros:\n";
        for (const auto& vec : C) {
            std::cout << "(";
            for (size_t i = 0; i < vec.size(); ++i) {
                std::cout << vec[i] << (i < vec.size() - 1 ? "," : "");
            }
            std::cout << ")\n";
        }
        std::cout << "\nVector Indice\n";
        for (size_t i = 0; i < A.size(); ++i) {
            std::cout << "(";
            for (size_t j = 0; j < A[i].size(); ++j) {
                std::cout << A[i][j] << (j < A[i].size() - 1 ? "," : "");
            }
            std::cout << ") " << closest_indices[i] << "\n";
        }
    }

    MPI_Finalize();
    return 0;
}