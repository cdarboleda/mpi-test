#include <iostream>
#include <mpi.h>
#include <vector>
#include <cmath>

// Dimensiones
#define A_SIZE 30
#define C_SIZE 3
#define DIM 3

// Función para generar datos aleatorios
void generate_random_data(std::vector<std::vector<double>>& data, int size) {
    for (int i = 0; i < size; ++i) {
        std::vector<double> vec(DIM);
        for (int j = 0; j < DIM; ++j) {
            vec[j] = 1 + rand() % 10; // Números aleatorios entre 1 y 10
        }
        data.push_back(vec);
    }
}

// Función para calcular la distancia euclidiana
double euclidean_distance(const std::vector<double>& a, const std::vector<double>& b) {
    double sum = 0.0;
    for (int i = 0; i < DIM; ++i) {
        sum += (a[i] - b[i]) * (a[i] - b[i]);
    }
    return std::sqrt(sum);
}

// Función para encontrar el índice del vector más cercano
int find_closest_center(const std::vector<double>& point, const std::vector<std::vector<double>>& centers) {
    int min_index = 0;
    double min_dist = euclidean_distance(point, centers[0]);
    for (int i = 1; i < C_SIZE; ++i) {
        double dist = euclidean_distance(point, centers[i]);
        if (dist < min_dist) {
            min_dist = dist;
            min_index = i;
        }
    }
    return min_index;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    srand(time(0) + rank);

    std::vector<std::vector<double>> A;
    std::vector<std::vector<double>> C = {{1, 4, 7}, {2, 6, 9}, {0, 8, 2}};
    std::vector<int> local_closest_centers;
    std::vector<int> closest_centers;

    if (rank == 0) {
        generate_random_data(A, A_SIZE);

        std::cout << "Conjunto C (Centros):\n";
        for (const auto& center : C) {
            std::cout << "(" << center[0] << "," << center[1] << "," << center[2] << ")\n";
        }

        closest_centers.resize(A_SIZE);
    }

    // Distribuir los centros a todos los procesos
    for (int i = 0; i < C_SIZE; ++i) {
        if (rank != 0) {
            C.push_back(std::vector<double>(DIM));
        }
        MPI_Bcast(C[i].data(), DIM, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    // Dividir el conjunto A entre los procesos
    int elements_per_proc = A_SIZE / nprocs;
    int remainder = A_SIZE % nprocs;
    if (rank == 0) {
        local_closest_centers.resize(elements_per_proc + remainder);
    } else {
        local_closest_centers.resize(elements_per_proc);
    }

    std::vector<double> local_A((elements_per_proc + (rank == 0 ? remainder : 0)) * DIM);
    std::vector<double> flat_A;

    if (rank == 0) {
        flat_A.resize(A_SIZE * DIM);
        for (int i = 0; i < A_SIZE; ++i) {
            for (int j = 0; j < DIM; ++j) {
                flat_A[i * DIM + j] = A[i][j];
            }
        }
    }

    MPI_Scatter(flat_A.data(), elements_per_proc * DIM, MPI_DOUBLE,
                local_A.data(), elements_per_proc * DIM, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::copy(flat_A.begin() + elements_per_proc * nprocs * DIM, flat_A.end(), local_A.begin() + elements_per_proc * DIM);
    }

    std::vector<std::vector<double>> local_A_vectors(local_A.size() / DIM, std::vector<double>(DIM));
    for (int i = 0; i < local_A.size() / DIM; ++i) {
        for (int j = 0; j < DIM; ++j) {
            local_A_vectors[i][j] = local_A[i * DIM + j];
        }
    }

    for (int i = 0; i < local_A_vectors.size(); ++i) {
        local_closest_centers[i] = find_closest_center(local_A_vectors[i], C);
    }

    MPI_Gather(local_closest_centers.data(), elements_per_proc, MPI_INT,
               closest_centers.data(), elements_per_proc, MPI_INT,
               0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::copy(local_closest_centers.begin() + elements_per_proc, local_closest_centers.end(), closest_centers.begin() + elements_per_proc * nprocs);

        std::cout << "\nVector\tÍndice\n";
        for (int i = 0; i < A_SIZE; ++i) {
            std::cout << "(" << flat_A[i * DIM] << "," << flat_A[i * DIM + 1] << "," << flat_A[i * DIM + 2] << ")\t" << closest_centers[i] << "\n";
        }
    }

    MPI_Finalize();
    return 0;
}