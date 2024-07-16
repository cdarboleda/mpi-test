#include <iostream>
#include <mpi.h>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <random>

#define DIMENSION 3
#define MAX_ITEMS_A 30
#define MAX_ITEMS_C 3

void random_vector(std::vector<std::vector<double>>& data, int size) {
    std::random_device rd;
    std::default_random_engine gen(rd());
    std::uniform_real_distribution<> distrib(1, 10);

    data.resize(size, std::vector<double>(DIMENSION));

    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < DIMENSION; ++j) {
            data[i][j] = distrib(gen);
        }
    }
}

double distancia_euc(const std::vector<double>& a, const std::vector<double>& b) {
    double dist = 0;
    for (int i = 0; i < DIMENSION; ++i) {
        dist += (a[i] - b[i]) * (a[i] - b[i]);
    }
    return std::sqrt(dist);
}

int index_cercano(const std::vector<double>& point, const std::vector<std::vector<double>>& centros_A) {
    int indice = 0;
    double dist_minima = distancia_euc(point, centros_A[0]);
    for (int i = 1; i < MAX_ITEMS_C; ++i) {
        double dist = distancia_euc(point, centros_A[i]);
        if (dist < dist_minima) {
            dist_minima = dist;
            indice = i;
        }
    }
    return indice;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    srand(time(0) + rank);

    std::vector<std::vector<double>> A;
    std::vector<std::vector<double>> C;
    std::vector<int> centros_cercanos_local;
    std::vector<int> centros_cercanos(MAX_ITEMS_A);  // Tamaño fijo para recibir datos

    if (rank == 0) {
        random_vector(A, MAX_ITEMS_A);
        random_vector(C, MAX_ITEMS_C);

        std::printf("Centros:\n");
        for (std::vector<double> center : C) {
            std::printf("(%.0f %.0f, %.0f)\n", center[0], center[1], center[2]);
        }
    }

    // Envío de los centros C a todos los procesos
    for (int i = 0; i < MAX_ITEMS_C; ++i) {
        MPI_Bcast(C[i].data(), DIMENSION, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    // Distribución de los datos de A entre los procesos
    int block_size = MAX_ITEMS_A / nprocs;
    int padding = MAX_ITEMS_A % nprocs;
    int local_size = block_size + (rank == 0 ? padding : 0);
    std::vector<std::vector<double>> local_A(local_size, std::vector<double>(DIMENSION));
    MPI_Scatter(A.data(), local_size * DIMENSION, MPI_DOUBLE,
                local_A.data(), local_size * DIMENSION, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    // Cálculo de los índices más cercanos para cada elemento local de A
    centros_cercanos_local.resize(local_size);
    for (int i = 0; i < local_size; ++i) {
        centros_cercanos_local[i] = index_cercano(local_A[i], C);
    }

    // Recopilación de los índices más cercanos en el proceso raíz
    MPI_Gather(centros_cercanos_local.data(), local_size, MPI_INT,
               centros_cercanos.data(), local_size, MPI_INT,
               0, MPI_COMM_WORLD);

    // Impresión de los resultados en el proceso raíz
    if (rank == 0) {
        std::printf("\nVector\tÍndice\n");
        for (int i = 0; i < MAX_ITEMS_A; ++i) {
            std::printf("(%.0f %.0f, %.0f)\t%d\n", A[i][0], A[i][1], A[i][2], centros_cercanos[i]);
        }
    }

    MPI_Finalize();
    return 0;
}
