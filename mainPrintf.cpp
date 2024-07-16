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


double distancia_euc(const std::vector<double>& a, const std::vector<double>& b) {
    double dist = 0;
    for (int i = 0; i < DIMENSION; ++i) {
        dist += (a[i] - b[i]) * (a[i] - b[i]);
    }
    return std::sqrt(dist);
}

void random_vector(std::vector<std::vector<double>>& data, int size) {
    for (int i = 0; i < size; ++i) {
        std::vector<double> vec(DIMENSION);
        for (int j = 0; j < DIMENSION; ++j) {
            vec[j] = 1 + rand() % 10;
        }
        data.push_back(vec);
    }
}

// void random_vector(std::vector<std::vector<double>>& data, int size) {

//     std::random_device rd;
//     std::default_random_engine gen(rd());
//     std::uniform_real_distribution<> distrib(1, 10);

//     data.resize(size, std::vector<double>(DIMENSION));

//     for (int i = 0; i < size; ++i) {
//         for (int j = 0; j < DIMENSION; ++j) {
//             data[i][j] = distrib(gen);
//         }
//     }
// }



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
    std::vector<int> centros_cercanos;

    if (rank == 0) {
        random_vector(C, MAX_ITEMS_C);
        random_vector(A, MAX_ITEMS_A);
        
        std::printf("Centros:\n");

        for (std::vector<double> center : C) {
            std::printf("(%.0f %.0f, %.0f)\n", center[0], center[1], center[2]);
        }
        centros_cercanos.resize(MAX_ITEMS_A);
    }

    //El vector c de centrosse pasa a todos los ranks
    for (int i = 0; i < MAX_ITEMS_C; ++i) {
        if (rank != 0) {
            C.push_back(std::vector<double>(DIMENSION));
        }
        MPI_Bcast(C[i].data(), DIMENSION, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }


    int block_size = MAX_ITEMS_A / nprocs;
    int padding = MAX_ITEMS_A % nprocs;

    std::vector<double> local_A;
    std::vector<double> A_vector;

    if (rank == 0) {
        A_vector.resize(MAX_ITEMS_A * DIMENSION);
        local_A.resize((block_size + padding) * DIMENSION);
        centros_cercanos_local.resize(block_size + padding);
        for (int i = 0; i < MAX_ITEMS_A; ++i) {
            for (int j = 0; j < DIMENSION; ++j) {
                A_vector[i * DIMENSION + j] = A[i][j];
            }
        }
    }else{
        local_A.resize(block_size * DIMENSION);
        centros_cercanos_local.resize(block_size);
    }

    MPI_Scatter(A_vector.data(), block_size * DIMENSION, MPI_DOUBLE, local_A.data(), block_size * DIMENSION, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::copy(A_vector.begin() + block_size * nprocs * DIMENSION, A_vector.end(), local_A.begin() + block_size * DIMENSION);
    }

    std::vector<std::vector<double>> local_A_v(local_A.size() / DIMENSION, std::vector<double>(DIMENSION));
    for (int i = 0; i < local_A.size() / DIMENSION; ++i) {
        for (int j = 0; j < DIMENSION; ++j) {
            local_A_v[i][j] = local_A[i * DIMENSION + j];
        }
    }

    for (int i = 0; i < local_A_v.size(); ++i) {
        centros_cercanos_local[i] = index_cercano(local_A_v[i], C);
    }

    MPI_Gather(centros_cercanos_local.data(), block_size, MPI_INT, centros_cercanos.data(), block_size, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::copy(centros_cercanos_local.begin() + block_size, centros_cercanos_local.end(), centros_cercanos.begin() + block_size * nprocs);

        std::printf("\nVector\tÍndice\n");
        for (int i = 0; i < MAX_ITEMS_A; ++i) {
            std::printf("(%.0f %.0f, %.0f)\t%d\n", A_vector[i * DIMENSION], A_vector[i * DIMENSION + 1], A_vector[i * DIMENSION + 2], centros_cercanos[i]);
        }
    }

    MPI_Finalize();
    return 0;
}
