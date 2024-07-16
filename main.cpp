#include <iostream>
#include <mpi.h>
#include <vector>
#include <memory>
#include <cmath>

#define MATRIX_DIMENSION 25

void matrix_mult(double* A, double* b, double* x, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        x[i] = 0;
        for (int j = 0; j < cols; ++j) {
            x[i] += A[i * cols + j] * b[j];
        }
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    int row_per_rank;
    int rows_alloc = MATRIX_DIMENSION;
    int padding = 0;

    if (MATRIX_DIMENSION % nprocs != 0) {
        rows_alloc = std::ceil((double)MATRIX_DIMENSION / nprocs) * nprocs;
        padding = rows_alloc - MATRIX_DIMENSION;
    }
    row_per_rank = rows_alloc / nprocs;

    std::unique_ptr<double[]> A;
    std::unique_ptr<double[]> b = std::make_unique<double[]>(MATRIX_DIMENSION);
    std::unique_ptr<double[]> x;
    std::unique_ptr<double[]> A_local = std::make_unique<double[]>(row_per_rank * MATRIX_DIMENSION);
    std::unique_ptr<double[]> x_local = std::make_unique<double[]>(row_per_rank);

    if (rank == 0) {
        A = std::make_unique<double[]>(rows_alloc * MATRIX_DIMENSION);
        x = std::make_unique<double[]>(rows_alloc);

        for (int i = 0; i < MATRIX_DIMENSION; ++i) {
            for (int j = 0; j < MATRIX_DIMENSION; ++j) {
                int index = i * MATRIX_DIMENSION + j;
                A[index] = i;
            }
        }

        for (int i = 0; i < MATRIX_DIMENSION; ++i) {
            b[i] = 1;
        }

        std::printf("Dimension: %d, rows_alloc: %d, rows_per_rank: %d, padding: %d\n",
                    MATRIX_DIMENSION, rows_alloc, row_per_rank, padding);
    }

    MPI_Bcast(b.get(), MATRIX_DIMENSION, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatter(A.get(), MATRIX_DIMENSION * row_per_rank, MPI_DOUBLE,
                A_local.get(), MATRIX_DIMENSION * row_per_rank, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    int row_per_rank_tmp = row_per_rank;
    if (rank == nprocs - 1 && padding > 0) {
        row_per_rank_tmp -= padding;
    }

    matrix_mult(A_local.get(), b.get(), x_local.get(), row_per_rank_tmp, MATRIX_DIMENSION);

    MPI_Gather(x_local.get(), row_per_rank, MPI_DOUBLE,
               x.get(), row_per_rank, MPI_DOUBLE,
               0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::printf("Resultado\n");
        for (int i = 0; i < MATRIX_DIMENSION; ++i) {
            std::printf("%.0f,", x[i]);
        }
        std::printf("\n");
    }

    MPI_Finalize();

    return 0;
}