#include <iostream>
#include <mpi.h>
#include <cmath>
#include <vector>
#include <random>

#define MAX_ITEMS 25
#define DIMENSION 3

int distancia_euclidiana_index(int *a, int *c, int dimension)
{
    int indice = 0;
    std::vector<double> distancias(dimension);

    for (int j = 0; j < dimension * dimension; j += 3)
    {
        double tmp = pow((a[0] - c[j]), 2) + pow((a[1] - c[j + 1]), 2) + pow((a[2] - c[j + 2]), 2);
        distancias[j / 3] = sqrt(tmp);
    }

    double dist_minima = distancias[0];
    for (int k = 1; k < 3; k++)
    {
        if (distancias[k] < dist_minima)
        {
            dist_minima = distancias[k];
            indice = k;
        }
    }
    return indice;
}

int *indices_aux(int *a, int *c, int block_size)
{
    std::vector<int> indices_aux;
    std::vector<int> a_pequenio(3);
    for (int i = 0; i < block_size; i += 3)
    {
        a_pequenio[0] = a[i];
        a_pequenio[1] = a[i + 1];
        a_pequenio[2] = a[i + 2];

        int indice = distancia_euclidiana_index(a_pequenio.data(), c, DIMENSION);
       
        indices_aux.push_back(indice);
    }
/*
    for (int j : indices_aux)
    {
        std::printf("Distancia indice: %d", j);
    }
*/
    return indices_aux.data();
}

void imprimir(int *a, int *c, int block_size)
{
    std::vector<int> a_pequenio(3);
    for (size_t i = 0; i < block_size; i += 3)
    {
        a_pequenio[0] = a[i];
        a_pequenio[1] = a[i + 1];
        a_pequenio[2] = a[i + 2];

        int indice = distancia_euclidiana_index(a_pequenio.data(), c, DIMENSION);

        for (int i = 0; i < a_pequenio.size(); i++)
            std::printf("|\%d, ", a_pequenio[i]);

        std::printf("\t| %d\n", indice);
    }
}
int main(int argc, char *argv[])
{
    int rank, nprocs;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    int real_size = MAX_ITEMS;
    int block_size; // Cantidad de filas/vectores de A que tiene cada rank
    int padding = 0;

    if (MAX_ITEMS % nprocs != 0)
    {
        real_size = std::ceil((double)MAX_ITEMS / nprocs) * nprocs;
        padding = real_size - MAX_ITEMS;
    }
    block_size = real_size / nprocs;

    // Vector A
    std::vector<int> A;
    // Vector indices
    std::vector<int> indices(real_size);
    // Conjunto C
    std::vector<int> C(DIMENSION * DIMENSION);

    if (rank == 0)
    {

        std::random_device rd;
        std::default_random_engine gen(rd());
        std::uniform_real_distribution<> distrib(1, 10);

        A.resize(real_size * DIMENSION); // A tiene 28*3 (28 vectores de 3)
        for (int i = 0; i < MAX_ITEMS * DIMENSION; i++)
            A[i] = distrib(gen); // std::printf("%d, \n", A[i]);

        for (int i = 0; i < DIMENSION * DIMENSION; i++) // C tiene 3*3 elementos (3 vectores de 3)
            C[i] = distrib(gen);

        // Cada rank tiene block_size de vectores de 3 dimensiones
        MPI_Scatter(A.data(), block_size * DIMENSION, MPI_INT, MPI_IN_PLACE, 0, MPI_INT, 0, MPI_COMM_WORLD);
        // Pasamos el C a cada rank, C tiene 3*3
        MPI_Bcast(C.data(), DIMENSION * DIMENSION, MPI_INT, 0, MPI_COMM_WORLD);

        int *inn = indices_aux(A.data(), C.data(), block_size * DIMENSION);
        MPI_Gather(inn, block_size, MPI_INT, indices.data(), block_size, MPI_INT, 0, MPI_COMM_WORLD);

        std::string str = "";
        for (int i = 0; i < block_size * DIMENSION; i++)
        {
            str = str + std::to_string(A[i]) + ",";
        }
        std::printf("RANK_%d recibiendo datos==>%s\n", rank, str.c_str());
        // imprimir(A.data(), C.data(), block_size*DIMENSION);

        imprimir(A.data(), indices.data(), real_size * DIMENSION);
    }
    else
    {

        // A local debe tener block_size vectores de 3 dimensiones (7 vectores de 3 elem)
        std::vector<int> A_local(block_size * DIMENSION);
        // C local tiene los 3 vectores de 3
        std::vector<int> C_local(DIMENSION * DIMENSION);
        std::vector<int> indices_local(real_size);

        MPI_Scatter(nullptr, 0, MPI_INT, A_local.data(), block_size * DIMENSION, MPI_INT, 0, MPI_COMM_WORLD);

        std::string str = "";
        for (int i = 0; i < block_size * DIMENSION; i++)
        {
            str = str + std::to_string(A_local[i]) + ",";
        }
        std::printf("RANK_%d recibiendo datos==>%s\n", rank, str.c_str());

        MPI_Bcast(C_local.data(), DIMENSION * DIMENSION, MPI_INT, 0, MPI_COMM_WORLD);

        int block_size_tmp = block_size;
        if (rank == nprocs - 1)
        {
            block_size_tmp = block_size_tmp - padding;
        }

        int *inn = indices_aux(A_local.data(), C_local.data(), block_size_tmp * DIMENSION);
        /*
        for (int i = 0; i < block_size_tmp; ++i)
        {
            std::cout << "Elemento " << i << ": " << inn[i] << std::endl;

            // Realiza operaciones adicionales según tus necesidades
        }
*/
        MPI_Gather(inn, block_size, MPI_INT, nullptr, 0, MPI_INT, 0, MPI_COMM_WORLD);

        //imprimir(A_local.data(), C_local.data(), block_size_tmp * DIMENSION);
    }

    MPI_Finalize();

    return 0;
}