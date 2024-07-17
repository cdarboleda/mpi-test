#include <iostream>
#include <vector>
#include <mpi.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    int width, height, channels;
    uint8_t* rgb_pixels;

    if (rank == 0) {
        rgb_pixels = stbi_load("image01.jpg", &width, &height, &channels, STBI_rgb);
    }

    MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&height, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&channels, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int rows_per_process = height / nprocs;
    int local_pixels = rows_per_process * width * channels; //esta es la cantidad total de pixeles que va a trabajr cada rank

    uint8_t* local_rgb_pixels = new uint8_t[local_pixels];//es el local para cada rank
    MPI_Scatter(rgb_pixels, local_pixels, MPI_UNSIGNED_CHAR, local_rgb_pixels, local_pixels, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    uint8_t* local_red_pixels = new uint8_t[local_pixels];//cada uno tiene un localsito de color
    uint8_t* local_green_pixels = new uint8_t[local_pixels];
    uint8_t* local_blue_pixels = new uint8_t[local_pixels];

    for (int i = 0; i < rows_per_process; i++) {//cada uno debe moverse por la cantidad de filas que maneja, aqui filas apunta i, y j apunta a lacolumna
        for (int j = 0; j < width; j++) {
            int index = (i * width + j) * channels;

            local_red_pixels[index] = local_rgb_pixels[index];
            local_red_pixels[index + 1] = 0;
            local_red_pixels[index + 2] = 0;

            local_green_pixels[index] = 0;
            local_green_pixels[index + 1] = local_rgb_pixels[index + 1];
            local_green_pixels[index + 2] = 0;

            local_blue_pixels[index] = 0;
            local_blue_pixels[index + 1] = 0;
            local_blue_pixels[index + 2] = local_rgb_pixels[index + 2];
        }
    }

    uint8_t* red_pixels = nullptr; //estos son los globales
    uint8_t* green_pixels = nullptr;
    uint8_t* blue_pixels = nullptr;

    if (rank == 0) {
        red_pixels = new uint8_t[width * height * channels];
        green_pixels = new uint8_t[width * height * channels];
        blue_pixels = new uint8_t[width * height * channels];
    }

    //reunimos todos los locales en los globales
    MPI_Gather(local_red_pixels, local_pixels, MPI_UNSIGNED_CHAR, red_pixels, local_pixels, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
    MPI_Gather(local_green_pixels, local_pixels, MPI_UNSIGNED_CHAR, green_pixels, local_pixels, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
    MPI_Gather(local_blue_pixels, local_pixels, MPI_UNSIGNED_CHAR, blue_pixels, local_pixels, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    if (rank == 0) {//guardamos y ya
        stbi_write_png("img-red.png", width, height, STBI_rgb, red_pixels, width * channels);
        stbi_write_png("img-green.png", width, height, STBI_rgb, green_pixels, width * channels);
        stbi_write_png("img-blue.png", width, height, STBI_rgb, blue_pixels, width * channels);
    }

    MPI_Finalize();

    return 0;
}