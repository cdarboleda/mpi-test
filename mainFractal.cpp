#include <iostream>
#include <vector>
#include <mpi.h>
#include <cmath>
#include <cstdlib>
#include <ctime>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
 
#define WIDTH 1200
#define HEIGHT 600
#define PALET_SIZE 16
 
int max_iteraciones = 1000;
 
const double x_min = -2;
const double x_max = 1;
 
const double y_min = -1;
const double y_max = 1;
 
uint32_t _bswap32(uint32_t a) {
    return
        ((a & 0X000000FF) << 24) |
        ((a & 0x0000FF00) << 8) |
        ((a & 0x00FF0000) << 8) |
        ((a & 0xFF000000) >> 24);
};

uint32_t _bswap32v2(uint32_t colorHex) {
    // Asegurarse de que el color hexadecimal esté en formato ARGB (32 bits)
    // Formato: 0xAARRGGBB
    
    uint32_t argbColor = 0xFF000000; // Iniciar con canal alfa (Alpha) establecido a 255 (completamente opaco)

    // Extraer componentes de color RGB del valor hexadecimal
    uint32_t rgb = colorHex & 0xFFFFFF; // Extraer los bits de RGB
    
    // Combinar los valores en el formato ARGB
    argbColor |= rgb; // Combinar RGB con ARGB

    return argbColor;
};

 
 
std::vector<unsigned int> color_ramp = {
    0xFF1010FF, 0xEF1019FF, 0xE01123FF, 0xD1112DFF,
    0xC11237FF, 0xB21341FF, 0xA3134BFF, 0x931455FF,
    0x84145EFF, 0x751568FF, 0x651672FF, 0x56167CFF,
    0x471786FF, 0x371790FF, 0x28189AFF, 0x1919A4FF
};

unsigned int divergente(double cx, double cy) {
    int iter = 0;
    double vx = cx;
    double vy = cy;
    while (iter < max_iteraciones && (vx * vx + vy * vy) <= 4) {
        double tx = vx * vx - vy * vy + cx;
        double ty = 2 * vx * vy + cy;
        vx = tx;
        vy = ty;
        iter++;
    }

    if((vx*vx+vy*vy)>4){
        int color_idx= iter % PALET_SIZE;
        return _bswap32v2(color_ramp[color_idx]);
    }
 
    if (iter > 0 && iter < max_iteraciones) {
        int color_idx = iter % PALET_SIZE;
        return _bswap32v2(color_ramp[color_idx]);
    } else {
        return  _bswap32v2(0x0000FFFF);
    }
}
 

 /*
  
void mandelbrotCpu(){
    double dx = (x_max - x_min)/ WIDTH;
    double dy = (y_max - y_min)/HEIGHT;
    for (int i = 0; i < WIDTH; i++) {
        for (int j = 0; j < HEIGHT; j++) {
            double x = x_min + i*dx;
            double y = y_max - j*dy;
            unsigned int color = divergente(x,y);
            pixel_buffer[j*WIDTH + i] = color;
        }
    }
}
 
 */
void mandelbrotMPI(int rank, int size, unsigned int* local_pixel_buffer, int start_row, int local_height) {
    double dx = (x_max - x_min) / WIDTH;
    double dy = (y_max - y_min) / HEIGHT;
 
    for (int j = start_row; j < start_row + local_height; j++) {
        for (int i = 0; i < WIDTH; i++) {
            double x = x_min + i * dx;
            double y = y_max - j * dy;
            unsigned int color = divergente(x, y);
            local_pixel_buffer[(j - start_row) * WIDTH + i] = color;
        }
    }
}
 
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
 
    int rows_per_process = HEIGHT / size;
    int extra_rows = HEIGHT % size;
    int local_height = (rank == size - 1) ? rows_per_process + extra_rows : rows_per_process;
 
    unsigned int* local_pixel_buffer = new unsigned int[local_height * WIDTH];
    unsigned int* pixel_buffer = nullptr;
 
    if (rank == 0) {
        pixel_buffer = new unsigned int[WIDTH * HEIGHT];
    }
 
    double start_time = MPI_Wtime();
    int start_row = rank * rows_per_process;
    mandelbrotMPI(rank, size, local_pixel_buffer, start_row, local_height);
    double end_time = MPI_Wtime();
 
    int* recvcounts = new int[size];
    int* displs = new int[size];
 
    for (int i = 0; i < size; ++i) {
        recvcounts[i] = (i == size - 1) ? (rows_per_process + extra_rows) * WIDTH : rows_per_process * WIDTH;
        displs[i] = i * rows_per_process * WIDTH;
    }
 
    MPI_Gatherv(local_pixel_buffer, local_height * WIDTH, MPI_UNSIGNED, pixel_buffer, recvcounts, displs, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
 
    if (rank == 0) {
        std::cout << "Processing time: " << (end_time - start_time) << " seconds" << std::endl;
        stbi_write_png("mandelbrot.png", WIDTH, HEIGHT, 4, pixel_buffer, WIDTH * sizeof(unsigned int));
        delete[] pixel_buffer;
    }
 
    delete[] local_pixel_buffer;
    delete[] recvcounts;
    delete[] displs;
 
    MPI_Finalize();
    return 0;
}