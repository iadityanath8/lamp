#include <stdio.h>
#include <stdlib.h>
#include "./include/ndarray.h"
#include <stdbool.h>

// #define N 10

// float A[N * N] = {
//     1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
//     11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
//     21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
//     31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
//     41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
//     51, 52, 53, 54, 55, 56, 57, 58, 59, 60,
//     61, 62, 63, 64, 65, 66, 67, 68, 69, 70,
//     71, 72, 73, 74, 75, 76, 77, 78, 79, 80,
//     81, 82, 83, 84, 85, 86, 87, 88, 89, 90,
//     91, 92, 93, 94, 95, 96, 97, 98, 99, 100};

// float B[N * N] = {
//     101, 102, 103, 104, 105, 106, 107, 108, 109, 110,
//     111, 112, 113, 114, 115, 116, 117, 118, 119, 120,
//     121, 122, 123, 124, 125, 126, 127, 128, 129, 130,
//     131, 132, 133, 134, 135, 136, 137, 138, 139, 140,
//     141, 142, 143, 144, 145, 146, 147, 148, 149, 150,
//     151, 152, 153, 154, 155, 156, 157, 158, 159, 160,
//     161, 162, 163, 164, 165, 166, 167, 168, 169, 170,
//     171, 172, 173, 174, 175, 176, 177, 178, 179, 180,
//     181, 182, 183, 184, 185, 186, 187, 188, 189, 190,
//     191, 192, 193, 194, 195, 196, 197, 198, 199, 200};

// void matmul(float *A, float *B, float *C, int n)
// {
//     for (int i = 0; i < n; i++)
//     {
//         for (int j = 0; j < n; j++)
//         {
//             float sum = 0;
//             for (int k = 0; k < n; k++)
//             {
//                 sum += A[i * n + k] * B[k * n + j];
//             }
//             C[i * n + j] = sum;
//         }
//     }
// }


void naive_matmul(float* A, float* B, float* C, int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0;
            for (int p = 0; p < K; p++) {
                sum += A[i * K + p] * B[p * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

void matmul_kernel(float* A, float* B, float* C, const int M, const int N, const int K) ;

void print_matrix(float *C, int n)
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            printf("%8.2f ", C[i * n + j]);
        }
        printf("\n");
    }
}

int main()
{
    const int M = 16, N = 6, K = 8;
    float A[M * K], B[K * N], C[M * N], C_naive[M * N];

    // Initialize A and B with some values
    for (int i = 0; i < M * K; i++)
        A[i] = i;
    for (int i = 0; i < K * N; i++)
        B[i] = i;

    // Perform matrix multiplication
    matmul_kernel(A, B, C, M, N, K);
    naive_matmul(A, B, C_naive, M, N, K);

    // Compare results
    bool correct = true;
    for (int i = 0; i < M * N; i++)
    {
        if (abs(C[i] - C_naive[i]) > 1e-5)
        {
            correct = false;
            break;
        }
    }

    printf(correct ? "correct" : "wrong");
    return 0;
}
