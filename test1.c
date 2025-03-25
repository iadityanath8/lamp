#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <time.h>

// #define N 1024

// Function to initialize a matrix with random values
void initializeMatrix(float *matrix, int size)
{
    for (int i = 0; i < size; i++)
    {
        matrix[i] = 2 + i; // Random value between 0 and 1
    }
}

void print_mat(float *mat, int ROW, int COL)
{
    for (size_t i = 0; i < ROW; i++)
    {
        for (size_t j = 0; j < COL; j++)
        {
            printf("%f ", mat[i * COL + j]);
        }
        printf("\n");
    }
}
//  AR     BCol    ACol
void naive_mul(float *A, float *B, float *C, int M, int N, int K)
{
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            float sum = 0.0f;
            for (int k = 0; k < K; k++)
            {
                float a = A[i * K + k];
                float b = B[k * N + j];
                sum += a * b;
            }
            C[i * N + j] = sum;
        }
    }
}

int cublas(float *h_A, float *h_B, float *h_C, int M, int N, int K)
{
    cublasHandle_t handle;
    cublasStatus_t status;

    status = cublasCreate(&handle);
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        fprintf(stderr, "CUBLAS initialization failed\n");
        return EXIT_FAILURE;
    }

    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    float *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, size_A);
    cudaMalloc((void **)&d_B, size_B);
    cudaMalloc((void **)&d_C, size_C);

    cublasSetMatrix(M, K, sizeof(float), h_A, M, d_A, M);
    cublasSetMatrix(K, N, sizeof(float), h_B, K, d_B, K);

    float alpha = 1.0f;
    float beta = 0.0f;

    // cudaEvent_t start, stop;
    // cudaEventCreate(&start);
    // cudaEventCreate(&stop);

    // cudaEventRecord(start, 0);

    // Perform matrix multiplication: C = A * B
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                N, M, K,
                &alpha, d_B, N,
                d_A, K,
                &beta, d_C, N);

    // cudaEventRecord(stop, 0);
    // cudaEventSynchronize(stop);

    // float milliseconds = 0;
    // cudaEventElapsedTime(&milliseconds, start, stop);

    // double flops = 2.0 * (double)M * (double)N * (double)K;
    // double gflops = (flops / (milliseconds / 1000.0)) / 1e9;

    // printf("Matrix multiplication (%d x %d) took %.3f ms\n", M, N, milliseconds);
    // printf("GFLOPS: %.3f\n", gflops);

    cublasGetMatrix(M, N, sizeof(float), d_C, M, h_C, M);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // cudaEventDestroy(start);
    // cudaEventDestroy(stop);

    cublasDestroy(handle);
    return 0;
}

int main()
{
    // const int N = 8192;
    // size_t matrixSize = N * N * sizeof(float);
    // float *h_A = (float *)malloc(matrixSize);
    // float *h_B = (float *)malloc(matrixSize);
    // float *h_C = (float *)malloc(matrixSize);

    // srand(time(NULL)); // Seed the random number generator
    // initializeMatrix(h_A, N * N);
    // initializeMatrix(h_B, N * N);

    float h_A[2 * 2] = {
        1, 2,
        3, 4};

    // // 2 x 2 | 2 x 3

    float h_B[2 * 3] = {
        5, 6, 7,
        8, 9, 10};

    float h_C[2 * 3] = {0};

    // for (int i = 0; i < 2 * 3; i++)
    //     h_C[i] = 0.0f;
    cublas(h_A, h_B,h_C,2,3,2);
    // naive_mul(h_A, h_B, h_C, 2, 3, 2);

    print_mat(h_C, 2, 3);
}