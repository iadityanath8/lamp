import cupy as cp
import time

def measure_gflops(N):
    A = cp.random.rand(N, N, dtype=cp.float32)
    B = cp.random.rand(N, N, dtype=cp.float32)

    cp.cuda.Device(0).synchronize()

    start = time.time()
    C = cp.matmul(A, B)
    cp.cuda.Device(0).synchronize()
    end = time.time()

    total_ops = 2 * N**3  # Matrix multiplication: 2 * N^3 floating-point operations
    time_taken = end - start
    gflops = (total_ops / time_taken) / 1e9  # Convert to GFLOPS

    print(f"Matrix Size: {N} x {N}")
    print(f"Time taken: {time_taken:.6f} sec")
    print(f"Performance: {gflops:.2f} GFLOPS")

# Example: Measure for a 4096x4096 matrix
measure_gflops(8192)
