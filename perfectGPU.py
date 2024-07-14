import os
import numpy as np
from numba import cuda
import csv

# Set the environment variable to use 4 GPUs (assuming IDs 0, 1, 2, 3)
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"


# Define the kernel function to check for perfect numbers
@cuda.jit
def find_perfect_numbers(start, end, results):
    tid = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    step = cuda.gridDim.x * cuda.blockDim.x

    for n in range(start + tid, end, step):
        sum_divisors = 0
        for i in range(1, n // 2 + 1):
            if n % i == 0:
                sum_divisors += i

        if sum_divisors == n:
            results[n - start] = n  # Store perfect number in results array


def main():
    # Detect all available GPUs after setting CUDA_VISIBLE_DEVICES
    num_devices = len(cuda.gpus)

    if num_devices == 0:
        print("No CUDA-compatible GPU found")
        return

    devices = list(range(num_devices))  # List of device IDs to use

    # Calculate maximum `max_num` based on available GPU memory
    gpu = cuda.current_context().device
    free_memory, total_memory = cuda.current_context().get_memory_info()
    available_memory = total_memory * 0.8  # Use 80% of total memory for calculations
    max_num = int(available_memory // 4)  # Since each int32 takes 4 bytes

    chunk_size = 10**5  # Define the size of each chunk to process
    threads_per_block = 256  # Threads per block
    blocks_per_grid = 128  # Blocks per grid

    print(f"Using max_num: {max_num}")

    for large_start in range(0, 10**8, max_num):  # Adjust range for a feasible demo
        large_end = min(large_start + max_num, 10**8)

        for start in range(large_start, large_end, chunk_size):
            end = min(start + chunk_size, large_end)
            results = np.zeros(end - start, dtype=np.int32)  # Array to store results for the chunk

            # Launch kernel on multiple GPUs
            for device_id in devices:
                with cuda.gpus[device_id]:
                    find_perfect_numbers[blocks_per_grid, threads_per_block](start, end, results)

            # Display and write results to a CSV file
            with open('perfect_numbers.csv', 'a') as csvfile:
                writer = csv.writer(csvfile)
                for number in results:
                    if number != 0:
                        print(f"Perfect number found: {number}")
                        writer.writerow([number])

    print("Perfect numbers found and saved to perfect_numbers.csv")


if __name__ == '__main__':
    main()
