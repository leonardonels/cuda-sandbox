Welcome to my CUDA Sandbox! This repository serves as a playground for experimenting with CUDA C++ and the Thrust library.

For comprehensive documentation on the algorithms and data structures, refer to the [nvidia thrust api](https://nvidia.github.io/cccl/thrust/api_docs/algorithms.html).

# personal notes

- [std::transform](#stdtransform)
- [thrust::transform](#thrusttransform)
- [counting_iterator](#counting_iterator)
- [transform_iterator](#transform_iterator)
- [zip_iterator](#zip_iterator)
- [transform_iterator + zip_iterator](#transform_iterator--zip_iterator)
- [transform_output_iterator](#transform_output_iterator)
- [Notes from Theory](#notes-from-theory)
- [cub vs thrust](#cub-vs-thrust)
- [Nsight Systems](#nsight-systems)
- [cudaStream](#cudastream)
- [pinned memory](#pinned-memory)

## std::transform
```cpp
std::vector<float> temp{42, 24, 50};

auto op = [=](float temp){
    float diff = ambient_temp - temp;
    return temp + k * diff;
};

std::transform( temp.begin(), temp.end(),   // input
                temp.begin(),               // output
                op);                        // lambda function

for(int i = 0; i <temp.size(); i++){
    temp[i] = op(temp[i]);
}
```

## thrust::transform
```cpp
thrust::universal_vector<float> temp{42, 24, 50};

auto op = [=] __host__ __device__ (float temp){
    float diff = ambient_temp - temp;
    return temp + k * diff;
};

thrust::transform(  thrust::device,             // where to perform the computation: device -> GPU host -> CPU
                    temp.begin(), temp.end(),   // input
                    temp.begin(),               // output
                    op);                        // lambda function

for(int i = 0; i <temp.size(); i++){
    temp[i] = op(temp[i]);
}
```

## counting_iterator
```cpp
struct counting_iterator
{
    int operator[](int i)
    {
        return i;
    }
};
```

## transform_iterator
```cpp
struct transform_iterator
{
    int *a;

    int operator[](int i)
    {
        return a[i] * 2;
    }
};
```

## zip_iterator
```cpp
struct zip_iterator
{
    int *a;
    int *b;

    std::tuple<int, int> operator[](int i)
    {
        return {a[i], b[i]};
    }
};
```

## transform_iterator + zip_iterator
```cpp
struct transform_iterator
{
    zip_iterator zip;
    int operator[](int i)
    {
        auto [a, b] = zip[i];
        return abs(a - b);
    }
};
```

## transform_output_iterator
```cpp
struct wrapper{
    int *ptr;

    void operator=(int value) { *ptr = value / 2; };
};

struct transform_output_iterator{
    int *a;

    wrapper operator[](int i){return {a + i}; }
};

std::array<int, 3> a{0, 1, 2};
transform_output_iterator it{a.data()};

it[0] = 10;
it[1] = 20;

std::printf("a[0]: %d\n", a[0]);    // prints 5
std::printf("a[1]: %d\n", a[1]);    // prints 10
```

## notes from theory
![Thrust policy](src/image-1.png)

![Always choose the best tool for the job](src/image.png)
The reason CPU latency is lower—despite the physical proximity of GPU memory—comes down to how their respective memory controllers and hierarchies are optimized.
The CPU is a latency-optimized processor. It is designed to minimize the time it takes to complete a single task (sequential execution). The GPU is a throughput-optimized processor, designed to maximize the total number of tasks completed per second (parallel execution).
Even though DDR4/5 are technically "slower" than GDDR6/6X/7, the CPU wins on latency for several structural reasons:
- the CPU’s L1, L2, and L3 caches are integrated directly onto the silicon.
- The CPU uses massive amounts of die area for "branch prediction" and "speculative execution." It essentially guesses what data you need next and pulls it into the cache before you even ask for it, making the effective latency feel near-zero.
- The CPU memory controller is optimized for "Random Access." When a CPU wants a byte, it wants it now. The GPU memory controller is designed to manage thousands of concurrent requests. To handle this volume, the GPU uses a "scheduler" that bundles requests together. This bundling process adds a "waiting period" (latency) to every single request, even if the bus itself is wide.
GDDR (Graphics DDR) is actually a modified version of standard DDR designed for high frequency and high power consumption at the cost of latency.
- DDR (CPU): Focuses on low CAS (Column Address Strobe) latency. It can switch between different "rows" of memory very quickly.
- GDDR (GPU): Uses a much higher "burst length." It is great at reading a long string of contiguous data (like pixels for a frame) but is relatively sluggish at jumping to a random, unrelated memory address.
In Standard C++: We write code to avoid "cache misses." Because the CPU is so fast, waiting for RAM is a death sentence for performance. We use "Data Oriented Design" to keep things in the L3 cache.
In CUDA C++: We don't try to hide latency with caches as much; we hide it with concurrency. If one "warp" (a group of threads) is waiting for a high-latency memory read from VRAM, the GPU hardware instantly switches to a different warp that is ready to calculate.

## cub vs thrust
To use Aynchronous use of CPU and GPU, to exploit cpu_time in which the cpu is waiting for the gpu to finish, we cannot use 'thrust' (for anything that we want to be asynchronous), instead we can acces the [CUB libabry](https://nvidia.github.io/cccl/cub/).

```cpp
// thrust
auto begin = std::chrono::high_resolution_clock::now();
thrust::tabulate(thrust::device, out.begin(), out.end(), compute);
auto end = std::chrono::high_resolution_clock::now();

// cub
auto begin = std::chrono::high_resolution_clock::now();
auto cell_ids = thrust::make_computing_iterator(0);
cub::DeviceTransform::transform(cell_ids, out.begin(), num_cells, compute);
auto end = std::chrono::high_resolution_clock::now();
```

<table>
<tr>
<td width="65%">
The CPU doesn't wait for the transformation to finish before executing the next instruction (regording end time).
That's why CUB time dowsn't scale with problem size.

</td>
<td width="35%">

![alt text](src/image-2.png)

</td>
</tr>
</table>

```cpp
auto begin = std::chrono::high_resolution_clock::now();
auto cell_ids = thrust::make_computing_iterator(0);
cub::DeviceTransform::transform(cell_ids, out.begin(), num_cells, compute);
cudaDeviceSynchronize();
// cudaDeviceSynchronize() will force the cpu to wait for the gpu to finish
// resultig in the same behaviour as thrust in this specific istamce
auto end = std::chrono::high_resolution_clock::now();
```
## Nsight Systems
To better visualize what's happening between cpu and gpu nvidia neveloped [Nsight Systems](https://developer.nvidia.com/nsight-systems/get-started).
```bash
!nvcc --extended-lambda -o /tmp/a.out Solutions/compute-io-overlap.cpp -x cu -arch=native # build executable
!sudo nsys profile --cuda-event-trace=false --force-overwrite true -o compute-io-overlap /tmp/a.out # run and profile executable
```

## cudaStream
```cpp
cudaStream_t copy_stream, compute_stream;

// Construct
cudaStreamCreate(&copy_stream);
cudaStreamCreate(&compute_stream);

// Synchronisation
cudaStreamSynchronize(compute_stream);
cudaStreamSynchronize(copy_stream);
// - waits until all preceding commands in the stream have completed
// - more lightweight compared to syncronizing the entire gpu

// Destruction
cudaStreamDestroy(compute_stream);
cudaStreamDestroy(copy_stream);
```
Majority of asynchronous CUDA libraries accept cudaStream_t.
The idea is that you'll likely want to overlap their API with:
- memory transfers,
- host-side compute or IO,
- or even another device-side compute!
```cpp
// CUDA Runtime
cudaStream_t stream = 0;
cudaMemcpyAsync(dst, 
                src, 
                count,  // in bytes
                kind,   // cudaMemcpyKind
                stream
                );
```

```cpp
// CUB
cudaStream_t stream = 0;
cub::DeviceTransform::Transform(input,      // IteratorIn
                                output,     // IteratorOut
                                nu_items,   // int
                                op,         // TransformOp
                                stream
                                );
```

```cpp
// cuBLAS
cudaStream_t stream = 0;
cublasLtMatmul(lightHandle,     // cubLasLtHandle_t
                computeDesc,    // cublasLtmatmulDesc_t
                *alpha,         // const void
                *A,             // const void
                                // ...
                stream
                );
```

If we need to copy data in between computations we can use `cudaStreamSynchronize()` to be sure that the next iteration wont override the data that is currently beein copied from device to host or vice versa. In this way we are going to program different blocks with checkpoint between blocks -> this is fast, but we can do faster.
Since the memory bandwidth on device is usually ~10 times faster that the memory bandwidth on the host, which is already ~3 times faster than the memory bandwidth avaiable on the pci-e bus, copies device to device and host to host are almost free (relative speaking to host to device or device to host).
Examples from [techpowerup.com](https://www.techpowerup.com/gpu-specs/):

                    |  memory bandwidth |
    PCI-E gen 5     |   32.0 GB/s       |
    DDR5@6400MT/s   |   102.0 GB/s      |
    RTX 2070s       |   448.0 GB/s      |
    RTX 5060        |   448.0 GB/s      |
    RTX 5070        |   672.0 GB/s      |
    RTX 3090Ti      |   1.01 TB/s       |
    RTX 5090        |   1.79 TB/s       |

We can introduce a buffer on device (or on the host) to copy the result of teh computation and allow the copy between device and host during the next computation.
```cpp
cudaStream_t copy_stream, compute_stream;   // Create compute and copy sreams
cudaStreamCreate(&compute_stream);
cudaStreamCreate(&copy_stream);

thrust::host_vector<float> hprev(height * width);

// we don't need Async here since device to device should be very fast anyway
thrust::copy(d_prev.begin(), d_prev.end(), d_buffer.begin());   // Synchronously copy into the staging buffer - prevent any datarace
cudaMemcpyAsync(h_temp_ptr, buffer_ptr, num_bytes, cudaMemcpyDeviceToHost, copy_stream);    // Asynchronously copy from staging buffer into host vector within the copy stream

for (int step = 0, step < steps; step++)
{
    sumulate(widt, height, dprev, dnext, compute_stream);   // Launch compute on compute stream
    dprev.swap(dnext);
}

cudaStreamSynchronoze(copy_stream); // wait for copy in the copy stream to finish before reding the data
store(write_step, height, width, hprev);

cudaStreamSynchronize(compute_stream);
```

## pinned memory
Due to paging, stuff in System memory can be moved to Disk memory unless is pinned into System memory. Fortunally the gpu can only read from pinned memory, but the System will move data between pinned memory and unpinned memory every time that need to move data to and from the gpu, transforming our cudaMemcpyAsync int a synchronous copy.
- When you "Pin" memory (using cudaMallocHost), you are telling the OS: "Lock this data down. Do not move it, and do not swap it to the disk."

By using a `thrust::universal_host_pinned_vector` we can force the data to remain into the pinned memory.
```cpp
thrust::host_vector<float> hprev(height * width);
// need to be changed into:
thrust::universal_host_pinned_vector<float> hprev(height * width);
```