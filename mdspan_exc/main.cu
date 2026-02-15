#include <thrust/universal_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/tabulate.h>
#include <cuda/std/pair>
#include <cuda/std/mdspan>
#include <iostream>

__host__ __device__
cuda::std::pair<int, int> row_col(int id, int width) {
    return cuda::std::make_pair(id / width, id % width);
}

void simulate(int height, int width,
              const thrust::universal_vector<float> &in,
                    thrust::universal_vector<float> &out)
{
  // TODO: Modify the following code to use `cuda::std::mdspan`
  const float *in_ptr = thrust::raw_pointer_cast(in.data());
  
  // let's use the mdspan function
  cuda::std::mdspan temp_in(in_ptr, height, width);

  thrust::tabulate(
    thrust::device, out.begin(), out.end(),
    [temp_in, height, width] __host__ __device__(int id) {
      auto [row, column] = row_col(id, width);

      if (row > 0 && column > 0 && row < height - 1 && column < width - 1) {
        float d2tdx2 = temp_in(row, column - 1) - 2 * temp_in(row, column) + temp_in(row, column + 1);
        float d2tdy2 = temp_in(row - 1, column) - 2 * temp_in(row, column) + temp_in(row + 1, column);

        return temp_in(row, column) + 0.2f * (d2tdx2 + d2tdy2);
      } else {
        return temp_in(row, column);
      }
    });
}