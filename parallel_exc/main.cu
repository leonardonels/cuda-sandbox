//

#include <thrust/reduce.h>
#include <thrust/execution_policy.h>
#include <thrust/universal_vector.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <iostream>

thrust::universal_vector<float> row_temperatures(
    int height, int width,
    thrust::universal_vector<int>& row_ids,
    thrust::universal_vector<float>& temp)
{
    thrust::universal_vector<float> sums(height);

    // TODO: Modify the line below to use counting and transform iterators to
    // generates row indices `id / width` instead
    

    // we use make_counting_iterator(0) to generate sequence 0, 1, 2...
    // instead of reading pre-computed row_ids from memory.
    auto row_ids_begin = thrust::make_transform_iterator(
        thrust::make_counting_iterator(0),
        [width] __host__ __device__(int id) {
            return id / width;
        }
    );

    auto row_ids_end = row_ids_begin + temp.size();

    thrust::reduce_by_key(thrust::device,
                          row_ids_begin, row_ids_end,
                          temp.begin(),
                          thrust::make_discard_iterator(),
                          sums.begin());

    return sums;
}