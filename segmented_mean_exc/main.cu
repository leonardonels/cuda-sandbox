#include <thrust/reduce.h>
#include <thrust/execution_policy.h>
#include <thrust/universal_vector.h>
#include <thrust/transform.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/make_transform_output_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <iostream>

struct mean_functor {
    int width;
    __host__ __device__ float operator()(float x) const {
        return x / width;
    }
};

thrust::universal_vector<float> row_temperatures(
    int height, int width,
    thrust::universal_vector<int>& row_ids,
    thrust::universal_vector<float>& temp)
{
    thrust::universal_vector<float> means(height);

    // TODO: Replace `means.begin()` by a `transform_output_iterator` using
    // the provided `mean_functor` functor
    auto means_output = thrust::make_transform_output_iterator(
        means.begin(),
        mean_functor{}
    );

    auto row_ids_begin = thrust::make_transform_iterator(
        thrust::make_counting_iterator(0),
        [=]__host__ __device__(int i) {
            return i / width;
        });
    auto row_ids_end = row_ids_begin + temp.size();

    thrust::reduce_by_key(thrust::device,
                          row_ids_begin, row_ids_end,   // key input
                          temp.begin(),                 // value input
                          thrust::make_discard_iterator(),  // key output (discarded)
                          means_output);                // value output

    auto transform_op = mean_functor{width};

    // TODO: remove this `transform` call after adding the
    // `transform_output_iterator`
    thrust::transform(thrust::device,
                      means.begin(),
                      means.end(),
                      means.begin(),
                      transform_op);

    return means;
}