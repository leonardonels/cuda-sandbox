// nvcc --extended-lambda main.cu -o var && ./var

#include <thrust/reduce.h>
#include <thrust/execution_policy.h>
#include <thrust/universal_vector.h>
#include <thrust/transform.h>
#include <thrust/iterator/transform_iterator.h>
#include <iostream>

float variance(const thrust::universal_vector<float> &x, float mean) {
  // thrust::make_transform_iterator creates a special iterator that
  // applies a function (the lambda) to the value of the underlying
  // iterator whenever it is dereferenced. It computes values "on the fly"
  // without needing to store them in a temporary vector.
  //
  // Arguments:
  // 1. x.begin() - The "base" iterator. When looking up a value, the transform_iterator 
  //    first gets the value at this position from the base vector.
  // 2. The lambda - The function to apply to that value. 'float xi' represents
  //    the single element fetched from the base iterator (x[i]).
  //    [mean] captures external variables (like 'mean') by value, making them
  //    available inside the lambda.
  auto squared_differences = thrust::make_transform_iterator(x.begin(), [mean] __host__ __device__(float xi){
    return (xi - mean) * (xi - mean);
  });

  return thrust::reduce(thrust::device, squared_differences,
                        squared_differences + x.size()) /
         x.size();
}

float mean(thrust::universal_vector<float> vec) {
  return thrust::reduce(thrust::device, vec.begin(), vec.end()) / vec.size();
}

int main() {
  float ambient_temp = 20;
  thrust::universal_vector<float> prev{42, 24, 50};
  thrust::universal_vector<float> next{0, 0, 0};

  std::printf("step  variance\n");
  for (int step = 0; step < 3; step++) {
    thrust::transform(thrust::device, prev.begin(), prev.end(), next.begin(),
                      [=] __host__ __device__(float temp) {
                        return temp + 0.5 * (ambient_temp - temp);
                      });
    std::printf("%d     %.2f\n", step, variance(next, mean(next)));
    next.swap(prev);
  }
}