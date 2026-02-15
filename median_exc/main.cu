// Modify code to GPU-acceleratemedian computation [nvcc --extended-lambda main.cu -o med && ./med]

#include <thrust/universal_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>
#include <vector>
// #include <algorithm>    // moving to thrust sort we do not need algorithm anymore
#include <iostream>

// first of all we need to move the algorithm to gpu std::sort -> thrust::sort
// the input vector is already a universal vecotr so we should already have a copy of the vecotr in gpu memory
float median(thrust::universal_vector<float>& vec){
    thrust::sort(thrust::device, vec.begin(), vec.end());
    return vec[vec.size() / 2];
}

int main(){

    std::vector<float> vec{42, 24, 50, 90, 2, 2, 5, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19};    // only odd number of elements to avoid ambiguity in median calculation
    std::cout << "Input vector: ";
    for (const auto& v : vec) {
        std::cout << v << " ";
    }
    std::cout << std::endl;

    // the vector is in system memory so we need to move it in gpu memory
    // device_vector is an option but universal vector was already in the excercise script so we'll stick with it
    thrust::universal_vector<float> uvec = vec;

    // last we need to change the std vec to gpu uvec
    float med = median(uvec); // <- float med = median(vec);
    std::cout << "Median: " << med << std::endl;
}