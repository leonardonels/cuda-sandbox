_temp = 20.0f; // Ambient room temperature in Celsius

    // Initial State: Vector of temperatures for 3 different cups
    // Cup 1: 42°C, Cup 2: 24°C, Cup 3: 50°C
    std::vector<float> h_temp{42, 24, 50}; 
    thrust::universal_vector<float> temp = h_temp; 

    // Lambda function to calculate the new temperature for a single cup.
    // [=] captures all local variables (like 'k' and 'ambient_temp') by value so they can be used inside.
    auto op = [=] __host__ __device__ (float temp) { // <- auto op = [=](float temp) {
        // Newton's Law of Cooling formula: T_new = T_old + k * (T_env - T_old)
        return temp + k * (ambient_temp - temp);
    };

    // Print initial state (Step 0)
    std::cout << "Step 0: ";
    for (const auto& t : temp) {
        std::cout << t << " ";
    }
    std::cout << std::endl;

    // Run simulation for 2 steps (Step 1 and Step 2)
    for (int step = 1; step < 3; ++step) {
        // Apply the cooling formula (op) to every element in the 'temp' vector
        // and store the results back into the 'temp' vector in-place.
        thrust::transform(thrust::device, temp.begin(), temp.end(), temp.begin(), op); // <- std::transform(temp.begin(), temp.end(), temp.begin(), op);
        
        // Print current state
        std::cout << "Step " << step << ": ";
        for (const auto& t : temp) {
            std::cout << t << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}