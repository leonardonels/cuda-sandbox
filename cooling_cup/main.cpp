// g++ -std=c++11 main.cpp -o app && ./app

# include <vector>
# include <algorithm>
# include <iostream>

int main() {

    // Simulation Constants
    const float k = 0.5f; // Thermal conductivity constant (rate of cooling)
    const float ambient_temp = 20.0f; // Ambient room temperature in Celsius

    // Initial State: Vector of temperatures for 3 different cups
    // Cup 1: 42°C, Cup 2: 24°C, Cup 3: 50°C
    std::vector<float> temp{42,24,50}; 

    // Lambda function to calculate the new temperature for a single cup.
    // [=] captures all local variables (like 'k' and 'ambient_temp') by value so they can be used inside.
    auto op = [=](float temp) {
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
        std::transform(temp.begin(), temp.end(), temp.begin(), op);
        
        // Print current state
        std::cout << "Step " << step << ": ";
        for (const auto& t : temp) {
            std::cout << t << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}