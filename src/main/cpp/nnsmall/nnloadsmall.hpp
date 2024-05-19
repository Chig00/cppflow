#pragma once

#include <iostream>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cppflow/cppflow.h>

namespace NeuralRun {
    namespace NNSmall {
        void run(const std::vector<std::string>& arguments) {
            // Create a tensor from a list, a = [1.0, 2.0, 3.0]
            auto a = cppflow::tensor({1.0, 2.0, 3.0});
            // Create a tensor of shape 3 filled with 1.0, b = [1.0, 1.0, 1.0]
            auto b = cppflow::fill({3}, 1.0);
            std::cout << a + b << std::endl;
        }
    }
}
