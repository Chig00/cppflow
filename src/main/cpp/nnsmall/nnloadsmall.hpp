#pragma once

#include <iostream>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cppflow/cppflow.h>

namespace NeuralRun {
    namespace NNSmall {
        void run(const std::vector<std::string>& arguments) {
            std::cout << cppflow::tensor(std::vector<int>({1, 2, 3, 4, 5, 6, 7}),
                                         std::vector<int64_t>({3, 2})) << std::endl;
        }
    }
}
