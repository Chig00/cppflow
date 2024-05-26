#pragma once

#include <iostream>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cppflow/cppflow.h>

namespace NeuralRun {
    namespace NNSmall {
        void run(const std::vector<std::string>& arguments) {
            cppflow::tensor a(std::vector<int>({1, 2}), std::vector<int64_t>({1, 2}));
            cppflow::tensor b(std::vector<int>({1, 2}), std::vector<int64_t>({2, 1}));
            cppflow::tensor c(a + b);
            cppflow::tensor d(a * b);
            cppflow::tensor e(b * a);
            cppflow::tensor f(cppflow::mat_mul(a, b));
            std::cout << a << std::endl
                      << b << std::endl
                      << c << std::endl
                      << d << std::endl
                      << e << std::endl
                      << f << std::endl;
        }
    }
}
