#pragma once

#include <iostream>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <tensorflow/c/c_api.h>

namespace NeuralRun {
    namespace NNSmall {
        void run(const std::vector<std::string>& arguments) {
            std::cout << TF_Version() << std::endl;
            std::printf("%s\n", TF_Version());
        }
    }
}
