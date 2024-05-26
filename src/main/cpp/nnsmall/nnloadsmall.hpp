#pragma once

#include "../util.hpp"

constexpr std::string FILE_NAME("models/nnsmall");
constexpr float START = 0;
constexpr float STOP = 1;
constexpr float INCREMENT = 0.333;

namespace NeuralRun {
    namespace NNSmall {
        void run(const std::vector<std::string>& arguments) {
            cppflow::model model(FILE_NAME);
            std::cout << std::endl;
            for (const std::string& operation : model.get_operations()) {
                std::cout << operation << std::endl;
            }
            std::cout << std::endl;
            
            for (float i = START; i < STOP; i += INCREMENT) {
                for (float j = START; j < STOP; j += INCREMENT) {
                    cppflow::tensor input(std::vector<float>({i, j}), {1, 2});
                    cppflow::tensor output(NeuralUtil::apply_dense_model(model,
                                                                         input));
                    std::vector<float> output_data = output.get_data<float>();
                    int output_class = output_data[0] < output_data[1];
                    std::cout << '(' << i << ", " << j << ") -> "
                              << output_class << std::endl;
                }
            }
        }
    }
}
