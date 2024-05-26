#pragma once

#include <sstream>
#include <cppflow/cppflow.h>

namespace NeuralUtil {
    namespace Private {
        constexpr const char* DENSE_INPUT_NAME = "serving_default_dense_input";
        constexpr const char* OUTPUT_NAME = "StatefulPartitionedCall";
        const std::vector<std::string> SINGLETON_OUTPUT_NAMES({OUTPUT_NAME});
    }
    
    cppflow::tensor apply_dense_model(cppflow::model& model,
                                      const cppflow::tensor& input) {
        return model({{Private::DENSE_INPUT_NAME, input}},
                     Private::SINGLETON_OUTPUT_NAMES)[0];
    }
}
