#pragma once

#include "../util.hpp"

namespace NeuralRun {
    namespace NNSmall {
        constexpr std::string FILE_NAME("models/nnsmall");
        
        constexpr float START = 0;
        constexpr float STOP = 1;
        constexpr float INCREMENT = 0.333;
        
        constexpr int WINDOW_WIDTH = 400;
        constexpr int WINDOW_HEIGHT = 400;
        constexpr int POSITIVE_SET_WIDTH = WINDOW_WIDTH / 2;
        constexpr int POSITIVE_SET_HEIGHT = WINDOW_HEIGHT / 2;
        constexpr int POSITIVE_SET_X = WINDOW_WIDTH - POSITIVE_SET_WIDTH;
        constexpr int POSITIVE_SET_Y = 0;
        
        void present() {
            System::initialise();
            Display display(WINDOW_WIDTH, WINDOW_HEIGHT);
            Rectangle positive_set_rectangle(POSITIVE_SET_X,
                                             POSITIVE_SET_Y,
                                             POSITIVE_SET_WIDTH,
                                             POSITIVE_SET_HEIGHT);
            display.fill(positive_set_rectangle, Sprite::WHITE);
            display.update();
            Event event;
            bool should_present = true;
            
            while (should_present) {
                while (should_present && event.poll()) {
                    switch (event.type()) {
                        case Event::TERMINATE:
                        case Event::LEFT_CLICK:
                        case Event::KEY_PRESS:
                            should_present = false;
                            break;
                    }
                }
            }
            
            System::terminate();
        }
        
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
            
            present();
        }
    }
}
