#pragma once

#include "../util.hpp"

namespace NeuralRun {
    namespace NNSmall {
        constexpr std::string FILE_NAME("models/nnsmall");
        
        constexpr int AXIS_COUNT = 2;
        constexpr int CLASS_COUNT = 2;
        constexpr float POSITIVE_CLASS_THRESHOLD = 0.5;
        constexpr float START = 0.00625;
        constexpr float STOP = 1;
        constexpr float INCREMENT = 0.0125;
        
        constexpr int WINDOW_WIDTH = 600;
        constexpr int WINDOW_HEIGHT = 600;
        constexpr int POSITIVE_CLASS_WIDTH = WINDOW_WIDTH / 2;
        constexpr int POSITIVE_CLASS_HEIGHT = WINDOW_HEIGHT / 2;
        constexpr int POSITIVE_CLASS_X = WINDOW_WIDTH - POSITIVE_CLASS_WIDTH;
        constexpr int POSITIVE_CLASS_Y = 0;
        constexpr int LABELLED_POINT_WIDTH = 4;
        constexpr int LABELLED_POINT_HEIGHT = 4;
        
        class LabelledPoint {
            public:
                LabelledPoint(float x,
                              float y,
                              float negative_class_score,
                              float positive_class_score) noexcept {
                    bool should_be_in_positive_class =
                        x >= POSITIVE_CLASS_THRESHOLD
                        && y >= POSITIVE_CLASS_THRESHOLD;
                    bool was_labelled_positive =
                        positive_class_score > negative_class_score;
                    bool was_labelled_correctly =
                        should_be_in_positive_class == was_labelled_positive;
                    
                    int rectangle_x = x * WINDOW_WIDTH - LABELLED_POINT_WIDTH / 2;
                    int rectangle_y = (1 - y) * WINDOW_HEIGHT
                                      - LABELLED_POINT_HEIGHT / 2;
                    this->rectangle = Rectangle(rectangle_x,
                                                rectangle_y,
                                                LABELLED_POINT_WIDTH,
                                                LABELLED_POINT_HEIGHT);
                    this->colour = was_labelled_correctly ? Sprite::GREEN
                                                          : Sprite::RED;
                }
                
                const Rectangle& get_rectangle() const noexcept {
                    return this->rectangle;
                }
                
                Sprite::Colour get_colour() const noexcept {
                    return this->colour;
                }
                
            private:
                Rectangle rectangle;
                Sprite::Colour colour;
        };
        
        void present(const std::vector<LabelledPoint>& labelled_points) noexcept {
            System::initialise();
            std::cout << '\n' << System::info() << '\n' << std::endl;
            
            Display display(WINDOW_WIDTH, WINDOW_HEIGHT);
            Rectangle positive_class_rectangle(POSITIVE_CLASS_X,
                                               POSITIVE_CLASS_Y,
                                               POSITIVE_CLASS_WIDTH,
                                               POSITIVE_CLASS_HEIGHT);
            display.fill(positive_class_rectangle, Sprite::WHITE);
            for (const LabelledPoint& labelled_point : labelled_points) {
                const Rectangle& rectangle = labelled_point.get_rectangle();
                Sprite::Colour colour = labelled_point.get_colour();
                display.fill(rectangle, colour);
            }
            display.update();
            
            Event event;
            bool should_present = true;
            while (should_present) {
                while (should_present && event.poll()) {
                    switch (event.type()) {
                        case Event::TERMINATE:
                            should_present = false;
                            break;
                    }
                }
            }
            System::terminate();
        }
        
        void run(const std::vector<std::string>& arguments) noexcept {
            cppflow::model model(FILE_NAME);
            std::cout << std::endl;
            for (const std::string& operation : model.get_operations()) {
                std::cout << operation << std::endl;
            }
            std::cout << std::endl;
            
            std::vector<float> input_data;
            for (float i = START; i < STOP; i += INCREMENT) {
                for (float j = START; j < STOP; j += INCREMENT) {
                    input_data.push_back(i);
                    input_data.push_back(j);
                }
            }
            
            int input_count = input_data.size() / AXIS_COUNT;
            cppflow::tensor input(input_data, {input_count, AXIS_COUNT});
            cppflow::tensor output(NeuralUtil::apply_dense_model(model, input));
            std::vector<float> output_data = output.get_data<float>();
            
            std::vector<LabelledPoint> labelled_points;
            for (int i = 0; i < input_count; ++i) {
                int input_index = i * AXIS_COUNT;
                int output_index = i * CLASS_COUNT;
                float x = input_data[input_index];
                float y = input_data[input_index + 1];
                float negative_class_score = output_data[output_index];
                float positive_class_score = output_data[output_index + 1];
                int output_class = negative_class_score < positive_class_score;
                std::cout << '(' << x << ", " << y << ") -> "
                          << output_class << std::endl;
                labelled_points.push_back(LabelledPoint(x,
                                                        y,
                                                        negative_class_score,
                                                        positive_class_score));
            }
            present(labelled_points);
        }
    }
}
