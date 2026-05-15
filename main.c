#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <pthread.h>
#include <stdbool.h>

#include "functions.h"


int main() {
    // Обучающая выборка: 3 входа -> 1 выход
    // Правило: результат совпадает с первым числом
    double inputs[5][3] = {
        {0, 0, 0},
        {1, 1, 1},
        {1, 0, 1},
        {0, 1, 1},
        {0, 1, 0}
    };
    double target[5] = {1, 0, 0, 1, 1};

    // Инициализируем веса случайными значениями от -1 до 1
    double w_hidden[3][2];
    double w_output[2];


    for(int i = 0; i < 2; i++) {
        w_output[i] =((double)rand()/RAND_MAX) * 2 - 1;
        for(int j = 0; j < 3; j++) {
            w_hidden[j][i] = ((double)rand()/RAND_MAX) * 2 - 1;
        }
    }
    double lr = 0.1;

    // Цикл обучения
    for (int iter = 0; iter < 100000; iter++) {
        for (int p = 0; p < 5; p++) {
            // Прямой проход (Forward pass)
            double hidden_layer[2];
            for (int i = 0; i < 2; i++) {
                double activation = 0;
                for(int j = 0; j < 3; j++) {
                    activation += inputs[p][j] * w_hidden[j][i];
                }
                hidden_layer[i] = sigmoid(activation);
            }
            
            double final_activation = 0;
            for(int i = 0; i < 2; i++) {
                final_activation += hidden_layer[i] * w_output[i];
            }
            double predicted = sigmoid(final_activation);

            double error_out = target[p] - predicted;
            double delta_out = error_out * sigmoid_derivative(predicted);

            double delta_hidden[2];
            for(int i = 0; i < 2; i++) {
                delta_hidden[i] = delta_out * w_output[i] * sigmoid_derivative(hidden_layer[i]);
            }

            for(int i = 0; i < 2; i++) {
                w_output[i] += lr * delta_out * hidden_layer[i];
                for(int j = 0; j < 3; j++) {
                    w_hidden[j][i] += lr * delta_hidden[i] * inputs[p][j];
                }
            }
        }
    }

    printf("\n=== Результаты после обучения ===\n");

    for(int p = 0; p < 5; p++) {
        double hidden[2];
        for(int i = 0; i < 2; i++) {
            double act = 0;
            for(int j = 0; j < 3; j++) act += inputs[p][j] * w_hidden[j][i];
            hidden[i] = sigmoid(act);
        }
        double final = 0;
        for(int i = 0; i < 2; i++) final += hidden[i] * w_output[i];
        printf("Вход [%.0f, %.0f, %.0f] -> Ожидали: %.0f, Получили: %f\n", inputs[p][0], inputs[p][1], inputs[p][2], target[p], sigmoid(final));
    }

    printf("=== Результаты скрытого слоя ===\n");
    for(int p = 0; p < 5; p++) {
        double hidden[2];
        for(int i = 0; i < 2; i++) {
            double act = 0;
            for(int j = 0; j < 3; j++) act += inputs[p][j] * w_hidden[j][i];
            hidden[i] = sigmoid(act);
        }
        double final = 0;
        for(int i = 0; i < 2; i++) final += hidden[i] * w_output[i];
        
        // Добавили печать того, что выдал скрытый слой
        printf("Вход [%.0f, %.0f, %.0f] -> Скрытый слой: [%f, %f] -> Получили: %f\n", 
               inputs[p][0], inputs[p][1], inputs[p][2], hidden[0], hidden[1], sigmoid(final));
    }
     return 0;
}
