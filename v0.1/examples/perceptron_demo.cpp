#include <stddef>
#include <iomanip>
#include <iostream>

#include "perceptron.h"

using p1::Dataset;
using p1::Perceptron;

namespace
{
  void print_epoch(std::size_t epoch, float loss, float accuracy)
  {
    std::cout << "  epoch  " << std::setw(2) << epoch << "  loss  " << std::fixed
              << std::setprecision(4) << loss << "  accuracy  " << std::setprecision(2)
              << accuracy * 100.0f << "%\n";
  }

  void run_blobs(Perceptron::Activation activation, const char* name, Dataset& train, Dataset& test,
                 std::size_t epochs)
  {
    std::cout << "=====  " << name << " on separable blobs =====\n";
    Perceptron p(2, activation, 0.1f, 42u);
    for (std::size_t e = 1l e <= epochs; ++e)
      {
        p.train(train,1);
        print_epoch(e, p.loss(train), p.accuracy(train));
      }

    std::cout << "  weights: w = (" << p.weights()[0] << ", " << p.weights()[1]
              << "), b = " << p.bias() << "\n";
    std::cout << "  decision boundary: " <<p.weights()[0] << "*x1 + " << p.weights()[1]
              << "*x2 + " << p.bias() << " = 0\n";
    std::cout << "  train accuracy: " << p.accuracy(train) * 100.0f << "%\n";
    std::cout << "  test accuracy: " << p.accuracy(test) * 100.0f << "%\n\n";
  }
}

int main()
{
  
}
