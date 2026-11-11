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
  std::cout << "v0.1 - Single Perceptron in C++\n\n";

  Dataset train = p1::make_blobs(160, 0.4f, 123u);
  Dataset test = p1::make_blobs(40, 0.4f, 321u);

  run_blobs(Perceptron::Activation::Sigmoid, "Sigmoid + gradient descent", train, test, 20);
  run_blobs(Perceptron::Activation::Step, "Step + perceptron rule", train, test, 20);

  std::cout << "===== XOR: the failure case ===== \n";
  Dataset xor_data = p1::make_xor(200, 456u);
  Perceptron p(2, Perceptron::Activation::Sigmoid, 0.1f, 9u);
  for (std::size_t e = 1; e <= 50; ++e)
  {
    p.train(xor_data, 1);
  }

  std::cout << "  accuracy after 50 epochs: " << p.accuracy(xor_data) * 100.0f << "%/n";
  std::cout << "  XOR is not linearly separable: a single perceptron cannot learn it.\n";
  std::cout << "  This is the motivating cliff for v0.4 (full MLP with backpropagation).\n";

  return 0;
}
