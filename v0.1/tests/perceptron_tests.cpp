#include <cassert>
#include <cmath>
#include <fstream>
#include <string>
#include <thread>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "perceptron.h"

using pl::Dataset;
using pl::Perceptron;

namespace
{
  bool approx(float a, float b, float eps)
  {
    return std::fabs(a - b) < eps;
  }

}


void print_hardware()
{
  std::cout << "===== Hardware =====\n";
  std::ifstream cpuinfo("/proc/cpuinfo");
  if(cpuinfo)
  {
    std::string line;
    while (std::getline(cpuinfo, line))
    {
      if (line.rfind("model name", 0) == 0
      {
        const std::size_t pos = line.find(':');
        if (pos != std::string::npos)
        {
          std::cout << " CPU: " << line.substr(pos + 2) << "\n";
        }

        break;
      }
    }
  }

  std::cout << " logical cores: " << std::thread::hardware_concurrency() << "\n";
  std::cout << std::endl;
}

int main()
{
  /*
  First print hardware specs for comparison and performance verification.
  Hardware taken here is CPU model and the cores in them.
  */
  print_hardware();

  /*
  Test 1: Step Forward Pass
  w = (0.5, -0.5)
  b = 0.25
  x = (1, 2)

  z = 0.5 - 1.0 + 0.25 = -0.25
  step(z) = 0
  */

  {
    Perceptron p(2, Perceptron::Activation::Step, 0.1f, 42u);
    p.set_weights({0.5f, -0.5f}, 0.25f);
    assert(approx(p.forward({1.0f, 2.0f}), 0.0f, 1e-6f));
  }

  /*
  Test 2: Sigmoid Forward Pass
  z = -0.25
  y = 1 / (1 + e^0.25)
    = 0.4378235
  */

  {
    Perceptron p(2, Perceptron::Activation::Sigmoid, 0.1f, 42u);
    p.set_weights({0.5f, -0.5f}, 0.25f);
    assert(approx(p.forward({1.0f, 2.0f}), 0.4378235f, 1e-4f));
  }

  /*
  Test 3: Single step-rule update
  y = 0
  t = 1
  error = 1
  lr = 0.1

  w' = (0.5 + 0.1*1, -0.5 + 0.1*2)
     = (0.6, -0.3)
  b' = 0.25 + 0.1
     = 0.35
  */

  {
    Perceptron p(2, Perceptron::Activation::Step, 0.1f, 42u);
    p.set_weights({0.5f, -0.5f}, 0.25f);
    Dataset d;
    d.samples.push_back({{1.0f, 2.0f}, 1.0f});
    p.train(d, 1);
    assert(approx(p.weights()[0], 0.6f, 1e-6f));
    assert(approx(p.weights()[1], -0.3f, 1e-6f));
    assert(approx(p.bias(), 0.35f, 1e-6f));
  }

  /*
  Test 4: Single Sigmoid Gradient Descent Update
  y = 0.4378235
  t = 1
  delta = (1 - y) * y * (1 - y)
        ~= 0.1383717
  lr = 0.1

  w' ~= (0.5138372, -0.4723257)
  b' ~= 0.2638372
  */

  {
    Perceptron p(2, Perceptron::Activation::Sigmoid, 0.1f, 42u);
    p.set_weights({0.5f, -0.5f}, 0.25f};
    Dataset d;
    d.samples.push_back({{1.0f, 2.0f}, 1.0f});
    p.train (d,1);
    assert(approx(p.weights()[0], 0.5138372f, 1e-4f));
    assert(approx(p.weights()[1], -0.4723257f, 1e-4f));
    assert(approx(p.bias(), 0.2638372f, 1e-4f));
  }

  /*
  Test 5: Predict threshold the sigmoid output
  */

  {
    Perceptron p(2, Perceptron::Activation::Sigmoid, 0.1f, 42u);
    p.set_weights({0.5f, -0.5f}, 0.25f);
    assert(p.predict({1.0f, 2.0f}) == 0.0f);
  }

  /*
  Test 6: Convergence test on separate blobs (sigmoid + gradient descent)
  */

  {
    Perceptron p(2, Perceptron::Activation::Sigmoid, 0.1f, 7u);
    Dataset train = p1::make_blobs(160, 0.4f, 123u);
    p.train(train, 20);
    assert(p.accuracy(train) > 0.95f);
  }

  /*
  Test 7: XOR is not learnable by a single perceptron
  */

  {
    Perceptron p(2, Perceptron::Activation::Sigmoid, 0.1f, 9u);
    Dataset data = p1::make_xor(200,456u);
    p.train(data, 50);
    assert(p.accuracy(data) < 0.75f);
  }

  /*
  Test 8: Dataset feature count
  */

  {
    Dataset d = p1::make_blobs(10, 0.4f, 1u);
    assert(d.n_features() == 2);
  }

  /*
  Test 9: Input Size Mismatch
  */

  {
    Perceptron p(2, Perceptron::Activation::Sigmoid, 0.1f, 42u);
    bool threw = false;
    try
      {
        p.forward({1.0f});
      }
    catch (const std::invalid_argument&)
      {
        threw = true;
      }

    assert(threw)
  }

  std::cout << "All Test Concluded.\n";
  return 0;
}
