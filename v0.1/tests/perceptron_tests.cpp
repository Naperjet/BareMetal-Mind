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
}
