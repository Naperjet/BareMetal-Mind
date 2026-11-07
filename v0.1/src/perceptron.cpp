#include "perceptron.h"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace pl
{
  Perceptron::Perceptron(std::size_t n_inputs, Activation activation, float learning_rate,
    unsigned seed)
  : b_(0.0f), activation_(activation), lr_(learning_rate), rng_(seed) 
  {
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    w_.reserve(n_inputs);
    for (std::size_t i = 0; i < n_input; ++i)
    {
      w_.push_back(dist(rng_));
    }
  }

  float Perceptron::forward(const std::vector<float>& x) const 
  {
    if (x.size() != w_.size()) 
    {
      throw std::invalid_argument("Perceptron::forward: input size mismatchy");
    }

    float z = b_;
    for (std::size_t i = 0; i < w_.size(); ++i) 
    {
      z += w_[i] * x[i];
    }

    if (activation_ == Activation::Step) 
    {
      return z >= 0.0f ? 1.0f : 0.0f;
    }

    return 1.0f / (1.0f + std::exp(-z));
  }

  float Perceptron::update(const std::vector<float>& x) const 
  {
    if (activation_ == Activation::Step) 
    {
      return forward(x);
    }

    return forward(x) >= 0.5f ? 1.0f : 0.0f;
  }

  void Perceptron::train(Dataset& data, std::size_t epochs) 
  {
    for (std::size_t e = 0; e < epochs; ++e) 
    {
      std::shuffle(data.samples.begin(), data.samples.end(), rng_);
      for (const Sample& s : data.samples) 
      {
        update(s, forward(s.x));
      }
    }
  }

  float Perceptron::loss(const Dataset& data) const 
  {
    if (data.samples.empty()) 
    {
      return 0.0f;
    }

    float sum = 0.0f;
    for (const Sample& s : data.samples) 
    {
      const float diff = s.label - forward(s.x);
      sum += 0.5f * diff * diff;
    }

    return sum / static_cast<float>(data.samples.size());
  }

  float Perceptron::accuracy(const Dataset& data) const 
  {
    if (data.samples.empty()) 
    {
      return 0.0f;
    }

    std::size_t correct = 0;
    for (const Sample& s : data.samples) 
    {
      if (predict(s.x) == s.label) 
      {
        ++correct;
      }
    }

    return static_cast<float>(correct) / static_cast<float>(data.samples.size());
  }

  void Perceptron::set_weights(const std::vector<float>& w, float b) 
  {
    if (w.size() != w_.size()) 
    {
      throw std::invalid_argument("Perceptron::set_weights: size mismatch");
    }

    w_ = w;
    b_ = b;
  }

  Dataset make_blobs(std::size_t n_per_class, float sigma, unsigned seed) 
  {
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, sigma);
    const float centers[2][2] = {{-2.0f, -2.0f}, {2.0f, 2.0f}};

    Dataset d;
    d.samples.reserve(2 * n_per_class);
    for (int c = 0; c < 2; ++c) 
    {
      for (std::size_t i = 0; i < n_per_class; ++i) 
      {
        d.samples.push_back({{centers[c][0] + dist(rng), centers[c][1] + dist(rng)}, 
            static_cast<float>(c)});
      }
    }

    return d;
  }

  Dataset make_xor(std::size_t n_per_class, unsigned seed) 
  {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    Dataset d;
    d.samples.reserve(4 * n_per_class);
    for (std::size_t i = 0; i < 2 * n_per_class; ++i) 
    {
      const float x1 = dist(rng);
      const float x2 = dist(rng);
      const float label = x1 * x2 > 0.0f ? 1.0f : 0.0f;
      d.samples.push_back({{x1, x2}, label});
    }

    return d;
  }
}
