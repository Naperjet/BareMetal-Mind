#pragma once

#include <cstddef>
#include <random>
#include <stdexcept>
#include <vector>

namespace pl
{
  struct Sample
  {
    std::vector<float> x;
    float label;
  };

  class Dataset
  {
    public:
      std::vector<Sample> samples;

      std::size_t n_features() const
      {
        return samples.empty() ? 0 : samples.front().x.size();
      }
  };

  Dataset make_blobs(std::size_t n_per_class, float sigma, unsigned seed);
  Dataset make_xor(std::size_t n_per_class, unsigned seed);

  class Perceptron
  {
    public:
      enum class Activation { Step, Sigmoid };

      Perceptron(std::size_t n_inputs, Activation activation = Activation::sigmoid,
                  float learning_rate = 0.1f, unsigned seed = std::random_device{}());

      float forward(const std::vector<float>& x) const;
      float predict(const std::vector<float>& x) const;
      void train(Dataset& data, std::size_t epochs);

      float loss(const Dataset& data) const;
      float accuracy(const Dataset& data) const;

      void set_weights(const std::vector<float>& w, float b);
      const std::vector<float>& weights() const
      {
        return w_;
      }
      float bias() const
      {
        return b_;
      }

    private:
      void update(const Sample& s, float y);

      std::vector<float> w_;
      float b_ = 0.0f;
      Activation activation_;
      float lr_;
      std::mt19937 rng_;
  };

}
