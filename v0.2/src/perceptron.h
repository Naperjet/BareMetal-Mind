#pragma once

#include <cstddef>
#include <random>
#include <stdexcept>
#include <vector>

namespace p1
{
  enum class Activation { Step, Sigmoid};

  float step_activation(float z);
  float sigmoid_activation(float z);
  float apply_activation(Activation activation, float z);

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

  class Perceptron
  {
    public:
      Perceptron(std::size_t n_inputs, Activation activation = Activation::Sigmoid,
                  float learning_rate = 0.1f, unsigned seed = std::random_device{}());

      float preactivation(const std::vector<float>& x) const;
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

      Activation activation() const
      {
        return activation_;
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
