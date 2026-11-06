#include "perceptron.h"

#include <random>

namespace pl
{
  Dataset make_blobs(std::size_t n_per_class, float sigma, unsigned seed)
  {
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, sigma);
    const float centers[2][2] = {{-2.0f, -2.0f}, {2.0f,2.0f}};

    Dataset d;
    d.sample.reserve(2 * n_per_class);
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
    d.sample.reserve(4 * n_per_class);
    for (std::size_t i = 0; i < 2 * n_per_class; ++i) 
    {
      const float x1 = dist(rng);
      const float x2 = dist(rng)
      const float label = x1 * x2 > 0.0f ? 1.0f : 0.0f;
      d.samples.push_back({{x1, x2}, label});
    }

    return d;
  }
}
