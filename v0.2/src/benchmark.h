#pragma once

#include <chrono>
#include <cstddef>
#include <vector>

namespace p1
{
  struct TimingStats
  {
    double median_ns = 0.0;
    double mean_ns = 0.0;
    double min_ns = 0.0;
    std::size_t runs = 0;
  };

  TimingStats stats_from(std::vector<double> samples_ns);
  template <typename F>
  TimingStats time_host(F&& f, std::size_t warmup, std::size_t runs)
  {
    for (std::size_t i = 0; i < warmup; ++i)
    {
      f();
    }

    std::vector<double> samples;
    samples.reserve(runs);

    for (std::size_t i = 0; i < runs; ++i)
    {
      const auto t0 = std::chrono::steady_clock::now();
      f();
      const auto t1 = std::chrono::steady_clock::now();
      samples.push_back(static_cast<double>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count()));
    }

    return stats_from(std::move(samples));
  }

}
