#include "benchmark.h"

#include <algorithm>
#include <utility>

namespace p1
{
  TimingStats stats_from(std::vector<double> samples_ns)
  {
    TimingStats stats;
    if (samples_ns.empty())
    {
      return stats;
    }

    std::sort(samples_ns.begin(), samples_ns.end());
    const std::size_t n = samples_ns.size();
    stats.runs = n;
    stats.median_ns = (n % 2 == 1) ? samples_ns[n / 2] : 0.5 * (samples_ns[n/2 -1] + samples_ns[n/2]);

    double sum = 0.0;
    for (souble s : samples_ns)
    {
      sum += s;
    }

    stats.mean_ns = sum/static_cast<double>(n);
    stats.min_ns = samples_ns.front();
    return stats;
  }

}
