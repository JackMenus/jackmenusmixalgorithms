// 简单基准：两个线程通过 SpscRing 传递 N 个整数，测吞吐与每条消息的平均往返时间。
// 编译：g++ -O3 -std=c++17 -pthread spsc_bench.cpp -o spsc_bench
#include "spsc_ring.hpp"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <thread>

int main() {
  constexpr std::uint64_t N = 20'000'000;
  static lr::SpscRing<std::uint64_t, 1 << 12> q;
  std::uint64_t sum = 0;
  auto t0 = std::chrono::steady_clock::now();
  std::thread consumer([&] {
    std::uint64_t v, got = 0;
    while (got < N) {
      if (q.try_pop(v)) { sum += v; ++got; }
    }
  });
  for (std::uint64_t i = 0; i < N; ++i) {
    while (!q.try_push(i)) {}
  }
  consumer.join();
  auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count();
  std::printf("msgs=%llu  total=%.1f ms  %.2f ns/msg  %.1f Mmsg/s  checksum_ok=%d\n",
              (unsigned long long)N, ns / 1e6, (double)ns / N, N * 1e3 / ns, sum == N * (N - 1) / 2);
  return 0;
}
