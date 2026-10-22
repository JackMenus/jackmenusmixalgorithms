// 单生产者单消费者无锁环形队列（C++17，header-only）
// 与 latency_routing/infra/ring_buffer.py 逻辑一一对应。
//
// 设计：
//  - 容量为 2 的幂，下标用 & mask；
//  - 读写指针单调递增，各自独占 cache line，避免伪共享；
//  - 双方各缓存一份对方指针，只有在看似满/空时才做 acquire 读，减少跨核流量；
//  - 只用 acquire/release 内存序，x86 上编译为普通 mov，无 lock 前缀指令。
//
// 用法：行情线程 push 解析后的报文，策略线程 pop；生产者与消费者应绑核（isolcpus + taskset），
// 消费者忙轮询（busy-poll）而不是阻塞等待。
#pragma once

#include <atomic>
#include <cstddef>
#include <type_traits>
#include <utility>

namespace lr {

// 固定为 64：std::hardware_destructive_interference_size 随编译参数变化，不适合做 ABI 常量。
// Apple M 系列等 128B 缓存行平台请改为 128。
inline constexpr std::size_t kCacheLine = 64;

template <typename T, std::size_t Capacity>
class SpscRing {
  static_assert(Capacity >= 2 && (Capacity & (Capacity - 1)) == 0, "Capacity must be power of 2");
  static_assert(std::is_nothrow_move_assignable_v<T>, "T must be nothrow move assignable");

 public:
  bool try_push(T v) noexcept {
    const std::size_t w = write_.load(std::memory_order_relaxed);
    if (w - cached_read_ == Capacity) {
      cached_read_ = read_.load(std::memory_order_acquire);
      if (w - cached_read_ == Capacity) return false;
    }
    buf_[w & kMask] = std::move(v);
    write_.store(w + 1, std::memory_order_release);
    return true;
  }

  bool try_pop(T& out) noexcept {
    const std::size_t r = read_.load(std::memory_order_relaxed);
    if (r == cached_write_) {
      cached_write_ = write_.load(std::memory_order_acquire);
      if (r == cached_write_) return false;
    }
    out = std::move(buf_[r & kMask]);
    read_.store(r + 1, std::memory_order_release);
    return true;
  }

  std::size_t size_approx() const noexcept {
    return write_.load(std::memory_order_acquire) - read_.load(std::memory_order_acquire);
  }

 private:
  static constexpr std::size_t kMask = Capacity - 1;

  alignas(kCacheLine) std::atomic<std::size_t> write_{0};
  alignas(kCacheLine) std::size_t cached_read_{0};   // 生产者私有
  alignas(kCacheLine) std::atomic<std::size_t> read_{0};
  alignas(kCacheLine) std::size_t cached_write_{0};  // 消费者私有
  alignas(kCacheLine) T buf_[Capacity];
};

}  // namespace lr
