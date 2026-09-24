// Bounded native publication for Qwen routed layers whose experts are resident.

#include "all_hit.h"

#include <Python.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <thread>
#include <time.h>

#include "gate.h"

namespace nb = nanobind;

namespace moespresso_gate::all_hit {

namespace {

constexpr std::size_t kPools = 3;
constexpr std::size_t kExperts = 512;
constexpr std::size_t kActive = 10;
constexpr std::size_t kRingWords = 18;
constexpr std::uint32_t kMissing = 512;
constexpr std::uint64_t kMaxSliceNs = 1000000;
// Delayed polls retain a bounded backoff. The first slice preserves the
// zero-delay yield cadence; every slice checks readiness before yielding.
constexpr std::uint64_t kInitialBackoffNs = 2000;
constexpr std::uint64_t kMaxBackoffNs = 50000;

struct Buffer {
  Py_buffer view{};
  bool acquired = false;

  Buffer() = default;
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;

  Buffer(Buffer&& other) noexcept : view(other.view), acquired(other.acquired) {
    other.view = {};
    other.acquired = false;
  }

  Buffer& operator=(Buffer&& other) noexcept {
    if (this != &other) {
      release();
      view = other.view;
      acquired = other.acquired;
      other.view = {};
      other.acquired = false;
    }
    return *this;
  }

  ~Buffer() {
    release();
  }

  void release() noexcept {
    if (acquired) {
      PyBuffer_Release(&view);
      view = {};
      acquired = false;
    }
  }

  void acquire(PyObject* object, bool writable) {
    int flags = PyBUF_C_CONTIGUOUS | PyBUF_FORMAT | PyBUF_ND | PyBUF_STRIDES;
    if (writable) {
      flags |= PyBUF_WRITABLE;
    }
    if (PyObject_GetBuffer(object, &view, flags) < 0) {
      throw nb::python_error();
    }
    acquired = true;
  }
};

struct Span {
  std::uintptr_t begin;
  std::uintptr_t end;
};

[[noreturn]] void invalid(const char* message) {
  throw nb::value_error(message);
}

std::uint64_t exact_integer(nb::handle object, const char* message) {
  if (!PyLong_CheckExact(object.ptr())) {
    invalid(message);
  }
  const auto value = PyLong_AsUnsignedLongLong(object.ptr());
  if (PyErr_Occurred()) {
    throw nb::python_error();
  }
  return value;
}

bool native_uint32(const Py_buffer& view) {
  return view.itemsize == 4 && view.format != nullptr && std::strcmp(view.format, "I") == 0;
}

Span checked_span(const Py_buffer& view) {
  const auto begin = reinterpret_cast<std::uintptr_t>(view.buf);
  if (view.buf == nullptr || view.len <= 0 ||
      static_cast<std::uintptr_t>(view.len) >
          std::numeric_limits<std::uintptr_t>::max() - begin) {
    invalid("native all-hit buffer address range differs");
  }
  return {begin, begin + static_cast<std::uintptr_t>(view.len)};
}

bool overlaps(const Span& left, const Span& right) {
  return left.begin < right.end && right.begin < left.end;
}

void require_ring(const Py_buffer& view) {
  const auto address = reinterpret_cast<std::uintptr_t>(view.buf);
  if (view.readonly != 1 || view.ndim != 1 || view.shape == nullptr ||
      view.shape[0] != static_cast<Py_ssize_t>(kRingWords) ||
      view.len != static_cast<Py_ssize_t>(kRingWords * sizeof(std::uint32_t)) ||
      !native_uint32(view) || !PyBuffer_IsContiguous(&view, 'C') ||
      address % alignof(std::uint32_t) != 0) {
    invalid("native all-hit ring must be exact read-only aligned uint32[18]");
  }
}

void require_maps(const Py_buffer& view) {
  const auto address = reinterpret_cast<std::uintptr_t>(view.buf);
  if (view.readonly != 1 || view.ndim != 2 || view.shape == nullptr ||
      view.shape[0] != static_cast<Py_ssize_t>(kPools) ||
      view.shape[1] != static_cast<Py_ssize_t>(kExperts) ||
      view.len != static_cast<Py_ssize_t>(kPools * kExperts * sizeof(std::uint32_t)) ||
      !native_uint32(view) || !PyBuffer_IsContiguous(&view, 'C') ||
      address % alignof(std::uint32_t) != 0) {
    invalid("native all-hit maps must be exact read-only aligned uint32[3,512]");
  }
}

void require_destination(const Py_buffer& view) {
  const auto address = reinterpret_cast<std::uintptr_t>(view.buf);
  if (view.readonly != 0 || view.ndim != 1 || view.shape == nullptr ||
      view.shape[0] != static_cast<Py_ssize_t>(kActive) ||
      view.len != static_cast<Py_ssize_t>(kActive * sizeof(std::uint32_t)) ||
      !native_uint32(view) || !PyBuffer_IsContiguous(&view, 'C') ||
      address % alignof(std::uint32_t) != 0) {
    invalid("native all-hit destination must be exact writable aligned uint32[10]");
  }
}

struct Prepared {
  Buffer ring;
  Buffer maps;
  std::array<Buffer, kPools> destinations;
  std::array<std::uint32_t, kPools> capacities{};
  std::uint32_t sequence = 0;

  void close() noexcept {
    for (auto& destination : destinations) {
      destination.release();
    }
    maps.release();
    ring.release();
  }
};

Prepared prepare(
    nb::handle ring_object,
    nb::handle maps_object,
    nb::handle destination_objects,
    nb::handle sequence_object,
    nb::handle capacity_objects) {
  Prepared value;
  const auto sequence = exact_integer(
      sequence_object, "native all-hit sequence must be an exact integer");
  if (sequence == 0 || sequence > std::numeric_limits<std::uint32_t>::max()) {
    invalid("native all-hit sequence is out of bounds");
  }
  if (!PyTuple_CheckExact(capacity_objects.ptr()) ||
      PyTuple_Size(capacity_objects.ptr()) != static_cast<Py_ssize_t>(kPools)) {
    invalid("native all-hit capacities must be an exact three-item tuple");
  }
  for (std::size_t pool = 0; pool < kPools; ++pool) {
    const auto capacity = exact_integer(
        nb::handle(PyTuple_GetItem(capacity_objects.ptr(), pool)),
        "native all-hit capacity must be an exact integer");
    if (capacity == 0 || capacity > kExperts) {
      invalid("native all-hit capacity is out of bounds");
    }
    value.capacities[pool] = static_cast<std::uint32_t>(capacity);
  }
  if (!PyTuple_CheckExact(destination_objects.ptr()) ||
      PyTuple_Size(destination_objects.ptr()) != static_cast<Py_ssize_t>(kPools)) {
    invalid("native all-hit destinations must be an exact three-item tuple");
  }

  value.ring.acquire(ring_object.ptr(), false);
  require_ring(value.ring.view);
  value.maps.acquire(maps_object.ptr(), false);
  require_maps(value.maps.view);
  for (std::size_t pool = 0; pool < kPools; ++pool) {
    value.destinations[pool].acquire(PyTuple_GetItem(destination_objects.ptr(), pool), true);
    require_destination(value.destinations[pool].view);
  }

  std::array<Span, 2 + kPools> spans{};
  spans[0] = checked_span(value.ring.view);
  spans[1] = checked_span(value.maps.view);
  for (std::size_t pool = 0; pool < kPools; ++pool) {
    spans[2 + pool] = checked_span(value.destinations[pool].view);
  }
  for (std::size_t left = 0; left < spans.size(); ++left) {
    for (std::size_t right = left + 1; right < spans.size(); ++right) {
      if (overlaps(spans[left], spans[right])) {
        invalid("native all-hit buffers must not alias");
      }
    }
  }

  const auto* maps = static_cast<const std::uint32_t*>(value.maps.view.buf);
  for (std::size_t pool = 0; pool < kPools; ++pool) {
    std::array<bool, kExperts> seen{};
    for (std::size_t expert = 0; expert < kExperts; ++expert) {
      const auto slot = maps[pool * kExperts + expert];
      if (slot == kMissing) {
        continue;
      }
      if (slot >= value.capacities[pool] || seen[slot]) {
        invalid("native all-hit map values or slot uniqueness differ");
      }
      seen[slot] = true;
    }
  }

  const auto signaled = moespresso_gate::signaled_value();
  if (signaled >= sequence) {
    invalid("native all-hit release frontier must precede the requested sequence");
  }
  value.sequence = static_cast<std::uint32_t>(sequence);
  return value;
}

enum class Outcome {
  Pending,
  Miss,
  Published,
  Future,
  InvalidId,
  ReleaseAdvanced,
  SignalError,
};

std::uint32_t checksum(
    const std::uint32_t* ids,
    std::size_t count,
    std::uint32_t sequence) {
  std::uint32_t value = 2166136261u;
  for (std::size_t index = 0; index < count; ++index) {
    value = (value ^ ids[index]) * 16777619u;
  }
  return (value ^ sequence) * 16777619u;
}

Outcome run(
    Prepared& value,
    std::uint64_t slice_ns,
    bool backoff_enabled,
    std::uint64_t& backoff_ns) {
  const auto* ring = static_cast<const volatile std::uint32_t*>(value.ring.view.buf);
  const auto* maps = static_cast<const std::uint32_t*>(value.maps.view.buf);
  std::array<std::uint32_t, kActive> ids{};
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::nanoseconds(slice_ns);
  do {
    const auto first = ring[0];
    if (first > value.sequence) {
      return Outcome::Future;
    }
    if (first == value.sequence) {
      for (std::size_t index = 0; index < kActive; ++index) {
        ids[index] = ring[8 + index];
      }
      const auto observed_checksum = ring[1];
      const auto last = ring[0];
      if (first == last &&
          observed_checksum == checksum(ids.data(), ids.size(), value.sequence)) {
        for (const auto expert : ids) {
          if (expert >= kExperts) {
            return Outcome::InvalidId;
          }
        }
        std::array<std::array<std::uint32_t, kActive>, kPools> slots{};
        for (std::size_t expert_index = 0; expert_index < kActive; ++expert_index) {
          const auto expert = ids[expert_index];
          for (std::size_t pool = 0; pool < kPools; ++pool) {
            const auto slot = maps[pool * kExperts + expert];
            if (slot == kMissing) {
              return Outcome::Miss;
            }
            slots[pool][expert_index] = slot;
          }
        }
        try {
          if (moespresso_gate::signaled_value() >= value.sequence) {
            return Outcome::ReleaseAdvanced;
          }
          for (std::size_t pool = 0; pool < kPools; ++pool) {
            std::memcpy(
                value.destinations[pool].view.buf,
                slots[pool].data(),
                kActive * sizeof(std::uint32_t));
          }
          moespresso_gate::signal_event(value.sequence);
        } catch (...) {
          return Outcome::SignalError;
        }
        return Outcome::Published;
      }
    }
    const auto now = std::chrono::steady_clock::now();
    if (now >= deadline) {
      return Outcome::Pending;
    }
    if (!backoff_enabled) {
      const timespec yield_time{0, 0};
      nanosleep(&yield_time, nullptr);
      continue;
    }
    const auto remaining = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(deadline - now).count());
    const auto sleep_ns = std::min(backoff_ns, remaining);
    const timespec yield_time{
        static_cast<time_t>(sleep_ns / 1000000000),
        static_cast<long>(sleep_ns % 1000000000)};
    nanosleep(&yield_time, nullptr);
    backoff_ns = std::min(backoff_ns * 2, kMaxBackoffNs);
  } while (true);
}

nb::str finish(Outcome outcome) {
  switch (outcome) {
    case Outcome::Pending:
      return nb::str("PENDING");
    case Outcome::Miss:
      return nb::str("MISS");
    case Outcome::Published:
      return nb::str("PUBLISHED");
    case Outcome::Future:
      throw std::runtime_error("native all-hit ring advanced beyond the requested sequence");
    case Outcome::InvalidId:
      invalid("native all-hit ring contains an out-of-range expert ID");
    case Outcome::ReleaseAdvanced:
      throw std::runtime_error("native all-hit release frontier advanced during publication");
    case Outcome::SignalError:
      throw std::runtime_error("native all-hit release signal failed after slot publication");
  }
  throw std::runtime_error("native all-hit returned an unknown outcome");
}

} // namespace

nb::object read_exported_ids(
    nb::handle ring_object,
    nb::handle sequence_object,
    nb::handle count_object,
    nb::handle wait_ns_object) {
  const auto sequence = exact_integer(
      sequence_object, "native ring sequence must be an exact integer");
  const auto count = exact_integer(
      count_object, "native ring count must be an exact integer");
  const auto wait_ns = exact_integer(
      wait_ns_object, "native ring wait must be an exact integer");
  if (sequence == 0 || sequence > std::numeric_limits<std::uint32_t>::max()) {
    invalid("native ring sequence is out of bounds");
  }
  if (count == 0 || count > 64) {
    invalid("native ring count must be between 1 and 64");
  }
  if (wait_ns > kMaxSliceNs) {
    invalid("native ring wait exceeds one millisecond");
  }
  Buffer buffer;
  buffer.acquire(ring_object.ptr(), false);
  const auto& view = buffer.view;
  checked_span(view);
  if (view.readonly != 1 || view.ndim != 1 || view.shape == nullptr ||
      view.shape[0] != static_cast<Py_ssize_t>(8 + count) ||
      view.len != static_cast<Py_ssize_t>((8 + count) * sizeof(std::uint32_t)) ||
      !native_uint32(view) || !PyBuffer_IsContiguous(&view, 'C') ||
      reinterpret_cast<std::uintptr_t>(view.buf) % alignof(std::uint32_t) != 0) {
    invalid("native ring must be exact read-only aligned uint32[8 + count]");
  }
  std::array<std::uint32_t, 64> ids{};
  bool ready = false;
  bool future = false;
  {
    nb::gil_scoped_release release;
    const auto* ring = static_cast<const volatile std::uint32_t*>(view.buf);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::nanoseconds(wait_ns);
    std::uint64_t backoff_ns = kInitialBackoffNs;
    do {
      const auto first = ring[0];
      if (first > sequence) {
        future = true;
        break;
      }
      if (first == sequence) {
        for (std::size_t index = 0; index < count; ++index) {
          ids[index] = ring[8 + index];
        }
        const auto observed_checksum = ring[1];
        const auto last = ring[0];
        if (first == last && observed_checksum == checksum(ids.data(), count, sequence)) {
          ready = true;
          break;
        }
      }
      const auto now = std::chrono::steady_clock::now();
      if (now >= deadline) {
        break;
      }
      const auto remaining = static_cast<std::uint64_t>(
          std::chrono::duration_cast<std::chrono::nanoseconds>(deadline - now).count());
      const timespec pause{0, static_cast<long>(std::min(backoff_ns, remaining))};
      nanosleep(&pause, nullptr);
      backoff_ns = std::min(backoff_ns * 2, kMaxBackoffNs);
    } while (true);
  }
  if (future) {
    throw std::runtime_error("native ring advanced beyond the requested sequence");
  }
  if (!ready) {
    return nb::none();
  }
  nb::list result;
  for (std::size_t index = 0; index < count; ++index) {
    result.append(nb::int_(ids[index]));
  }
  return result;
}

struct AllHitAttempt::Impl {
  Impl(
      nb::handle ring,
      nb::handle maps,
      nb::handle destinations,
      nb::handle sequence,
      nb::handle capacities)
      : prepared(prepare(ring, maps, destinations, sequence, capacities)),
        owner(std::this_thread::get_id()) {}

  Prepared prepared;
  std::thread::id owner;
  std::uint64_t backoff_ns = kInitialBackoffNs;
  bool backoff_enabled = false;
  bool active = false;
  bool terminal = false;
  bool closed = false;
};

AllHitAttempt::AllHitAttempt(
    nb::handle ring,
    nb::handle maps,
    nb::handle destinations,
    nb::handle sequence,
    nb::handle capacities)
    : impl_(std::make_unique<Impl>(ring, maps, destinations, sequence, capacities)) {}

AllHitAttempt::~AllHitAttempt() {
  if (impl_ != nullptr) {
    impl_->prepared.close();
  }
}

nb::str AllHitAttempt::poll(nb::handle slice_ns_object) {
  if (impl_->active) {
    throw std::runtime_error("native all-hit attempt poll is already active");
  }
  if (impl_->owner != std::this_thread::get_id()) {
    throw std::runtime_error("native all-hit attempt is owned by another thread");
  }
  if (impl_->closed) {
    throw std::runtime_error("native all-hit attempt is closed");
  }
  if (impl_->terminal) {
    throw std::runtime_error("native all-hit attempt is terminal");
  }
  const auto slice_ns = exact_integer(
      slice_ns_object, "native all-hit slice must be an exact integer");
  if (slice_ns > kMaxSliceNs) {
    invalid("native all-hit polling slice is out of bounds");
  }

  impl_->active = true;
  Outcome outcome;
  try {
    {
      nb::gil_scoped_release release;
      outcome = run(impl_->prepared, slice_ns, impl_->backoff_enabled, impl_->backoff_ns);
    }
  } catch (...) {
    impl_->active = false;
    impl_->terminal = true;
    throw;
  }
  impl_->active = false;
  if (outcome == Outcome::Pending && slice_ns > 0) {
    impl_->backoff_enabled = true;
  }
  if (outcome != Outcome::Pending) {
    impl_->terminal = true;
  }
  return finish(outcome);
}

void AllHitAttempt::close() {
  if (impl_->active) {
    throw std::runtime_error("native all-hit attempt cannot close during an active poll");
  }
  if (impl_->closed) {
    return;
  }
  if (impl_->owner != std::this_thread::get_id()) {
    throw std::runtime_error("native all-hit attempt is owned by another thread");
  }
  impl_->prepared.close();
  impl_->closed = true;
}

} // namespace moespresso_gate::all_hit
