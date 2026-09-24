// Bounded native publication for Qwen routed layers whose experts are resident.

#pragma once

#include <memory>

#include <nanobind/nanobind.h>

namespace moespresso_gate::all_hit {

inline constexpr const char* kAbi = "qwen-native-all-hit-gate-v2";

// Read a bounded exported route without publishing slots or signaling an event.
nanobind::object read_exported_ids(
    nanobind::handle ring,
    nanobind::handle sequence,
    nanobind::handle count,
    nanobind::handle wait_ns);

class AllHitAttempt {
 public:
  AllHitAttempt(
      nanobind::handle ring,
      nanobind::handle maps,
      nanobind::handle destinations,
      nanobind::handle sequence,
      nanobind::handle capacities);
  ~AllHitAttempt();

  AllHitAttempt(const AllHitAttempt&) = delete;
  AllHitAttempt& operator=(const AllHitAttempt&) = delete;

  nanobind::str poll(nanobind::handle slice_ns);
  void close();

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace moespresso_gate::all_hit
