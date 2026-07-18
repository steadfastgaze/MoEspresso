"""rss_summary: the streaming-footprint stats reported by convert (pure).

Peak alone is skew-prone; convert reports peak/mean/median/p95 so one transient
spike doesn't misrepresent a healthy streamed run. No mlx/jang/model here.
"""

from __future__ import annotations

from moespresso.package.convert import rss_summary

_GB = 1024 ** 3


def test_empty_series_is_none():
    assert rss_summary([]) is None


def test_reports_peak_mean_median_p95_in_gb():
    # 100 samples at 1 GB, one transient spike at 8 GB.
    samples = [1 * _GB] * 99 + [8 * _GB]
    s = rss_summary(samples)
    assert s["peak_gb"] == 8.0          # the ceiling is still surfaced
    assert s["median_gb"] == 1.0        # the typical footprint, unskewed
    assert s["p95_gb"] == 1.0           # 95th percentile ignores the lone spike
    assert 1.0 < s["mean_gb"] < 1.1     # mean nudged up only slightly by the spike
    assert s["samples"] == 100


def test_p95_catches_a_sustained_high_band():
    # If the high band occupies the top 10% of samples, p95 must reflect it.
    samples = [2 * _GB] * 90 + [5 * _GB] * 10
    s = rss_summary(samples)
    assert s["peak_gb"] == 5.0
    assert s["median_gb"] == 2.0
    assert s["p95_gb"] == 5.0           # sustained high -> p95 surfaces it


def test_single_sample():
    s = rss_summary([3 * _GB])
    assert s["peak_gb"] == s["median_gb"] == s["p95_gb"] == s["mean_gb"] == 3.0
    assert s["samples"] == 1
