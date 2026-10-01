"""
Unit tests for the cache fingerprint function (Fix 1.3).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_fetcher import _config_fingerprint
from unittest.mock import patch


def test_fingerprint_length():
    assert len(_config_fingerprint()) == 8


def test_fingerprint_deterministic():
    fp1 = _config_fingerprint()
    fp2 = _config_fingerprint()
    assert fp1 == fp2, "Same config must yield same fingerprint"


def test_fingerprint_changes_on_volume_param():
    fp_default = _config_fingerprint()
    with patch("data_fetcher.MIN_DAILY_VOLUME_USD", 9999.0):
        fp_changed = _config_fingerprint()
    assert fp_default != fp_changed, "Changing MIN_DAILY_VOLUME_USD must change fingerprint"


def test_fingerprint_changes_on_clip_params():
    fp_default = _config_fingerprint()
    with patch("data_fetcher.PRICE_CLIP_HIGH", 0.99):
        fp_changed = _config_fingerprint()
    assert fp_default != fp_changed, "Changing PRICE_CLIP_HIGH must change fingerprint"


def test_fingerprint_changes_on_resolution_buffer():
    fp_default = _config_fingerprint()
    with patch("data_fetcher.RESOLUTION_BUFFER_DAYS", 14):
        fp_changed = _config_fingerprint()
    assert fp_default != fp_changed, "Changing RESOLUTION_BUFFER_DAYS must change fingerprint"


def test_cache_path_includes_fingerprint(tmp_path):
    """Cache filename must contain the fingerprint."""
    with patch("data_fetcher.CACHE_DIR", tmp_path):
        from data_fetcher import _cache_path
        p = _cache_path(365)
    fp = _config_fingerprint()
    assert fp in p.name, f"Fingerprint {fp!r} not found in cache path {p.name!r}"
