"""compute_env records the thread counts and the VQC chunk override a run was launched with."""

from qgridbench.utils.run_tracking import compute_env


def test_records_vqc_chunk_override(monkeypatch):
    monkeypatch.setenv("QGB_VQC_CHUNK", "256")
    monkeypatch.setenv("OMP_NUM_THREADS", "4")
    env = compute_env()
    assert env["qgb_vqc_chunk"] == "256" and env["omp_num_threads"] == "4"


def test_absent_override_is_none(monkeypatch):
    monkeypatch.delenv("QGB_VQC_CHUNK", raising=False)
    assert compute_env()["qgb_vqc_chunk"] is None
