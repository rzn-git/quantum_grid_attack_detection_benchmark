"""`spawn_rng` as every stored run used it: a stream is keyed on the first eight characters of
its name. The function must not change (it would move every seeded result), so its limit is
pinned here as behaviour, not fixed."""

import numpy as np

from qgridbench.utils.seeding import spawn_rng


def _draws(seed: int, stream: str) -> np.ndarray:
    return spawn_rng(seed, stream).random(8)


def test_names_sharing_eight_leading_characters_are_one_stream():
    # the three label-flip rates, and the supplied-start pool and pick (decision log
    # 2026-11-21T06:57Z, INCIDENT): changing spawn_rng to separate them breaks this test
    assert np.array_equal(_draws(0, "poison_0.05"), _draws(0, "poison_0.2"))
    assert np.array_equal(_draws(3, "hsj_start_pool"), _draws(3, "hsj_start_pick"))


def test_names_differing_within_eight_characters_are_separate_streams():
    assert not np.array_equal(_draws(0, "abcdefgh"), _draws(0, "abcdefgi"))
    assert not np.array_equal(_draws(0, "pool_hsj_start"), _draws(0, "pick_hsj_start"))


def test_streams_keep_the_values_every_stored_run_drew():
    """Raw 64-bit outputs of three streams, recorded 2026-11-21. Any change to how the seed
    and the name enter the generator, or to the generator, moves them (the merge tests alone
    passed with the SeedSequence entries swapped: fix-check of the round of 07:15Z, F7)."""
    golden = {
        (0, "attack"): [3069180591303165213, 8615556524323632030, 4218426473529482138],
        (3, "poison_0.05"): [15887172263490654286, 4961368478145894187, 9566231126338135865],
        (7, "adv"): [9297009480339655343, 6623466107061061998, 7415609796814414953],
    }
    for (seed, stream), values in golden.items():
        raw = spawn_rng(seed, stream).bit_generator.random_raw(3)
        assert [int(v) for v in raw] == values, (seed, stream)


def test_a_stream_is_reproducible_and_depends_on_the_seed():
    assert np.array_equal(_draws(1, "attack"), _draws(1, "attack"))
    assert not np.array_equal(_draws(1, "attack"), _draws(2, "attack"))
