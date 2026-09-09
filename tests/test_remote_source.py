import numpy as np

from failsafe.corpus.assets import CutoutLibrary
from failsafe.workload.remote_source import RemoteSyntheticSource
from failsafe.workload.source import SyntheticSceneSource


def test_remote_source_matches_in_process_rendering():
    local = SyntheticSceneSource.from_tier("smoke", 1, CutoutLibrary())
    remote = RemoteSyntheticSource("smoke", 1)
    try:
        assert remote.corpus_hash == local.corpus_hash
        assert remote.cameras == local.cameras
        assert len(remote.ground_truth.events) == len(local.ground_truth.events)
        e = local.ground_truth.events[0]
        t = (e.start + e.end) / 2
        # request/response mode
        assert np.array_equal(remote.frame(e.camera, t), local.frame(e.camera, t))
        # stream mode, in order
        schedule = [(0.0, "A"), (0.0, "B"), (1 / 30, "A"), (t, e.camera)]
        remote.start_stream(schedule)
        for tt, cam in schedule:
            assert np.array_equal(remote.frame(cam, tt), local.frame(cam, tt))
    finally:
        remote.close()
    assert not remote._proc.is_alive()
