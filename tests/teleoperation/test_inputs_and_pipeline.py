import json
import time

import numpy as np
import pytest
import zmq

from teleoperation import run_sim
from teleoperation.core import load_config
from teleoperation.inputs import LiveSource, parse_board_poses, parse_ergonomics


def test_parse_ergonomics_splits_gloves_and_drops_missing_one():
    values = [0.0] * 20 + [float(i) for i in range(20)]
    parsed = parse_ergonomics(",".join(f"{v:.3f}" for v in values))
    assert parsed["left"] is None
    np.testing.assert_array_equal(parsed["right"], np.arange(20))


def test_parse_board_poses_with_alternatives():
    flipped = np.diag([1.0, -1.0, -1.0])
    message = {"timestamp": 1.0, "poses": {"1": {"position": [1, 2, 3],
                                                 "rotation_matrix": np.eye(3).tolist(),
                                                 "confidence": 0.9,
                                                 "alternatives": [{"position": [1, 2, 3.01],
                                                                   "rotation_matrix": flipped.tolist()}]}}}
    pose, confidence, alternatives = parse_board_poses(message)["1"]
    np.testing.assert_array_equal(pose.position, [1, 2, 3])
    assert confidence == 0.9
    assert len(alternatives) == 1
    np.testing.assert_array_equal(alternatives[0].rotation, flipped)


def test_parse_skeleton_reads_each_gloves_wrist_orientation():
    from scipy.spatial.transform import Rotation as R
    from teleoperation.inputs import parse_skeleton
    def glove(gid, quat):
        nodes = [[0.0, 0.0, 0.0, *quat]] + [[0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0]] * 24
        return [gid] + [f"{v:.3f}" for node in nodes for v in node]
    q = R.from_euler("z", 30, degrees=True).as_quat()
    parsed = parse_skeleton(",".join(glove("DC8B6BA8", q) + glove("5186fac7", [0, 0, 0, 1])))
    assert set(parsed) == {"dc8b6ba8", "5186fac7"}
    assert abs(parsed["dc8b6ba8"].magnitude() - np.radians(30)) < 1e-3


def test_live_source_receives_both_streams():
    context = zmq.Context.instance()
    glove = context.socket(zmq.PUSH)
    glove_port = glove.bind_to_random_port("tcp://127.0.0.1")
    poses = context.socket(zmq.PUB)
    pose_port = poses.bind_to_random_port("tcp://127.0.0.1")
    source = LiveSource(f"tcp://127.0.0.1:{glove_port}", f"tcp://127.0.0.1:{pose_port}",
                        {"right": "0", "left": "1"})
    try:
        deadline = time.monotonic() + 3.0
        inputs = source.latest()
        while time.monotonic() < deadline and (inputs.glove["left"] is None or inputs.wrist["left"] is None):
            glove.send_string(",".join(["10.0"] * 20 + ["0.0"] * 20))
            poses.send_json({"timestamp": 0, "poses": {"1": {"position": [0.1, 0.2, 0.3],
                                                             "rotation_matrix": np.eye(3).tolist(),
                                                             "confidence": 1.0}}})
            time.sleep(0.05)
            inputs = source.latest()
        assert inputs.glove["left"] is not None and inputs.glove["right"] is None
        np.testing.assert_array_equal(inputs.wrist["left"].pose.position, [0.1, 0.2, 0.3])
        assert inputs.wrist["right"] is None
    finally:
        source.close()
        glove.close(linger=0)
        poses.close(linger=0)


@pytest.fixture(scope="module")
def recorded(tmp_path_factory):
    """A short mock teleoperation run, recorded and then replayed."""
    directory = tmp_path_factory.mktemp("demos")
    config = load_config()
    config["tracking"]["countdown_s"] = 0.5
    config["imu_fusion"]["mount_file"] = None
    config_path = directory / "teleop.json"
    config_path.write_text(json.dumps(config))
    run_sim.main(["--source", "mock", "--headless", "--duration", "3", "--record",
                  "--config", str(config_path), "--output-dir", str(directory / "original")])
    original = next((directory / "original").glob("*.npz"))
    run_sim.main(["--source", "replay", "--replay", str(original), "--headless", "--record",
                  "--config", str(config_path), "--output-dir", str(directory / "replayed")])
    replayed = next((directory / "replayed").glob("*.npz"))
    return original, replayed


def test_mock_run_records_a_demonstration(recorded):
    original, _ = recorded
    data = np.load(original)
    info = json.loads(original.with_suffix(".json").read_text())
    n = len(data["time"])
    assert n > 50 and info["samples"] == n and info["success"] is None
    for side in ("right", "left"):
        assert data[f"{side}_arm_q_commanded"].shape == (n, 6)
        assert data[f"{side}_hand_q_measured"].shape == (n, 20)
        assert data[f"{side}_wrist_rotation"].shape == (n, 3, 3)
        assert data[f"{side}_engaged"].all()
        # The simulated Hand base keeps up with its target.
        lag = np.linalg.norm(data[f"{side}_hand_base_measured"][:, :3] - data[f"{side}_hand_base_target"][:, :3], axis=1)
        assert np.median(lag) < 0.01
    assert data["object_pose"].shape == (n, 7)


def test_replay_reproduces_the_demonstration(recorded):
    original, replayed = (np.load(p) for p in recorded)
    n = min(len(original["time"]), len(replayed["time"]))
    assert abs(len(original["time"]) - len(replayed["time"])) <= 2
    for side in ("right", "left"):
        diff = np.abs(original[f"{side}_hand_base_target"][:n, :3] - replayed[f"{side}_hand_base_target"][:n, :3])
        assert np.median(diff) < 1e-3
