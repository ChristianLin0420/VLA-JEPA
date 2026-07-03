import unittest

import numpy as np

from examples.LIBERO.scene_edits import displace_object, insert_occluder


class _FakeModel:
    """Duck-typed robosuite MjModel: one hinge robot joint + two free object joints."""

    def __init__(self):
        self._names = ["robot0_joint1", "akita_black_bowl_1_joint0", "plate_1_joint0"]
        self.njnt = len(self._names)
        self.jnt_type = np.array([3, 0, 0])  # hinge, free, free
        self._qpos_addr = {
            "akita_black_bowl_1_joint0": (9, 16),
            "plate_1_joint0": (16, 23),
        }

    def joint_id2name(self, joint_id):
        return self._names[joint_id]

    def get_joint_qpos_addr(self, name):
        return self._qpos_addr[name]


class _FakeSim:
    def __init__(self):
        self.model = _FakeModel()
        self.data = type("Data", (), {})()
        self.data.qpos = np.zeros(23)
        self.data.qpos[9:12] = [0.1, 0.2, 0.9]
        self.data.qpos[12:16] = [1.0, 0.0, 0.0, 0.0]
        self.forward_calls = 0

    def forward(self):
        self.forward_calls += 1


class _FakeEnv:
    def __init__(self):
        self.sim = _FakeSim()

    def check_success(self):
        return False


class SceneEditsTest(unittest.TestCase):
    def test_displace_object_edits_qpos_and_forwards(self):
        env = _FakeEnv()
        old_pos, new_pos = displace_object(env, "akita_black_bowl_1", [0.05, -0.02, 0.0])
        np.testing.assert_allclose(old_pos, [0.1, 0.2, 0.9])
        np.testing.assert_allclose(new_pos, [0.15, 0.18, 0.9])
        np.testing.assert_allclose(env.sim.data.qpos[9:12], [0.15, 0.18, 0.9])
        self.assertEqual(env.sim.forward_calls, 1)

    def test_scalar_offset_is_seed_deterministic_and_planar(self):
        first = displace_object(_FakeEnv(), "plate_1", 0.05, seed=11)
        second = displace_object(_FakeEnv(), "plate_1", 0.05, seed=11)
        np.testing.assert_allclose(first[1], second[1])
        delta = first[1] - first[0]
        self.assertAlmostEqual(float(np.linalg.norm(delta)), 0.05)
        self.assertEqual(delta[2], 0.0)

    def test_scalar_offset_requires_seed(self):
        with self.assertRaises(ValueError):
            displace_object(_FakeEnv(), "plate_1", 0.05)

    def test_missing_object_raises(self):
        with self.assertRaises(ValueError):
            displace_object(_FakeEnv(), "mug_1", [0.05, 0.0, 0.0])

    def test_robot_hinge_joint_is_never_matched(self):
        with self.assertRaises(ValueError):
            displace_object(_FakeEnv(), "robot0", [0.05, 0.0, 0.0])

    def test_occluder_stub_raises_not_implemented(self):
        with self.assertRaisesRegex(NotImplementedError, "BDDL"):
            insert_occluder(_FakeEnv(), pos=[0.0, 0.0, 1.0], size=[0.1, 0.1, 0.01])

    def test_reverted_qpos_edit_raises_runtime_error(self):
        # Validity guards must be RuntimeError raises, not asserts (which
        # vanish under python -O).
        env = _FakeEnv()

        def _reverting_forward():
            env.sim.data.qpos[9:12] = [0.1, 0.2, 0.9]

        env.sim.forward = _reverting_forward
        with self.assertRaisesRegex(RuntimeError, "did not persist"):
            displace_object(env, "akita_black_bowl_1", [0.05, -0.02, 0.0])

    def test_non_bool_check_success_raises_runtime_error(self):
        env = _FakeEnv()
        env.check_success = lambda: 0.7
        with self.assertRaisesRegex(RuntimeError, "check_success"):
            displace_object(env, "akita_black_bowl_1", [0.05, -0.02, 0.0])


if __name__ == "__main__":
    unittest.main()
