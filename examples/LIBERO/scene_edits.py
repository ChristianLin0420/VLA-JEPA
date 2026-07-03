"""Seeded MuJoCo scene edits applied at decision boundaries (T0.5/T0.8/T2.1).

Duck-typed against the robosuite ``env.sim`` wrapper: all MuJoCo access
happens inside the helpers, so the module imports without a simulator
installed and the edit logic is unit-testable against a fake sim.
"""

import numpy as np

_MJ_JOINT_FREE = 0  # mujoco mjtJoint.mjJNT_FREE


def _free_joint_name(sim, object_name: str) -> str:
    """Free joint backing ``object_name`` (robosuite names them '<object>_joint0')."""
    for joint_id in range(sim.model.njnt):
        name = sim.model.joint_id2name(joint_id)
        if name and name.startswith(object_name) and sim.model.jnt_type[joint_id] == _MJ_JOINT_FREE:
            return name
    raise ValueError(f"no free joint found for object '{object_name}'")


def displace_object(env, object_name: str, dxyz, seed: int | None = None):
    """Displace a free-body object by ``dxyz`` meters and re-forward the sim.

    ``dxyz`` is either a 3-vector applied verbatim, or a scalar magnitude
    combined with ``seed`` to draw a reproducible random direction in the
    xy-plane (the T2.1 "seeded 4-6 cm offset"). Call at a decision boundary,
    before the next observation is rendered. Returns ``(old_pos, new_pos)``.
    Guards raise RuntimeError (asserts vanish under ``-O``) if the qpos edit
    did not persist through ``sim.forward()`` or ``env.check_success()`` stops
    returning a bool afterwards.
    """
    dxyz = np.asarray(dxyz, dtype=np.float64)
    if dxyz.ndim == 0:
        if seed is None:
            raise ValueError("scalar dxyz requires a seed to draw a direction")
        theta = np.random.default_rng(seed).uniform(0.0, 2.0 * np.pi)
        dxyz = float(dxyz) * np.array([np.cos(theta), np.sin(theta), 0.0])
    if dxyz.shape != (3,):
        raise ValueError(f"dxyz must be a scalar or 3-vector, got shape {dxyz.shape}")

    sim = env.sim
    joint_name = _free_joint_name(sim, object_name)
    start, end = sim.model.get_joint_qpos_addr(joint_name)
    if end - start != 7:
        raise ValueError(f"joint '{joint_name}' is not a free joint (qpos span {end - start})")
    old_pos = np.array(sim.data.qpos[start : start + 3])
    sim.data.qpos[start : start + 3] = old_pos + dxyz
    sim.forward()

    new_pos = np.array(sim.data.qpos[start : start + 3])
    if not np.allclose(new_pos, old_pos + dxyz):
        raise RuntimeError("qpos edit did not persist through sim.forward()")
    success = env.check_success()
    if not isinstance(success, (bool, np.bool_)):
        raise RuntimeError(
            f"env.check_success() no longer returns a bool after the edit: {success!r}"
        )
    return old_pos, new_pos


def insert_occluder(env, pos, size, name: str = "mem_occluder", seed: int | None = None):
    """Insert (or position) an occluder geom between agentview and the target.

    Not implemented: MuJoCo compiles geoms into the model at construction and
    robosuite/LIBERO build that model from the task BDDL, so injecting a new
    geom at runtime requires model recompilation, i.e. versioned BDDL edits
    under ``benchmarks/libero_mem_v0`` (T0.8 owns that iteration). Use
    ``displace_object`` for qpos-editable scene manipulations until then.
    """
    raise NotImplementedError(
        "Runtime occluder injection needs the geom declared in the task BDDL "
        "(MuJoCo cannot add geoms to a compiled model); see T0.8 / "
        "benchmarks/libero_mem_v0 for the versioned-BDDL path."
    )
