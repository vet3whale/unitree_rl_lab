# B2W Deployment Target

This directory is the C++ deployment target for the **Unitree B2W** (wheeled B2). It was created by forking `deploy/robots/b2/` and applying the minimum changes needed to support the B2W's 16-DOF hybrid control (12 position-controlled leg joints + 4 velocity-controlled wheel joints).

---

## How this folder was created

### Source: `deploy/robots/b2/`

The B2W deployment was forked from the existing B2 target (`deploy/robots/b2/`). The B2 was chosen as the base because:

- B2W is physically the same robot as B2, with wheels added at the four feet.
- Both robots use the same **`unitree_go` DDS IDL** (`unitree/dds_wrapper/robots/go2/go2.h`), unlike the G1/H1 which use `unitree_hg`.
- The leg joint topology, joint ordering, and FSM structure are identical.

The existing **`deploy/robots/go2w/`** (wheeled Go2) was used as the secondary reference, because it is the only other wheeled robot in this repo and its `State_RLBase.cpp` already implements the position/velocity split loop.

No files under `deploy/robots/b2/` or `deploy/include/` were modified during the initial fork. A subsequent change (see below) added a new observation to `deploy/include/isaaclab/envs/mdp/observations/observations.h`.

---

## File-by-file changes

### `CMakeLists.txt`

| What changed | Why |
|---|---|
| `project(b2_controller)` → `project(b2w_controller)` | Each robot needs a distinct CMake project name to avoid target collisions if both are built in the same workspace. |
| `add_executable(b2_ctrl ...)` → `add_executable(b2w_ctrl ...)` | Produces a distinct binary name so both B2 and B2W executables can coexist on the same host. |

Everything else (include paths, link libraries, ONNX runtime path) is identical to `b2/`.

---

### `include/Types.h`

No functional changes. A comment was added explaining that:

1. B2W uses the **`go2` DDS IDL**, not `unitree_hg` (which is only for G1/H1).
2. There is **no compile-time joint-count constant** here. The 16-DOF joint mapping is entirely driven by `deploy.yaml`'s `joint_ids_map` array, which is loaded at runtime by the `isaaclab::ManagerBasedRLEnv` framework.

---

### `main.cpp`

Two changes from `b2/main.cpp`:

1. **Banner**: `"B2 Controller"` → `"B2W Controller"`.

2. **DDS domain**: B2 hardcodes `ChannelFactory::Instance()->Init(0, ...)`. B2W reads the domain from `config/config.yaml`:

```cpp
// b2/main.cpp (original — hardcoded domain 0 = real robot)
unitree::robot::ChannelFactory::Instance()->Init(0, vm["network"].as<std::string>());

// b2w/main.cpp (config-driven)
unitree::robot::ChannelFactory::Instance()->Init(
    param::config["dds_domain_id"].as<int>(), vm["network"].as<std::string>());
```

`param::config` is the `YAML::Node` loaded from `config/config.yaml` by `param::helper()` before this line executes. The domain value is set to `1` (simulator) by default and documented with a `TODO(b2w)` to flip to `0` for real-robot deployment. See `config/config.yaml` → `dds_domain_id` below.

FSM initialization, DDS channel setup, and joystick prompt strings are otherwise identical to `b2/`.

---

### `config/config.yaml`

#### `dds_domain_id` (new top-level field)

```yaml
# 1 = unitree_mujoco simulator (default — safe for sim2sim)
# TODO(b2w): change to 0 for real robot deployment
dds_domain_id: 1
```

This mirrors how `unitree_mujoco/simulate/config.yaml` exposes `domain_id`. Domain 0 is the real robot; domain 1 is the simulator. Keeping the default at 1 prevents accidentally sending commands to a live robot when running sim2sim.

The other robots (`b2`, `go2`, `go2w`, etc.) still hardcode `0` in their `main.cpp`. The same config-driven pattern could be applied to them, but was not done to keep the change b2w-local.

#### Array sizes: 12 → 16

All per-motor YAML arrays (Passive `mode`, Passive `kd`, FixStand `kp`, FixStand `kd`) were extended from 12 entries to 16. Indices 0–11 are the leg joints (same values as B2); indices 12–15 are the four wheel joints.

```yaml
# B2 (12 entries)
kp: [400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400]

# B2W (16 entries — wheels get kp=0, they are velocity-controlled)
kp: [400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 0, 0, 0, 0]
```

#### FixStand `qs` keyframes: 12 → 16 entries

`go2w/config/config.yaml` keeps only 12-entry position keyframes even though kp has 16 entries. This creates a latent out-of-bounds read in `LinearInterpolator`: after `State_FixStand::enter()` overwrites `qs[0]` with 16 current motor positions, the interpolation from `qs[0]` (16 entries) to `qs[1]` (12 entries) accesses `qs[1][12..15]` which is out of range.

B2W fixes this deliberately: all keyframes have 16 entries, with four trailing `0.0` values for the wheels (confirmed from the URDF and `env.yaml` init state).

```yaml
qs: [
  [],   # overwritten at runtime by State_FixStand::enter()
  [0.0, 0.8, -1.5, 0.0, 0.8, -1.5, 0.0, 0.8, -1.5, 0.0, 0.8, -1.5, 0., 0., 0., 0.],
  [0.0, 0.8, -1.5, 0.0, 0.8, -1.5, 0.0, 0.8, -1.5, 0.0, 0.8, -1.5, 0., 0., 0., 0.],
]
```

#### `policy_dir`

Points at:
```
../../../../robot_lab/logs/rsl_rl/unitree_b2w_rough
```
`param::parser_policy_dir()` automatically selects the latest timestamped subdirectory that contains an `exported/` folder.

---

### `src/State_RLBase.cpp`

This is the only file with substantive new logic.

#### Hybrid control split

B2 (position-only):
```cpp
void State_RLBase::run() {
    auto action = env->action_manager->processed_actions();
    for(int i(0); i < env->robot->data.joint_ids_map.size(); i++) {
        lowcmd->msg_.motor_cmd()[env->robot->data.joint_ids_map[i]].q() = action[i];
    }
}
```

B2W (position for legs, velocity for wheels):
```cpp
void State_RLBase::run() {
    auto action = env->action_manager->processed_actions();

    // Leg joints [0, 12): position PD control
    for(int i(0); i < 12; i++) {
        lowcmd->msg_.motor_cmd()[env->robot->data.joint_ids_map[i]].q() = action[i];
    }

    // Wheel joints [12, 16): velocity control
    for(int i(12); i < 16; i++) {
        auto& mc = lowcmd->msg_.motor_cmd()[env->robot->data.joint_ids_map[i]];
        mc.q()   = 0.0f;
        mc.dq()  = action[i];   // already scaled by JointVelocityAction.scale in deploy.yaml
        mc.kp()  = 0.0f;
        mc.kd()  = KD_WHEEL;
        mc.tau() = 0.0f;
    }
}
```

#### Why `action[i]` is not multiplied by `WHEEL_SCALE` here

The `isaaclab::ManagerBasedRLEnv` action manager applies `JointVelocityAction.scale = 5.0` (rad/s per unit policy output) **before** returning `processed_actions()`. Multiplying again in `run()` would give 25× amplification. The scale lives in `deploy.yaml` and is applied once, upstream. A comment in `State_RLBase.cpp` documents this explicitly.

Neither B2 nor go2w use any C++ constant for action scaling — the pattern across the repo is to put all scales in `deploy.yaml` and rely on `processed_actions()`.

#### Why kp/kd are re-asserted in `run()` for wheel joints

`State_RLBase::enter()` (in the shared header `deploy/include/FSM/State_RLBase.h`) sets kp/kd once from `joint_stiffness/joint_damping` (loaded from `deploy.yaml`). However, `State_FixStand::enter()` also writes kp/kd arrays (from `config.yaml`) using raw motor indices. Re-asserting `kp=0, kd=KD_WHEEL` on every `run()` cycle makes the wheel control mode robust to any ordering of FSM state transitions.

---

### `deploy/include/isaaclab/envs/mdp/observations/observations.h` (shared header)

A new observation was added to the shared header. This is the only change outside the `b2w/` directory.

#### `REGISTER_OBSERVATION(joint_pos_rel_without_wheel)`

The B2W policy was trained with a custom observation function (`mdp.joint_pos_rel_without_wheel` in `robot_lab/`) that computes `q − q_default` for all 16 joints but **zeroes the wheel slots** (indices 12–15). Wheel joints have no meaningful relative position (they spin continuously), and the policy was trained expecting `0.0` in those four slots.

The existing `joint_pos_rel` observation does not support zeroing; it only optionally filters to a subset of joints (which would shrink the output dimension and break the policy input shape). A new registration was therefore added:

```cpp
// Like joint_pos_rel, but zeroes the slots listed in params["wheel_joint_ids"].
// Output length always equals the full joint count (16 for B2W), matching the
// policy input dimension.
REGISTER_OBSERVATION(joint_pos_rel_without_wheel)
{
    auto & asset = env->robot;
    std::vector<float> data(asset->data.joint_pos.size());
    for (size_t i = 0; i < asset->data.joint_pos.size(); ++i)
        data[i] = asset->data.joint_pos[i] - asset->data.default_joint_pos[i];

    try {
        const auto wheel_ids = params["wheel_joint_ids"].as<std::vector<int>>();
        for (int idx : wheel_ids)
            if (idx >= 0 && idx < static_cast<int>(data.size())) data[idx] = 0.0f;
    } catch (const std::exception &) {}

    return data;
}
```

The function is generic: the zero indices come from `params["wheel_joint_ids"]` in `deploy.yaml`, not hardcoded. Other wheeled robots can use it by listing their own wheel indices.

The corresponding `deploy.yaml` observation entry:
```yaml
joint_pos_rel_without_wheel:
    params: {wheel_joint_ids: [12, 13, 14, 15]}
```

---

## How `robot_lab/` was used

The `robot_lab/` submodule contains all the Isaac Lab training configuration and logs for B2W. Several pieces of ground-truth information were read directly from it before writing any deployment code.

### URDF and joint names

`robot_lab/source/robot_lab/data/Robots/unitree/b2w_description/urdf/b2w_description.urdf` — confirmed the robot has 16 actuated joints: 12 leg joints (`*_hip_joint`, `*_thigh_joint`, `*_calf_joint`) and 4 wheel joints (`*_foot_joint`).

> **Note**: The URDF document order is FL/FR/RL/RR, but training uses `preserve_order=True` with an explicit joint name list in `rough_env_cfg.py` giving FR/FL/RR/RL. The MJCF (`unitree_mujoco/unitree_robots/b2w/b2w.xml`) actuator order also follows FR/FL/RR/RL, so no permutation is needed — `joint_ids_map = [0..15]` is correct.

> **Note**: The URDF wheel joints are named `*_foot_joint`; the MJCF names them `*_wheel_joint`. The physical ordering is identical; only the names differ between the two model files.

### Training environment config

`robot_lab/source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/wheeled/unitree_b2w/rough_env_cfg.py` — provided:

| Value | Source field |
|---|---|
| Joint ordering: legs then wheels | `leg_joint_names + wheel_joint_names` |
| Leg position action scale: 0.125 (hip), 0.25 (thigh/calf) | `actions.joint_pos.scale` |
| Wheel velocity action scale: 5.0 rad/s | `actions.joint_vel.scale` |
| Wheel `default_q = 0.0` | `joint_pos init_state: .*_foot_joint: 0.0` |
| FixStand target: thigh=0.8, calf=−1.5, hip=0.0 | `joint_pos init_state` |
| Observation `joint_pos` uses `joint_pos_rel_without_wheel` | `observations.policy.joint_pos.func` |

### Saved training run parameters

`robot_lab/logs/rsl_rl/unitree_b2w_rough/2026-05-24_05-47-04/params/env.yaml` — the frozen environment config saved alongside the best trained policy. Used to confirm:

- `step_dt = sim_dt × decimation = 0.005 × 4 = 0.02 s`
- Actuator stiffness: hip/thigh/calf = 160.0 (simulation value)
- Actuator damping: leg = 5.0, wheel = 1.0 (simulation value)
- All joint names and ordering in both `actions` and `observations` sections

### Existing trained policy

`robot_lab/logs/rsl_rl/unitree_b2w_rough/2026-05-24_05-47-04/exported/policy.onnx` — a complete trained rough-terrain policy already exists. The `deploy.yaml` alongside it is required by the C++ runner (`State_RLBase` loads it via `YAML::LoadFile(policy_dir / "params" / "deploy.yaml")`); without it the executable would abort on startup.

### MJCF joint-order verification

`unitree_mujoco/unitree_robots/b2w/b2w.xml` `<actuator>` section — confirmed that the MJCF actuator indices 0–15 follow the same FR/FL/RR/RL ordering as training. No permutation vector is needed.

---

## Open TODOs before first hardware run

| File | What to verify/tune |
|---|---|
| `config/config.yaml` | **`dds_domain_id`**: change from `1` (sim) to `0` (real robot) |
| `deploy.yaml` | **`joint_ids_map` indices 12–15** (wheel hardware motor indices): assumed sequential 12,13,14,15; confirm against B2W SDK motor map |
| `deploy.yaml` | `stiffness[0..11]` — hardware leg stiffness (160.0 mirrors sim; real B2W hardware may use different values) |
| `deploy.yaml` | `damping[12..15]` — wheel hardware damping (1.0 is the sim value; target ~2.0 per B2W spec) |
| `src/State_RLBase.cpp` | `KD_WHEEL = 1.0f` — must match `deploy.yaml damping[12..15]` at all times |
| `config/config.yaml` | Passive and FixStand wheel kd (indices 12–15), currently 3 |
| `config/config.yaml` | Confirm `qs` wheel entries `0.0` are safe default positions |
| `config/config.yaml` | Switch `policy_dir` to a dedicated velocity-training run once available |

---

## Sim2Sim validation (unitree_mujoco)

Before running on hardware, validate the policy in `unitree_mujoco`.

### Pre-flight

1. **`policy.onnx`**: place at `robot_lab/logs/rsl_rl/unitree_b2w_rough/<run>/exported/policy.onnx`. `parser_policy_dir()` picks the latest run with an `exported/` folder automatically.

2. **Edit `unitree_mujoco/simulate/config.yaml`**:
   ```yaml
   robot: "b2w"   # was: "go2"
   ```
   All other fields (`domain_id: 1`, `interface: "lo"`) already match the controller defaults.

3. **No environment variables needed** for loopback / domain 1. If a system-wide `CYCLONEDDS_URI` restricts to a physical interface, `unset CYCLONEDDS_URI` before launching.

### Terminal 1 — simulator

```bash
cd unitree_mujoco/simulate
mkdir -p build && cd build && cmake .. && make -j$(nproc) && cd ..
./build/unitree_mujoco
```

Expected: MuJoCo window opens with B2W; console prints `"Mujoco data is prepared"`.

### Terminal 2 — controller

```bash
cd unitree_rl_lab/deploy/robots/b2w
mkdir -p build && cd build && cmake .. && make -j$(nproc) && cd ..
./build/b2w_ctrl --network lo
```

Expected: `"Connected to robot."` then FSM in Passive state.

### Operating sequence

| Action | Button combo | Expected |
|---|---|---|
| Passive → FixStand | `LT + A` | Legs interpolate to hip=0, thigh=0.8, calf=−1.5 over ~3 s |
| FixStand → RL | `Start` | Policy activates at 50 Hz; wheels velocity-controlled |
| Emergency stop | `LT + B` | Any state → Passive (damped) |

### Key checks

- Wheels spin when forward velocity commanded (primary locomotion).
- No drift with cmd_vel = 0 (wheel kd holding).
- Yaw command produces differential wheel speeds.
- No NaN/Inf in console; `action[12..15]` near 0 at rest.

---

## Build and run (hardware)

```bash
# 1. Set dds_domain_id: 0 in config/config.yaml
# 2. Build
cd deploy/robots/b2w
mkdir build && cd build
cmake .. && make
# 3. Launch on the robot's network interface (e.g. eth0)
./b2w_ctrl --network eth0
```

FSM button map (inherited from B2):

| Button | Action |
|---|---|
| `L2 + A` | Passive → FixStand |
| `Start` | FixStand → Velocity (RL policy) |
| `L2 + B` | Any → Passive (emergency damp) |
| *(auto)* | Bad orientation detected → Passive |
