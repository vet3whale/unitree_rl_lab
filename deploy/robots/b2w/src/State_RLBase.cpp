#include "FSM/State_RLBase.h"
#include "unitree_articulation.h"
#include "isaaclab/envs/mdp/observations/observations.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"
#include "RecurrentOrtRunner.h"
#include <unordered_map>
#include <cstdio>

namespace isaaclab
{
// Keyboard velocity commands for sim2sim testing (no gamepad required).
// To use: in deploy.yaml rename "velocity_commands" → "keyboard_velocity_commands".
// To revert to gamepad/DDS joystick: rename back to "velocity_commands".
//
// Speed gear (QWERTYUIOP row, latched): selects a global magnitude scale applied
// to the chosen direction. Level n -> n/9 of max command magnitude.
//   q=0 (stop)  w=1  e=2  r=3  t=4  y=5  u=6  i=7  o=8  p=9 (max)
// The gear persists until another gear key is pressed (default 0 = stationary).
//
// Direction (numpad with NumLock ON, momentary - only while held):
//   forward : numpad-8        backward: numpad-2
//   left    : numpad-4        right   : numpad-6
//   yaw CCW : numpad-7        yaw CW  : numpad-9
//   stop    : any other key (or no key)
//
// Final command = direction_unit * gear_scale.
REGISTER_OBSERVATION(keyboard_velocity_commands)
{
    std::string key = FSMState::keyboard->key();

    // Latched speed gear from the QWERTYUIOP row: q=0 ... p=9 -> level/9 of max.
    static const std::unordered_map<std::string, int> gear_keys = {
        {"q", 0}, {"w", 1}, {"e", 2}, {"r", 3}, {"t", 4},
        {"y", 5}, {"u", 6}, {"i", 7}, {"o", 8}, {"p", 9},
    };
    static float speed_scale = 0.0f;            // persists across calls (latched)
    static int   gear        = 0;
    auto git = gear_keys.find(key);
    if (git != gear_keys.end() && git->second != gear) {  // only on change
        gear = git->second;
        speed_scale = gear / 9.0f;
        std::printf("[b2w] Speed gear set: %d/9 (%.0f%% of max)\n",
                    gear, speed_scale * 100.0f);
        std::fflush(stdout);
    }

    // Numpad picks direction (momentary); gear scales the magnitude.
    static const std::unordered_map<std::string, std::vector<float>> dir_keys = {
        {"8", {1.0f,  0.0f,  0.0f}},  // forward
        {"2", {-1.0f, 0.0f,  0.0f}},  // backward
        {"4", {0.0f,  1.0f,  0.0f}},  // strafe left
        {"6", {0.0f,  -1.0f, 0.0f}},  // strafe right
        {"7", {0.0f,  0.0f,  1.0f}},  // yaw CCW
        {"9", {0.0f,  0.0f,  -1.0f}}, // yaw CW
    };

    std::vector<float> cmd = {0.0f, 0.0f, 0.0f};
    auto dit = dir_keys.find(key);
    if (dit != dir_keys.end())
        for (size_t i = 0; i < 3; i++)
            cmd[i] = dit->second[i] * speed_scale;
    return cmd;
}

} // namespace isaaclab

// Wheel velocity control gains (hardware).
// KD_WHEEL is also set via deploy.yaml joint_damping[12..15]; keep in sync.
// The velocity scale (WHEEL_SCALE=5.0) is configured in deploy.yaml
// JointVelocityAction.scale and is already applied by processed_actions() —
// do NOT multiply action[i] by it again here.
static constexpr float KD_WHEEL = 1.0f;  // TODO(b2w): tune — sim damping is 1.0, hardware may differ

State_RLBase::State_RLBase(int state_mode, std::string state_string)
: FSMState(state_mode, state_string)
{
    auto cfg = param::config["FSM"][state_string];
    auto policy_dir = param::parser_policy_dir(cfg["policy_dir"].as<std::string>());

    env = std::make_unique<isaaclab::ManagerBasedRLEnv>(
        YAML::LoadFile(param::config_dir / "deploy.yaml"),
        std::make_shared<unitree::BaseArticulation<LowState_t::SharedPtr>>(FSMState::lowstate)
    );
    // b2w-local runner: handles both stateless MLP and recurrent (LSTM/GRU) students by carrying
    // the policy's hidden state across steps. See RecurrentOrtRunner.h.
    env->alg = std::make_unique<b2w::RecurrentOrtRunner>(policy_dir / "exported" / "policy.onnx");

    this->registered_checks.emplace_back(
        std::make_pair(
            [&]()->bool{ return isaaclab::mdp::bad_orientation(env.get(), 1.0); },
            FSMStringMap.right.at("Passive")
        )
    );
}

// this sends the RL policy's processed outputs into the robot's low-level motor command message
void State_RLBase::run()
{
    auto action = env->action_manager->processed_actions();

    // Leg joints [0, 12): position PD control.
    // processed_actions() returns default_pos + scale * raw_action (radians).
    for(int i(0); i < 12; i++) {
        lowcmd->msg_.motor_cmd()[env->robot->data.joint_ids_map[i]].q() = action[i];
    }

    // Wheel joints [12, 16): velocity control.
    // processed_actions() returns scale * raw_action (rad/s); scale=5.0 from deploy.yaml.
    // kp=0 and kd=KD_WHEEL are set once in enter() via joint_stiffness/joint_damping (deploy.yaml);
    // we assert kp/kd here on every cycle to be safe in case another state wrote to these slots.
    for(int i(12); i < 16; i++) {
        auto& mc = lowcmd->msg_.motor_cmd()[env->robot->data.joint_ids_map[i]];
        mc.q()   = 0.0f;
        mc.dq()  = action[i];
        mc.kp()  = 0.0f;
        mc.kd()  = KD_WHEEL;
        mc.tau() = 0.0f;
    }
}
