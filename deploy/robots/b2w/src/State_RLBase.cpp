#include "FSM/State_RLBase.h"
#include "unitree_articulation.h"
#include "isaaclab/envs/mdp/observations/observations.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"
#include <unordered_map>

namespace isaaclab
{
// Keyboard velocity commands for sim2sim testing (no gamepad required).
// To use: in deploy.yaml rename "velocity_commands" → "keyboard_velocity_commands".
// To revert to gamepad/DDS joystick: rename back to "velocity_commands".
//
// Key layout (WASD ring + QE yaw + arrows + numpad with NumLock on):
//   forward : w, up-arrow, numpad-8
//   backward: s, down-arrow, numpad-2
//   left    : a, numpad-4
//   right   : d, numpad-6
//   yaw CCW : q, left-arrow, numpad-7
//   yaw CW  : e, right-arrow, numpad-9
//   stop    : any other key (or no key)
REGISTER_OBSERVATION(keyboard_velocity_commands)
{
    std::string key = FSMState::keyboard->key();
    static const std::unordered_map<std::string, std::vector<float>> key_commands = {
        {"w", {1.0f,  0.0f,  0.0f}},
        {"s", {-1.0f, 0.0f,  0.0f}},
        {"a", {0.0f,  1.0f,  0.0f}},
        {"d", {0.0f,  -1.0f, 0.0f}},
        {"q", {0.0f,  0.0f,  1.0f}},
        {"e", {0.0f,  0.0f,  -1.0f}},
    };

    std::vector<float> cmd = {0.0f, 0.0f, 0.0f};
    auto it = key_commands.find(key);
    if (it != key_commands.end())
        cmd = it->second;
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
    env->alg = std::make_unique<isaaclab::OrtRunner>(policy_dir / "exported" / "policy.onnx");

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
