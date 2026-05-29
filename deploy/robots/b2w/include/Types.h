#pragma once

// B2W uses the same unitree_go DDS IDL as B2/Go2, not unitree_hg (G1/H1).
// The 16-DOF joint mapping (12 legs + 4 wheels) is configured via deploy.yaml
// (joint_ids_map array) — there is no compile-time joint-count constant here.
#include "unitree/dds_wrapper/robots/go2/go2.h"

using LowCmd_t = unitree::robot::go2::publisher::LowCmd;
using LowState_t = unitree::robot::go2::subscription::LowState;
