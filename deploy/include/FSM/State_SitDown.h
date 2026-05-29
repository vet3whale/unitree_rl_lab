#pragma once

#include "FSMState.h"
#include <algorithm>

// Two-phase controlled sit:
//   Phase 1 (settle_time): kp=0, kd=Passive kd — mirrors Passive, lets the
//     robot stop under damping before position control re-engages.  This
//     prevents the side-fall that happens when switching directly from an
//     active RL gait to position targets.
//   Phase 2 (duration): position-controls leg joints from the settled actual
//     pose to the sit target.  Wheel joints (kp==0 in FixStand config) keep
//     kp=0 and damp to a stop rather than chasing a position target.
//   End: auto-transitions to Passive.
class State_SitDown : public FSMState
{
public:
    State_SitDown(int state, std::string state_string = "SitDown")
    : FSMState(state, state_string), done_(false), settled_(false)
    {
        sit_q_       = param::config["FSM"]["FixStand"]["qs"][1].as<std::vector<float>>();
        settle_time_ = param::config["FSM"]["SitDown"]["settle_time"].as<float>();
        duration_    = param::config["FSM"]["SitDown"]["duration"].as<float>();
        kp_stand_    = param::config["FSM"]["FixStand"]["kp"].as<std::vector<float>>();
        kd_stand_    = param::config["FSM"]["FixStand"]["kd"].as<std::vector<float>>();
        kd_passive_  = param::config["FSM"]["Passive"]["kd"].as<std::vector<float>>();

        registered_checks.emplace_back(
            std::make_pair(
                [this]()->bool{ return done_; },
                FSMStringMap.right.at("Passive")
            )
        );
    }

    void enter()
    {
        // Phase 1 gains: damp only, no position stiffness
        for(int i = 0; i < (int)kd_passive_.size(); ++i)
        {
            auto & motor = lowcmd->msg_.motor_cmd()[i];
            motor.kp()  = 0;
            motor.kd()  = kd_passive_[i];
            motor.dq()  = motor.tau() = 0;
        }
        q0_.clear();
        done_    = false;
        settled_ = false;
        t0_      = now();
    }

    void run()
    {
        double t = now() - t0_;

        if(!settled_)
        {
            // Phase 1: follow actual joint positions under pure damping
            for(int i = 0; i < (int)kd_passive_.size(); ++i)
                lowcmd->msg_.motor_cmd()[i].q() = lowstate->msg_.motor_state()[i].q();

            if(t >= settle_time_)
            {
                // Capture actual pose as interpolation start; apply position gains
                for(int i = 0; i < (int)kp_stand_.size(); ++i)
                {
                    q0_.push_back(lowstate->msg_.motor_state()[i].q());
                    auto & motor = lowcmd->msg_.motor_cmd()[i];
                    motor.kp() = kp_stand_[i];
                    motor.kd() = kd_stand_[i];
                }
                t_interp_ = t;
                settled_  = true;
            }
        }
        else
        {
            // Phase 2: interpolate leg joints toward sit pose.
            // Skip joints where kp==0 (wheels) — they just damp to a stop.
            float alpha = std::min((float)((t - t_interp_) / duration_), 1.0f);
            int   n     = (int)std::min(q0_.size(), sit_q_.size());
            for(int i = 0; i < n; ++i)
            {
                if(kp_stand_[i] > 0)
                    lowcmd->msg_.motor_cmd()[i].q() = q0_[i] + alpha * (sit_q_[i] - q0_[i]);
            }
            if(alpha >= 1.0f) done_ = true;
        }
    }

private:
    static double now()
    {
        return (double)unitree::common::GetCurrentTimeMillisecond() * 1e-3;
    }

    double             t0_, t_interp_, settle_time_, duration_;
    bool               done_, settled_;
    std::vector<float> q0_, sit_q_, kp_stand_, kd_stand_, kd_passive_;
};

REGISTER_FSM(State_SitDown)
