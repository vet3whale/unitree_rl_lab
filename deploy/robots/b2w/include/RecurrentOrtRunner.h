// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.
//
// b2w-local ONNX policy runner. Lives in the b2w robot folder so the shared deploy library
// (isaaclab::OrtRunner in ../../include/isaaclab/algorithms/algorithms.h) is left untouched.
//
// It adapts to whatever the exported graph declares - it does not assume an architecture:
//   - A stateless MLP exposes one input (the observation) and one output (actions); this class
//     then behaves exactly like the stock isaaclab::OrtRunner.
//   - A recurrent student additionally exposes carry-over state declared by the rsl-rl export
//     naming convention "<x>_in" (input) / "<x>_out" (output): a GRU has one pair (h), an LSTM
//     has two (h, c). Those inputs are not part of the observation - we hold them in internal
//     buffers, seed them with zeros, and feed each step's "<x>_out" back into the next step's
//     "<x>_in".
// Detection is driven entirely by the graph, so the same code path serves MLP / GRU / LSTM / etc.

#pragma once

#include "isaaclab/algorithms/algorithms.h"
#include "onnxruntime_cxx_api.h"
#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace b2w
{

class RecurrentOrtRunner : public isaaclab::Algorithms
{
public:
    RecurrentOrtRunner(std::string model_path)
    {
        // Init model
        env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "onnx_model");
        session_options.SetGraphOptimizationLevel(ORT_ENABLE_EXTENDED);
        session = std::make_unique<Ort::Session>(env, model_path.c_str(), session_options);

        // Inputs
        for (size_t i = 0; i < session->GetInputCount(); ++i) {
            Ort::TypeInfo input_type = session->GetInputTypeInfo(i);
            input_shapes.push_back(input_type.GetTensorTypeAndShapeInfo().GetShape());
            auto input_name = session->GetInputNameAllocated(i, allocator);
            input_names.push_back(input_name.release());
        }
        for (const auto& shape : input_shapes) {
            size_t size = 1;
            for (const auto& dim : shape) size *= dim;
            input_sizes.push_back(size);
        }

        // Outputs (actions, plus any carry-over state a recurrent student emits)
        for (size_t i = 0; i < session->GetOutputCount(); ++i) {
            Ort::TypeInfo output_type = session->GetOutputTypeInfo(i);
            auto shape = output_type.GetTensorTypeAndShapeInfo().GetShape();
            output_shapes.push_back(shape);
            size_t size = 1;
            for (const auto& dim : shape) size *= dim;
            output_sizes.push_back(size);
            auto output_name = session->GetOutputNameAllocated(i, allocator);
            output_names.push_back(output_name.release());
        }

        // Pair carry-over inputs to their outputs by the "<x>_in" / "<x>_out" convention. A
        // stateless graph has no such pairs, leaving the runner behaving as a plain feed-forward
        // policy.
        const std::string in_suffix = "_in";
        for (int i = 0; i < static_cast<int>(input_names.size()); ++i) {
            const std::string in_name(input_names[i]);
            if (in_name.size() <= in_suffix.size() ||
                in_name.compare(in_name.size() - in_suffix.size(), in_suffix.size(), in_suffix) != 0) {
                continue;  // not a carry-over input (e.g. the observation)
            }
            const std::string base = in_name.substr(0, in_name.size() - in_suffix.size());
            for (int j = 0; j < static_cast<int>(output_names.size()); ++j) {
                if (std::string(output_names[j]) == base + "_out") {
                    state_input_idx.push_back(i);
                    state_output_idx.push_back(j);
                    state_buffers.emplace_back(input_sizes[i], 0.0f);  // seed hidden state to zero
                    break;
                }
            }
        }

        // The action output is the one output that is not a carry-over feedback. Prefer an output
        // literally named "actions"; otherwise take the first non-state output.
        action_output_idx = first_non_state_output();
        for (int j = 0; j < static_cast<int>(output_names.size()); ++j) {
            if (std::string(output_names[j]) == "actions" && !is_state_output(j)) {
                action_output_idx = j;
                break;
            }
        }

        action.resize(output_sizes[action_output_idx]);
    }

    // Zero the carry-over state (safe no-op for stateless policies). Call when (re)starting a
    // policy episode so a recurrent student begins from the same state it was trained to reset to.
    void reset_state() override
    {
        std::lock_guard<std::mutex> lock(act_mtx_);
        for (auto& buf : state_buffers) std::fill(buf.begin(), buf.end(), 0.0f);
    }

    std::vector<float> act(std::unordered_map<std::string, std::vector<float>> obs) override
    {
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

        std::lock_guard<std::mutex> lock(act_mtx_);

        // Input tensors: observation inputs come from `obs`, carry-over inputs from buffers.
        std::vector<Ort::Value> input_tensors;
        input_tensors.reserve(input_names.size());
        for (int i = 0; i < static_cast<int>(input_names.size()); ++i) {
            const int slot = state_slot(i);
            float* data_ptr;
            if (slot >= 0) {
                data_ptr = state_buffers[slot].data();
            } else {
                const std::string name_str(input_names[i]);
                auto it = obs.find(name_str);
                if (it == obs.end()) {
                    throw std::runtime_error("Input name " + name_str + " not found in observations.");
                }
                data_ptr = it->second.data();
            }
            input_tensors.push_back(Ort::Value::CreateTensor<float>(
                memory_info, data_ptr, input_sizes[i], input_shapes[i].data(), input_shapes[i].size()));
        }

        // Run, requesting every output (actions + any next carry-over state).
        auto output_tensors = session->Run(
            Ort::RunOptions{nullptr}, input_names.data(), input_tensors.data(), input_tensors.size(),
            output_names.data(), output_names.size());

        // Copy the action output.
        auto* action_arr = output_tensors[action_output_idx].GetTensorMutableData<float>();
        std::memcpy(action.data(), action_arr, output_sizes[action_output_idx] * sizeof(float));

        // Carry each state output back into its input buffer for the next step.
        for (int k = 0; k < static_cast<int>(state_input_idx.size()); ++k) {
            auto* state_arr = output_tensors[state_output_idx[k]].GetTensorMutableData<float>();
            std::memcpy(state_buffers[k].data(), state_arr, input_sizes[state_input_idx[k]] * sizeof(float));
        }

        return action;
    }

private:
    int state_slot(int input_i) const
    {
        for (int k = 0; k < static_cast<int>(state_input_idx.size()); ++k)
            if (state_input_idx[k] == input_i) return k;
        return -1;
    }

    bool is_state_output(int output_j) const
    {
        for (int k : state_output_idx) if (k == output_j) return true;
        return false;
    }

    int first_non_state_output() const
    {
        for (int j = 0; j < static_cast<int>(output_names.size()); ++j)
            if (!is_state_output(j)) return j;
        return 0;
    }

    Ort::Env env;
    Ort::SessionOptions session_options;
    std::unique_ptr<Ort::Session> session;
    Ort::AllocatorWithDefaultOptions allocator;

    std::vector<const char*> input_names;
    std::vector<const char*> output_names;

    std::vector<std::vector<int64_t>> input_shapes;
    std::vector<int64_t> input_sizes;
    std::vector<std::vector<int64_t>> output_shapes;
    std::vector<int64_t> output_sizes;

    // Carry-over state plumbing (empty for stateless policies).
    std::vector<int> state_input_idx;               // indices into input_names that are carry-over inputs
    std::vector<int> state_output_idx;              // matching indices into output_names
    std::vector<std::vector<float>> state_buffers;  // current carry-over state, one per state input
    int action_output_idx = 0;
};

} // namespace b2w
