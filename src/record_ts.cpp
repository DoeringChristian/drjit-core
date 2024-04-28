#include "record_ts.h"
#include "common.h"
#include "drjit-core/jit.h"
#include "internal.h"
#include "log.h"

void Recording::replay(const uint32_t *replay_input, uint32_t *outputs){
    
    ThreadState *ts = thread_state(backend);

    // This struct holds the data and tracks the size of varaibles, 
    // used during replay.
    struct ReplayVariable{
        void *data;
        uint32_t size;
    };

    std::vector<ReplayVariable> variables(this->variables.size());
    std::vector<void *> kernel_params;
    std::vector<Task*> scheduled_tasks;

    // Populate with input variables
    for (uint32_t i = 0; i < this->inputs.size(); ++i){
        Variable* input_variable = jitc_var(replay_input[i]);
        ReplayVariable &replay_variable = variables[this->inputs[i]];
        replay_variable.size = input_variable->size;
        replay_variable.data = input_variable->data;
    }

    // Execute kernels and allocate missing output variables

    for (uint32_t i = 0; i < this->operations.size(); ++i){
        Operation& op = this->operations[i];

        switch (op.type) {
            case OpType::KernelLaunch:
                kernel_params.clear();

                if (backend == JitBackend::CUDA) {
                    uintptr_t size = 0;
                    std::memcpy(&size, &op.size, sizeof(uint32_t));
                    kernel_params.push_back((void *) size);
                } else {
                    // First 3 parameters reserved for: kernel ptr, size, ITT identifier
                    for (int i = 0; i < 3; ++i)
                        kernel_params.push_back(nullptr);
                }

                // Allocate Missing variables for kernel launch.
                // The assumption here is that for every kernel launch, the inputs are already allocated.
                // Therefore we only allocate output variables, which have the same size as the kernel.
                // TODO: deallocate unused memory.
                for (uint32_t j = op.dependency_range.first; j < op.dependency_range.second; ++j){
                    ReplayVariable &replay_variable = variables[this->dependencies[j]];
                    if (replay_variable.data == nullptr){
                        jitc_log(LogLevel::Info, "Allocating output variable of size %zu.", op.size);

                        RecordVariable &rv = this->variables[this->dependencies[j]];
                        uint32_t dsize = op.size * type_size[(int) rv.type];

                        AllocType alloc_type = this->backend == JitBackend::CUDA ? AllocType::Device : AllocType::Host;

                        replay_variable.data = jitc_malloc(alloc_type, dsize);
                        replay_variable.size = op.size;
                    }
                    kernel_params.push_back(replay_variable.data);
                }

                {
                    scoped_set_context_maybe guard2(ts->context);
                    std::vector<uint32_t> tmp;
                    scheduled_tasks.push_back(ts->launch(op.kernel, op.size, &kernel_params, &tmp));
                }

                break;
            case OpType::Barrier:

                if (this->backend == JitBackend::LLVM) {
                    if (scheduled_tasks.size() == 1) {
                        task_release(jitc_task);
                        jitc_task = scheduled_tasks[0];
                    } else {
                        jitc_assert(!scheduled_tasks.empty(),
                                    "jit_eval(): no tasks generated!");

                        // Insert a barrier task
                        Task *new_task = task_submit_dep(nullptr, scheduled_tasks.data(),
                                                         (uint32_t) scheduled_tasks.size());
                        task_release(jitc_task);
                        for (Task *t : scheduled_tasks)
                        task_release(t);
                        jitc_task = new_task;
                    }
                }

                break;
            default:
                jitc_fail("An operation has been recorded, that is not known to the replay functionality!");
                break;
        }
    }

    for(uint32_t i = 0; i < this->outputs.size(); ++i){
        uint32_t index = this->outputs[i];
        ReplayVariable &rv = variables[index];
        RecordVariable &record_variable = this->variables[index];
        Variable v;
        v.kind = VarKind::Evaluated;
        v.type = (uint32_t) record_variable.type;
        v.size = rv.size;
        v.data = rv.data;
        v.backend = (uint32_t) this->backend;
        outputs[i] = jitc_var_new(v);
    }
}

void jitc_record_start(JitBackend backend, const uint32_t *inputs,
                       uint32_t n_inputs) {

    ThreadState *ts = thread_state(backend);
    RecordThreadState *record_ts = new RecordThreadState(ts);

    if (backend == JitBackend::CUDA) {
        thread_state_cuda = record_ts;
    } else {
        thread_state_llvm = record_ts;
    }

    for (uint32_t i = 0; i < n_inputs; ++i) {
        record_ts->set_input(inputs[i]);
    }
}
Recording *jitc_record_stop(JitBackend backend, const uint32_t *outputs,
                                    uint32_t n_outputs) {
    ThreadState *ts = thread_state(backend);
    RecordThreadState *record_ts = dynamic_cast<RecordThreadState *>(ts);

    for (uint32_t i = 0; i < n_outputs; ++i) {
        record_ts->set_output(outputs[i]);
    }

    if (backend == JitBackend::CUDA) {
        thread_state_cuda = record_ts->internal;
    } else {
        thread_state_llvm = record_ts->internal;
    }
    return new Recording(record_ts->recording);
}
