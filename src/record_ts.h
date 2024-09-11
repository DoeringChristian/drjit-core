#include "call.h"
#include "drjit-core/hash.h"
#include "drjit-core/jit.h"
#include "eval.h"
#include "internal.h"
#include "log.h"
#include "malloc.h"
#include "util.h"
#include "var.h"
#include <algorithm>
#include <cstdint>

// HashMap used to deduplicate variables
using PtrToSlot = tsl::robin_map<const void *, uint32_t, PointerHasher>;

enum class OpType {
    Barrier,
    KernelLaunch,
    MemsetAsync,
    Reduce,
    Expand,
    ReduceExpanded,
    PrefixSum,
    Compress,
    MemcpyAsync,
    Mkperm,
    Aggregate,
    Free,
    Count,
};

extern const char *op_type_name[(int) OpType::Count];

struct Operation {
    OpType type;
    // Indices into the dependencies vector
    std::pair<uint32_t, uint32_t> dependency_range;
    // Kernel hash if a kernel was launched
    union {
        Kernel kernel;
        ReduceOp rtype;
        bool exclusive;
        uint32_t bucket_count;
        uint64_t data;
    };
    size_t size;
    size_t input_size = 0;
    bool enabled = true;
    bool uses_optix = false;
    OptixShaderBindingTable *sbt;
};

/// Denotes the type of variable.
///
/// Output variables are only tracked through the outputs array, as this
/// information is only needed when constructing the output variables.
///
enum class RecordVarState {
    /// This variable was not initialized
    Uninit,
    /// This variable has been created by an operation
    OpOutput,
    /// This variable is part of the function input
    Input,
    /// This variable has been captured
    Captured,
};


/// Tracks how this variable was initialized
enum class RecordVarInit{
    None,
    Captured,
    Input,
};

struct RecordVariable {
    VarType type = VarType::Void;
    /// Stores index into input array if variable is input or index of captured
    /// variable
    uint32_t index = 0;
    /// Tracks the current state of a variable
    RecordVarState state = RecordVarState::Uninit;
    /// Tracks how this variable has been initialized
    RecordVarInit init = RecordVarInit::None;
    // used to deallocate unused variables during replay.
    uint32_t last_memset = 0;
    uint32_t last_memcpy = 0;

    const void *ptr;

    RecordVariable() {
    }

    /**
     * Not all information about variables might be known right away (see
     * memcpy). When new information about the variable is available, we can add
     * it to the already saved RecordVariable.
     */
    RecordVariable &operator|=(const RecordVariable &rhs) {
        if (this->state == RecordVarState::Uninit) {
            this->state = rhs.state;
            this->index = rhs.index;
        }
        if(rhs.last_memcpy)
            this->last_memcpy = rhs.last_memcpy;
        if(rhs.last_memset)
            this->last_memset = rhs.last_memset;
        return *this;
    }
};

struct ParamInfo {
    uint32_t slot;
    ParamType type = ParamType::Input;
    VarType vtype = VarType::Void;
    bool pointer_access = false;
    bool test_uninit = true;
    struct {
        uint32_t offset;
        uint64_t data;
        int32_t type_size;
    } extra;

    ParamInfo() {
    }
    ParamInfo(uint32_t index, VarType vtype) : slot(index), vtype(vtype) {
    }
    ParamInfo(uint32_t index, uint32_t vtype)
        : slot(index), vtype((VarType)vtype) {
    }
};

struct Recording {

    bool requires_dry_run = false;

    std::vector<RecordVariable> record_variables;

    std::vector<uint32_t> inputs;
    std::vector<ParamInfo> outputs;

    std::vector<Operation> operations;
    std::vector<ParamInfo> dependencies;

    JitBackend backend;

    int replay(const uint32_t *replay_input, uint32_t *outputs);
    uint32_t n_kernels = 0;


    void validate();
};

struct RecordThreadState : ThreadState {

    RecordThreadState(ThreadState *internal) {
        this->context = internal->context;
        this->stream = internal->stream;
        this->event = internal->event;
        this->sync_stream_event = internal->sync_stream_event;
        this->device = internal->device;
        this->compute_capability = internal->compute_capability;
        this->ptx_version = internal->ptx_version;
        this->memory_pool = internal->memory_pool;

        this->backend = internal->backend;
        this->scope = internal->scope;
        this->call_self_value = internal->call_self_value;
        this->call_self_index = internal->call_self_index;

#if defined(DRJIT_ENABLE_OPTIX)
        this->optix_pipeline = internal->optix_pipeline;
        this->optix_sbt = internal->optix_sbt;
#endif

        this->internal = internal;

        this->recording.backend = internal->backend;

        this->scope = internal->scope;
    };

    void barrier() override {
        if (!paused) {
            uint32_t start = this->recording.dependencies.size();

            Operation op;
            op.type = OpType::Barrier;
            op.dependency_range = std::pair(start, start);
            this->recording.operations.push_back(op);
        }

        scoped_pause();
        return this->internal->barrier();
    }

    Task *launch(Kernel kernel, uint32_t size,
                 std::vector<void *> *kernel_params,
                 const std::vector<uint32_t> *kernel_param_ids) override {
        if (!paused) {
            uint32_t kernel_param_offset =
                this->backend == JitBackend::CUDA ? 1 : 3;

            size_t input_size = 0;
            size_t ptr_size = 0;
            
            // Handle reduce_expanded case
            for (uint32_t param_index = 0;
                 param_index < kernel_param_ids->size(); param_index++) {
                uint32_t index = kernel_param_ids->at(param_index);
                Variable *v = jitc_var(index);
                ParamType param_type = (ParamType)v->param_type;
                if ((VarType)v->type == VarType::Pointer) {
                    jitc_log(LogLevel::Debug,
                             "pointer walking r%u points to r%u", index,
                             v->dep[3]);
                    // Follow pointer
                    index = v->dep[3];
                    v = jitc_var(index);
                }

                if (param_type == ParamType::Input && v->reduce_op) {
                    record_expand(index);
                }
            }

            jitc_log(LogLevel::Info, "record(): recording kernel %u", this->recording.n_kernels++);

            uint32_t start = this->recording.dependencies.size();
            for (uint32_t param_index = 0;
                 param_index < kernel_param_ids->size(); param_index++) {

                bool pointer_access = false;
                uint32_t index = kernel_param_ids->at(param_index);
                Variable *v = jitc_var(index);

                // Note, the ptr might not come from the variable but the
                // `ScheduledVariable` if it is an output.
                void *ptr =
                    kernel_params->at(kernel_param_offset + param_index);
                ParamType param_type = (ParamType)v->param_type;

                if (param_type == ParamType::Input &&
                    (VarType)v->type != VarType::Pointer) {
                    input_size = std::max(input_size, (size_t)v->size);
                }

                // In case the variable is a pointer, we follow the pointer to
                // the source and record the source size.
                // NOTE: this means that `v` is now the source variable
                if ((VarType)v->type == VarType::Pointer) {
                    jitc_assert(v->is_literal(),
                                "record(): Recording non-literal pointers are "
                                "not yet supported!");
                    jitc_assert(param_type != ParamType::Output,
                                "record(): A pointer, pointing to a kernel "
                                "ouptut is not yet supported!");

                    // Follow pointer
                    uint32_t ptr_index = index;
                    index = v->dep[3];
                    v = jitc_var(index);
                    if (v->data != ptr)
                        jitc_fail("record(): Tried to record variable r%u, "
                                  "pointing to r%u, but their memory address "
                                  "did not match! (%p != %p)",
                                  ptr_index, index, ptr, v->data);

                    pointer_access = true;
                    ptr_size = std::max(ptr_size, (size_t)v->size);
                }

                uint32_t slot;
                if (param_type == ParamType::Input){
                    if(has_variable(ptr)) {
                        slot = this->get_variable(ptr);
                    }else{
                        slot = capture_variable(index);
                    }

                }
                else if (param_type == ParamType::Output){
                    RecordVariable rv;
                    slot = this->add_variable(ptr, rv);
                }
                else
                    jitc_fail("Parameter Type not supported!");

                if (pointer_access) {
                    jitc_log(LogLevel::Debug,
                             " %s recording param %u = var(%u, points to r%u, "
                             "size=%u, data=%p, type=%s) at slot(%u)",
                             param_type == ParamType::Output ? "<-" : "->",
                             param_index, kernel_param_ids->at(param_index),
                             index, v->size, ptr, type_name[(uint32_t)v->type],
                             slot);
                } else {
                    jitc_log(LogLevel::Debug,
                             " %s recording param %u = var(%u, size=%u, "
                             "data=%p, type=%s) at slot(%u)",
                             param_type == ParamType::Output ? "<-" : "->",
                             param_index, kernel_param_ids->at(param_index),
                             v->size, ptr, type_name[(uint32_t)v->type], slot);
                }

                jitc_log(LogLevel::Debug, "    label=%s",
                         jitc_var_label(index));

                ParamInfo info;
                info.slot = slot;
                info.type = param_type;
                info.pointer_access = pointer_access;
                info.vtype = (VarType)v->type;
                add_param(info);
            }
            uint32_t end = this->recording.dependencies.size();

            Operation op;
            op.type = OpType::KernelLaunch;
            op.dependency_range = std::pair(start, end);
            op.kernel = kernel;
            op.size = size;
            if (uses_optix) {
                op.uses_optix = true;

                scoped_pause();
                // Copy SBT
                op.sbt = new OptixShaderBindingTable();
                std::memcpy(op.sbt, this->optix_sbt,
                            sizeof(OptixShaderBindingTable));

                // Copy hit groups
                size_t hit_group_size = optix_sbt->hitgroupRecordStrideInBytes *
                                        optix_sbt->hitgroupRecordCount;
                op.sbt->hitgroupRecordBase =
                    jitc_malloc(AllocType::Device, hit_group_size);
                jitc_memcpy(backend, op.sbt->hitgroupRecordBase,
                            optix_sbt->hitgroupRecordBase, hit_group_size);

                // Copy miss groups
                size_t miss_group_size = optix_sbt->missRecordStrideInBytes *
                                         optix_sbt->missRecordCount;
                op.sbt->missRecordBase =
                    jitc_malloc(AllocType::Device, miss_group_size);
                jitc_memcpy(backend, op.sbt->missRecordBase,
                            optix_sbt->missRecordBase, miss_group_size);
            }

            // Record max_input_size if we have only pointer inputs.
            // Therefore, if max_input_size > 0 we know this at replay.
            if (input_size == 0) {
                jitc_log(LogLevel::Info, "    input_size(pointers)=%zu",
                         ptr_size);
                op.input_size = ptr_size;
            } else {
                jitc_log(LogLevel::Info, "    input_size(direct)=%zu",
                         input_size);
                op.input_size = input_size;
            }

            // Reset input size if ratio/fraction is not valid
            if (op.input_size > 0) {
                if (op.size > op.input_size && op.size % op.input_size != 0)
                    op.input_size = 0;
                if (op.size < op.input_size && op.input_size % op.size != 0)
                    op.input_size = 0;
            }

            if(op.input_size){
                if(op.size > op.input_size)
                    jitc_log(LogLevel::Debug, "    size=input_size*%zu",
                             op.size / op.input_size);
                else if(op.size < op.input_size)
                    jitc_log(LogLevel::Debug, "    size=input_size/%zu",
                             op.input_size / op.size);
            }else{
                jitc_log(LogLevel::Debug,
                         "    input size could not be determined "
                         "by input and is recorded as is.");
            }

            this->recording.operations.push_back(op);

            // Re-assign optix specific variables to internal thread state since
            // they might have changed
#if defined(DRJIT_ENABLE_OPTIX)
            this->internal->optix_pipeline = this->optix_pipeline;
            this->internal->optix_sbt = this->optix_sbt;
#endif
        }
        scoped_pause();
        return this->internal->launch(kernel, size, kernel_params,
                                      kernel_param_ids);
    }

    /// Fill a device memory region with constants of a given type
    void memset_async(void *ptr, uint32_t size, uint32_t isize,
                      const void *src) override {

        if (!paused) {
            jitc_log(LogLevel::Debug,
                     "record(): memset_async(ptr=%p, size=%u, "
                     "isize=%u, src=%p)",
                     ptr, size, isize, src);
            jitc_assert(isize <= 8,
                        "record(): Tried to call memset_async with isize=%u, "
                        "only isize<=8 is supported!",
                        isize);

            RecordVariable rv;
            rv.last_memset = this->recording.operations.size() + 1;
            uint32_t ptr_id = this->add_variable(ptr, rv);

            uint32_t start = this->recording.dependencies.size();
            add_out_param(ptr_id, VarType::Void);
            uint32_t end = this->recording.dependencies.size();

            Operation op;
            op.type = OpType::MemsetAsync;
            op.dependency_range = std::pair(start, end);
            op.size = size;
            op.input_size = isize;
            std::memcpy(&op.data, src, isize);

            jitc_log(LogLevel::Debug, "record(): memset_async(ptr=s%u)", ptr_id);

            this->recording.operations.push_back(op);
        }

        scoped_pause();
        return this->internal->memset_async(ptr, size, isize, src);
    }

    /// Reduce the given array to a single value
    void reduce(VarType type, ReduceOp rtype, const void *ptr, uint32_t size,
                void *out) override {
        if (!paused) {

            uint32_t start = this->recording.dependencies.size();
            add_in_param(ptr);
            add_out_param(out, type);
            uint32_t end = this->recording.dependencies.size();

            Operation op;
            op.type = OpType::Reduce;
            op.dependency_range = std::pair(start, end);
            op.rtype = rtype;
            op.size = size;
            this->recording.operations.push_back(op);
        }

        scoped_pause();
        return this->internal->reduce(type, rtype, ptr, size, out);
    }

    /// 'All' reduction for boolean arrays
    bool all(uint8_t *values, uint32_t size) override {
        jitc_log(LogLevel::Warn,
                 "RecordThreadState::all(): unsupported function recording!");
        scoped_pause();
        return this->internal->all(values, size);
    }

    /// 'Any' reduction for boolean arrays
    bool any(uint8_t *values, uint32_t size) override {
        jitc_log(LogLevel::Warn,
                 "RecordThreadState::any(): unsupported function recording!");
        scoped_pause();
        return this->internal->any(values, size);
    }

    /// Exclusive prefix sum
    void prefix_sum(VarType vt, bool exclusive, const void *in, uint32_t size,
                    void *out) override {
        if (!paused) {
            uint32_t start = this->recording.dependencies.size();
            add_in_param(in);
            add_out_param(out, vt);
            uint32_t end = this->recording.dependencies.size();

            Operation op;
            op.type = OpType::PrefixSum;
            op.dependency_range = std::pair(start, end);
            op.exclusive = exclusive;
            op.size = size;
            this->recording.operations.push_back(op);
        }
        scoped_pause();
        return this->internal->prefix_sum(vt, exclusive, in, size, out);
    }

    /// Mask compression
    uint32_t compress(const uint8_t *in, uint32_t size,
                      uint32_t *out) override {

        if (!paused) {
            jitc_assert(has_variable(in),
                        "record(): Input variable has not been recorded!");
            jitc_log(LogLevel::Debug,
                     "record(): compress(in=%p, size=%u, out=%p)", in, size,
                     out);

            uint32_t start = this->recording.dependencies.size();
            add_in_param(in);
            add_out_param(out, VarType::UInt32);
            uint32_t end = this->recording.dependencies.size();

            Operation op;
            op.type = OpType::Compress;
            op.dependency_range = std::pair(start, end);
            op.size = size;
            this->recording.operations.push_back(op);
        }

        scoped_pause();
        return this->internal->compress(in, size, out);
    }

    /// Compute a permutation to reorder an integer array into discrete groups
    uint32_t mkperm(const uint32_t *values, uint32_t size,
                    uint32_t bucket_count, uint32_t *perm,
                    uint32_t *offsets) override {
        if (!paused) {
            if (has_variable(values)) {
                jitc_log(LogLevel::Debug,
                         "record(): mkperm(values=%p, size=%u, "
                         "bucket_count=%u, perm=%p, offsets=%p)",
                         values, size, bucket_count, perm, offsets);

                uint32_t start = this->recording.dependencies.size();
                add_in_param(values);
                add_out_param(perm, VarType::UInt32);
                add_out_param(offsets, VarType::UInt32);
                uint32_t end = this->recording.dependencies.size();

                Operation op;
                op.type = OpType::Mkperm;
                op.dependency_range = std::pair(start, end);
                op.size = size;
                op.bucket_count = bucket_count;
                this->recording.operations.push_back(op);
            }
        }
        scoped_pause();
        return this->internal->mkperm(values, size, bucket_count, perm,
                                      offsets);
    }

    /// Perform a synchronous copy operation
    void memcpy(void *dst, const void *src, size_t size) override {
        jitc_log(LogLevel::Debug,
                 "record(): memcpy(dst=%p, src=%p, size=%zu)", dst,
                 src, size);
        scoped_pause();
        return this->internal->memcpy(dst, src, size);
    }

    /// Perform an assynchronous copy operation
    void memcpy_async(void *dst, const void *src, size_t size) override {
        jitc_log(LogLevel::Debug,
                 "record(): memcpy_async(dst=%p, src=%p, size=%zu)", dst,
                 src, size);
        bool has_var = has_variable(src);
        if (!paused && (has_var)) {

            uint32_t src_id;
            src_id = this->get_variable(src);

            RecordVariable rv;
            rv.last_memcpy  = this->recording.operations.size() + 1;
            uint32_t dst_id = this->add_variable(dst, rv);

            uint32_t start = this->recording.dependencies.size();
            add_in_param(src_id);
            add_out_param(dst_id,
                          this->recording.record_variables[src_id].type);
            uint32_t end = this->recording.dependencies.size();

            Operation op;
            op.type             = OpType::MemcpyAsync;
            op.dependency_range = std::pair(start, end);
            op.size             = size;
            this->recording.operations.push_back(op);
        }
        {
            scoped_pause();
            this->internal->memcpy_async(dst, src, size);
        }
        if(!paused && !has_var){
            // If we did not know the source variable, this memcpy might be
            // coming from a `jitc_call_upload` call.
            // If that is the case, we have to capture the offset buffer.
            // Since the pointer might be used, for example by an aggregate call
            // (nested calls), we have to overwrite the RecordVariable.
            //
            CallData *call = nullptr;
            for (CallData *tmp : calls_assembled) {
                if (tmp->offset == dst) {
                    call = tmp;
                    break;
                }
            }
            if(call){
                capture_data(dst, size, VarType::UInt64, true, true);
                jitc_log(LogLevel::Debug, "    captured call offset");
            }
        }
    }

    /// Sum over elements within blocks
    void block_reduce(VarType type, ReduceOp op, const void *in, uint32_t size,
                      uint32_t block_size, void *out) override {
        jitc_log(LogLevel::Warn, "RecordThreadState::block_reduce(): "
                                 "unsupported function recording!");
        scoped_pause();
        return this->internal->block_reduce(type, op, in, size, block_size,
                                            out);
    }

    /// Compute a dot product of two equal-sized arrays
    void reduce_dot(VarType type, const void *ptr_1, const void *ptr_2,
                    uint32_t size, void *out) override {
        jitc_log(
            LogLevel::Warn,
            "RecordThreadState::reduce_dot(): unsupported function recording!");
        scoped_pause();
        return this->internal->reduce_dot(type, ptr_1, ptr_2, size, out);
    }

    /// Asynchronously update a single element in memory
    void poke(void *dst, const void *src, uint32_t size) override {
        jitc_log(LogLevel::Warn,
                 "RecordThreadState::poke(): unsupported function recording!");
        scoped_pause();
        return this->internal->poke(dst, src, size);
    }

    void aggregate(void *dst, AggregationEntry *agg, uint32_t size) override {
        if (!paused) {
            jitc_log(LogLevel::Debug, "record(): aggregate(dst=%p, size=%u)",
                     dst, size);

            uint32_t dst_id = this->add_variable(dst, RecordVariable{});

            jitc_log(LogLevel::Debug, " <- slot(%u)", dst_id);

            uint32_t start = this->recording.dependencies.size();

            ParamInfo info;
            info.type = ParamType::Output;
            info.slot = dst_id;
            info.pointer_access = false;
            info.vtype = VarType::UInt8;
            add_param(info);

            for (uint32_t i = 0; i < size; ++i) {
                AggregationEntry &p = agg[i];

                // There are three cases, we might have to handle.
                // 1. The input is a pointer (size = 8 and it is known to the malloc cache)
                // 2. The input is an evaluated variable (size < 0)
                // 3. The variabel is a literal (size > 0 and it is not a
                // pointer to a known allocation).

                bool is_ptr;
                auto it = state.alloc_used.find((uintptr_t)p.src);
                if(it == state.alloc_used.end())
                    is_ptr = false;
                else
                    is_ptr = true;

                if ((p.size == 8 && is_ptr) || p.size < 0) {
                    // Pointer or evaluated

                    bool has_var = has_variable(p.src);

                    if(!has_var){
                        jitc_log(LogLevel::Debug, "    deferring capture");
                    }
                    // NOTE: Offset buffers of nested calls might be used by
                    // this aggregate call, before the offset buffer is
                    // uploaded.
                    // We therefore defer the offset buffer capture to the
                    // memcpy_async operation.
                    uint32_t slot = add_variable(p.src, RecordVariable());
                            
                    jitc_log(LogLevel::Debug, "    var at slot s%u", slot);

                    ParamInfo info;
                    info.slot = slot;
                    info.type = ParamType::Input;
                    info.pointer_access = p.size == 8;
                    info.extra.offset = p.offset;
                    info.test_uninit = false;
                    add_param(info);
                    
                    jitc_log(LogLevel::Debug, "    entry(src=%p, size=%i, offset=%u)", p.src,
                             p.size, p.offset);
                } else {
                    // Literal
                    ParamInfo info;
                    std::memcpy(&info.extra.data, &p.src, sizeof(uint64_t));
                    info.extra.offset = p.offset;
                    info.extra.type_size = p.size;
                    info.type = ParamType::Register;
                    info.pointer_access = false;
                    add_param(info);
                    
                    jitc_log(LogLevel::Debug, "    literal");
                }
            }

            uint32_t end = this->recording.dependencies.size();

            Operation op;
            op.type = OpType::Aggregate;
            op.dependency_range = std::pair(start, end);
            op.size = size;
            this->recording.operations.push_back(op);
        }
        scoped_pause();
        this->internal->aggregate(dst, agg, size);
    }

    // Enqueue a function to be run on the host once backend computation is done
    void enqueue_host_func(void (*callback)(void *), void *payload) override {
        scoped_pause();
        return this->internal->enqueue_host_func(callback, payload);
    }

    /// LLVM: reduce a variable that was previously expanded due to
    /// dr.ReduceOp.Expand
    void reduce_expanded(VarType vt, ReduceOp reduce_op, void *data,
                         uint32_t exp, uint32_t size) override {

        if (!paused) {
            jitc_log(LogLevel::Debug,
                     "record(): reduce_expanded(vt=%u, op=%u, data=%p, exp=%u, "
                     "size=%u)",
                     (uint32_t)vt, (uint32_t)reduce_op, data, exp, size);

            uint32_t data_id = this->add_variable(data, RecordVariable{});

            uint32_t start = this->recording.dependencies.size();
            add_out_param(data_id, vt);
            uint32_t end = this->recording.dependencies.size();

            jitc_log(LogLevel::Debug, "<-> data: slot(%u)", data_id);

            Operation op;
            op.type = OpType::ReduceExpanded;
            op.dependency_range = std::pair(start, end);
            op.rtype = reduce_op;
            op.size = size;
            this->recording.operations.push_back(op);
        }
        scoped_pause();
        return this->internal->reduce_expanded(vt, reduce_op, data, exp, size);
    }

    void notify_free(const void *ptr) override{
        if(has_variable(ptr)){
            jitc_log(LogLevel::Debug, "record(): jitc_free(ptr=%p)", ptr);
            
            uint32_t start = this->recording.dependencies.size();
            add_in_param(ptr, false);
            uint32_t end = this->recording.dependencies.size();

            Operation op;
            op.type = OpType::Free;
            op.dependency_range = std::pair(start, end);

            this->ptr_to_slot.erase(ptr);
        }
    }

    ~RecordThreadState() {
    }

    void add_input(uint32_t input) {
        uint32_t input_index = this->recording.inputs.size();
        Variable *v = jitc_var(input);
        RecordVariable rv;
        rv.state = RecordVarState::Input;
        rv.init = RecordVarInit::Input;
        rv.index = input_index;
        rv.type = (VarType)v->type;
        uint32_t slot = this->add_variable(v->data, rv);
        jitc_log(LogLevel::Info,
                 "record(): Adding variable %u <%p> input %u to slot s%u", input,
                 v->data, input_index, slot);
        this->recording.inputs.push_back(slot);
    }
    void add_output(uint32_t output) {
        uint32_t output_index = this->recording.outputs.size();
        Variable *v = jitc_var(output);
        uint32_t slot;
        if (!has_variable(v->data)) {
            slot = capture_variable(output);
        } else {
            slot = this->get_variable(v->data);
        }

        jitc_log(LogLevel::Trace,
                 "record(): Adding variable %u output %u to slot s%u", output,
                 output_index, slot);
        ParamInfo info;
        info.slot = slot;
        info.vtype = (VarType)v->type;
        this->recording.outputs.push_back(info);
    }

    bool pause() {
        bool tmp = paused;
        paused = true;
        return tmp;
    }
    bool resume() {
        bool tmp = paused;
        paused = false;
        return tmp;
    }

    struct pause_scope {
        RecordThreadState *rts;
        bool tmp;

        pause_scope(RecordThreadState *rts) : rts(rts), tmp(rts->pause()) {
        }
        ~pause_scope() {
            rts->paused = tmp;
        }
    };

    pause_scope scoped_pause() {
        return pause_scope(this);
    }

    bool paused = false;

    ThreadState *internal;

    Recording recording;

  private:
    // Mapping from data pointer of a variable to a index into the slot of the
    // recording.
    PtrToSlot ptr_to_slot;

    /**
     * Record the Expand operation, corresponding to the `jitc_var_expand` call,
     * whith which the variable `index` has been expanded.
     * This is called before a kernel is recorded.
     */
    void record_expand(uint32_t index) {
        Variable *v              = jitc_var(index);
        
        uint32_t dst_slot        = get_variable(v->data);
        const RecordVariable &rv = this->recording.record_variables[dst_slot];
        if (rv.last_memset == 0)
            jitc_fail(
                "record(): Could not infer last memset operation of r%u s%u, "
                "to construct expand operation!",
                index, dst_slot);
        Operation &memset = this->recording.operations[rv.last_memset - 1];
        memset.enabled    = false;

        Operation op;
        uint32_t start = this->recording.dependencies.size();
        add_out_param(dst_slot, v->type);
        if (rv.last_memcpy) {
            Operation &memcpy = this->recording.operations[rv.last_memcpy - 1];
            memcpy.enabled    = false;

            uint32_t dependency_index = memcpy.dependency_range.first;
            ParamInfo src_info = this->recording.dependencies[dependency_index];

            add_in_param(src_info.slot);

            jitc_log(LogLevel::Debug, "record(): expand(dst=s%u, src=s%u)",
                     dst_slot, src_info.slot);

            op.size = memcpy.size / type_size[(uint32_t) src_info.type];
        } else {
            // Case where in jitc_var_expand, v->is_literal && v->literal ==
            // identity
            uint64_t identity = jitc_reduce_identity((VarType) v->type,
                                                     (ReduceOp) v->reduce_op);

            jitc_log(LogLevel::Debug,
                     "record(): expand(dst=s%u, src=literal 0x%lx)", dst_slot,
                     identity);

            op.size = v->size;
        }
        uint32_t end = this->recording.dependencies.size();

        op.type             = OpType::Expand;
        op.dependency_range = std::pair(start, end);
        op.data             = memset.data;
        this->recording.operations.push_back(op);

        this->recording.requires_dry_run = true;
    }

    uint32_t capture_data(const void *ptr, size_t dsize, VarType vt = VarType::UInt8, bool remember = false, bool overwrite = false){
        if (!dsize){
            dsize = jitc_malloc_size(ptr);
        }

        uint32_t size;
        size = dsize / type_size[(uint32_t)vt];

        uint32_t data = jitc_var_mem_map(backend, vt, (void*)ptr, size, false);

        return capture_variable(data, ptr, remember, false, overwrite);
    }

    /**
     * This function tries to capture a variable that is not known to the
     * recording threadstate.
     */
    uint32_t capture_variable(uint32_t index, const void *ptr = nullptr,
                              bool remember = true, bool test_scope = true, bool overwrite = false) {

        scoped_pause();
        Variable *v = jitc_var(index);
        if (v->scope < this->internal->scope && test_scope) {
            jitc_raise(
                "record(): Variable r%u[%u] -> %p, label=%s, was created "
                "before recording was started, but it was "
                "not speciefied as an input variable!",
                index, v->size, v->data, jitc_var_label(index));
        }

        // Might make sense to limit the size of captured variables to 1.
        // if (v->size > 1)
        //     jitc_raise("record(): Variable r%u[%u] -> %p, label=%s, data=%s,
        //     "
        //                "of size > 1 was created while recording.",
        //                index, v->size, v->data, jitc_var_label(index),
        //                jitc_var_str(index));

        if (!ptr)
            ptr = v->data;

        // Have to copy the variable, so that it cannot be modified by other
        // calls later.
        jitc_log(LogLevel::Debug,
                 "record(): capturing variable r%u, type=%s, data=%s", index,
                 type_name[(uint32_t) v->type], jitc_var_str(index));

        AllocType atype = backend == JitBackend::CUDA ? AllocType::Device
                                                      : AllocType::HostAsync;
        size_t dsize    = v->size * type_size[(uint32_t) v->type];

        uint64_t *data = (uint64_t *) jitc_malloc(atype, dsize);
        jitc_memcpy(backend, data, ptr, dsize);

        uint32_t data_var = jitc_var_mem_map(backend, (VarType) v->type, data,
                                             (size_t) v->size, 1);

        RecordVariable rv;
        rv.ptr = ptr;
        rv.state = RecordVarState::Captured;
        rv.init  = RecordVarInit::Captured;
        rv.index = data_var;

        // Add the frozen value to the ptr_to_slot map
        // NOTE: this only works if we are using notify_free.
        // But it could allow us to capture call offsets here.
        uint32_t slot;
        if(overwrite){
            auto it = this->ptr_to_slot.find(ptr);

            if (it == this->ptr_to_slot.end()) {
                slot = this->recording.record_variables.size();

                this->recording.record_variables.push_back(rv);

                this->ptr_to_slot.insert({ ptr, slot });
            } else {
                slot = it.value();
                jitc_log(LogLevel::Debug, "    overwriting at s%u", slot);
                this->recording.record_variables[slot] = rv;
            }
        }
        else{
            slot = this->recording.record_variables.size();
            this->recording.record_variables.push_back(rv);
            if (remember) {
                auto it = this->ptr_to_slot.find(ptr);
                if (it == this->ptr_to_slot.end())
                    this->ptr_to_slot.insert({ ptr, slot });
                else
                    it.value() = slot;
            }
        }

        jitc_log(LogLevel::Debug, "    at slot s%u", slot);

        return slot;
    }

    /**
     * Add information about a variable, deduplicating it and returning the slot
     * in the `variables` field of the recording.
     * Information is combined when the variable has already been added.
     * This is used by the input variables of a kernel.
     */
    uint32_t add_variable(const void *ptr, RecordVariable rv) {

        rv.ptr = ptr;
        auto it = this->ptr_to_slot.find(ptr);

        if (it == this->ptr_to_slot.end()) {
            uint32_t slot = this->recording.record_variables.size();

            this->recording.record_variables.push_back(rv);

            this->ptr_to_slot.insert({ptr, slot});

            return slot;
        } else {
            uint32_t slot = it.value();

            this->recording.record_variables[slot] |= rv;

            return slot;
        }
    }

    // Return the slot index given the data pointer of a variable.
    // This fails if the variable has not been added.
    uint32_t get_variable(const void *ptr) {
        auto it = this->ptr_to_slot.find(ptr);

        if(it == this->ptr_to_slot.end())
            jitc_fail("Failed to find the slot corresponding to the variable "
                      "with data at %p",
                      ptr);

        return it.value();
    }

    bool has_variable(const void *ptr) {
        auto it = this->ptr_to_slot.find(ptr);

        return it != this->ptr_to_slot.end();
    }

    void add_param(ParamInfo info) {
        
        RecordVariable &rv = this->recording.record_variables[info.slot];
        if (info.type == ParamType::Output){

            jitc_log(LogLevel::Debug, " <- param s%u", info.slot);
            
            if(info.vtype != VarType::Void)
                rv.type = info.vtype;
            
            rv.state = RecordVarState::OpOutput;
            
        }else if (info.type == ParamType::Input){
            
            jitc_log(LogLevel::Debug, " -> param s%u", info.slot);
            
            if(info.test_uninit && rv.state == RecordVarState::Uninit)
                jitc_fail("record(): Varaible at slot s%u was read from by "
                          "operation o%u, but has not yet been initialized!",
                          info.slot,
                          (uint32_t) this->recording.operations.size());
            
            if (info.vtype == VarType::Void)
                info.vtype = rv.type;
            
        }
        
        this->recording.dependencies.push_back(info);
    }
    void add_in_param(uint32_t slot, bool test_uninit = true) {
        ParamInfo info;
        info.type = ParamType::Input;
        info.slot = slot;
        info.test_uninit = test_uninit;
        add_param(info);
    }
    void add_in_param(const void *ptr, bool test_uninit = true){
        uint32_t slot = this->get_variable(ptr);
        add_in_param(slot, test_uninit);
    }
    void add_out_param(uint32_t slot, VarType vtype) {
        ParamInfo info;
        info.type = ParamType::Output;
        info.slot = slot;
        info.vtype = vtype;
        add_param(info);
    }
    void add_out_param(const void *ptr, VarType vtype){
        RecordVariable rv;
        uint32_t slot = this->add_variable(ptr, rv);
        add_out_param(slot, vtype);
    }
    void add_out_param(uint32_t slot, uint32_t vtype) {
        add_out_param(slot, (VarType)vtype);
    }
};

void jitc_record_start(JitBackend backend, const uint32_t *inputs,
                       uint32_t n_inputs);

Recording *jitc_record_stop(JitBackend backend, const uint32_t *outputs,
                            uint32_t n_outputs);

void jitc_record_abort(JitBackend backend);

void jitc_record_destroy(Recording *recording);

bool jitc_record_pause(JitBackend backend);

bool jitc_record_resume(JitBackend backend);

void jitc_record_replay(Recording *recording, const uint32_t *inputs,
                        uint32_t *outputs);

int jitc_record_dry_run(Recording *recording, const uint32_t *inputs,
                        uint32_t *outputs);
