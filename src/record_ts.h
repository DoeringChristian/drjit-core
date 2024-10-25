#include "call.h"
#include "drjit-core/hash.h"
#include "drjit-core/jit.h"
#include "eval.h"
#include "internal.h"
#include "log.h"
#include "malloc.h"
#include "util.h"
#include "var.h"
#include <cstdint>

/// HashMap, mapping an allocation to a recorded variable
using PtrToSlot = tsl::robin_map<const void *, uint32_t, PointerHasher>;

enum class OpType {
    Barrier,
    KernelLaunch,
    MemsetAsync,
    Expand,
    ReduceExpanded,
    Compress,
    MemcpyAsync,
    Mkperm,
    BlockReduce,
    BlockPrefixReduce,
    Aggregate,
    Free,
    Count,
};

extern const char *op_type_name[(int) OpType::Count];

/**
 * Represents an operation, that was recorded by a \ref RecordThreadState.
 *
 * To record an operation the first step is to get the current size of the
 * dependencies vector.
 * uint32_t start = m_recording.dependencies.size();
 *
 * Then, the parameters can be added.
 * add_out_param(ptr_id, VarType::Void);
 *
 * Finally, the size after adding the parameters can be read and the operation
 * can be pushed.
 * uint32_t end = m_recording.dependencies.size();
 *
 * Operation op;
 * op.dependency_range = std::pair(start, end);
 */
struct Operation {
    OpType type;
    /// Indices into the dependencies vector
    std::pair<uint32_t, uint32_t> dependency_range;
    union {
        /// Additional information of a kernel launch
        struct {
            KernelKey *key;
            Kernel kernel;
            XXH128_hash_t hash;
        } kernel;
        /// The reduce type of a block reduction operation
        ReduceOp rtype;
        struct {
            /// The reduce type of a prefix reduction operation
            ReduceOp rtype;
            /// Weather a prefix sum operation is exclusive
            bool exclusive;
            bool reverse;
        } prefix_reduce;
        /// The bucket count for the mkperm operation
        uint32_t bucket_count;
        /// Additional data such as the source of memset
        uint64_t data;
    };
    /// Records the size of the operation.
    size_t size;
    /// Records the size of the largest input variable (directly accessed or
    /// through a pointer if the kernel has no direct inputs).
    size_t input_size = 0;
    /// Weather this operation is enabled. We might have to disable some
    /// operations after the fact, and removing them from the Recording would be
    /// more complicated.
    bool enabled = true;
    /// Does this operation use optix?
    bool uses_optix = false;
    /// A copy of the shader binding table, used by the kernel.
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

/// Records how this variable was initialized
enum class RecordVarInit {
    None,
    Captured,
    Input,
};

/**
 * \brief Represents a recorded variable.
 *
 * An evaluated variable is tracked by the memory it refers to.
 * This struct records the memory region from the time it was first used in one
 * of the operations, until it is freed by `jit_free`.
 * The memory also has to be allocated using `jit_malloc`, otherwise it cannot
 * be tracked.
 */
struct RecordVariable {
    /// Stores index into input array if variable is input or index of captured
    /// variable
    uint32_t index = 0;
    /// Records how this variable has been initialized
    RecordVarInit init = RecordVarInit::None;

    /// Tracks the last memset and memcpy operations necessary for recording the
    /// expand operation.
    uint32_t last_memset = 0;
    uint32_t last_memcpy = 0;

    /// Tracks the current state of a variable
    RecordVarState state = RecordVarState::Uninit;
    /// Tracks the current type of the variable
    VarType type = VarType::Void;
    /// Tracks the pointer of the variable for debug purposes
    const void *ptr;

    RecordVariable() {}

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
        if (rhs.last_memcpy)
            this->last_memcpy = rhs.last_memcpy;
        if (rhs.last_memset)
            this->last_memset = rhs.last_memset;
        return *this;
    }
};

/**
 * This represents how a variable is accessed by an operation.
 */
struct ParamInfo {
    /// References the variable in \ref Recording.record_variables that is
    /// accessed.
    uint32_t slot;
    /// Records how the variable was accessed i.e. was it the output or input
    /// of/to a kernel.
    ParamType type = ParamType::Input;
    /// The variable type as which the access occurred. Different operations
    /// might reinterpret the same allocation as different types this changes
    /// the inferred launch size of the operation.
    /// For example, a memcpy operation interprets the allocation as `UInt8`
    /// types, whereas a kernel interprets it as `UInt64`.
    VarType vtype = VarType::Void;
    /// Was this allocation accessed through a pointer?
    bool pointer_access = false;
    /// Should the next input operation fail, if the variable is still
    /// uninitialized?
    bool test_uninit = true;
    struct {
        /// Represents the offset of that parameter for aggregate operations.
        uint32_t offset;
        /// Represents some literal data when aggregating literals.
        uint64_t data;
        /// The type size, when the actual type is not known. For example for
        /// the aggregate operation.
        int32_t type_size;
    } extra;

    ParamInfo() {}
    ParamInfo(uint32_t index, VarType vtype) : slot(index), vtype(vtype) {}
    ParamInfo(uint32_t index, uint32_t vtype)
        : slot(index), vtype((VarType) vtype) {}
};

/**
 * \brief Represents a frozen function recording. And can be used to replay it.
 */
struct Recording {
    /// Weather this recording requires a dryrun, to discover the size of
    /// certain operations.
    bool requires_dry_run = false;

    /// The variables used in this recording.
    /// Each variable refers to an allocation.
    /// If an allocation reuses a memory region, it is referred to by a separate
    /// variable.
    std::vector<RecordVariable> record_variables;

    /// This vector maps the flat and deduplicated inputs to the frozen to their
    /// variables in the \ref record_variables array.
    std::vector<uint32_t> inputs;
    /// This vector maps the flat outputs of the frozen function to their
    /// recorded variables and how they have been accessed.
    std::vector<ParamInfo> outputs;

    /// Records the operations performed by this frozen function recording.
    std::vector<Operation> operations;
    /// Every operation refers to some number of variables, and encodes how they
    /// are accessed. Instead of allocating a vector for each operation, \ref
    /// Operation struct contains a pair that indexes into this vector.
    std::vector<ParamInfo> dependencies;

    /// The backend, which was used while recording.
    JitBackend backend;

    /// Replays the recording, given some inputs and fills the outputs with the
    /// created variable indices.
    /// Note that both \ref replay_input and \replay output have to have the
    /// same size as the number of inputs and outputs with which the frozen
    /// function was recorded.
    int replay(const uint32_t *replay_input, uint32_t *outputs);

    /// Counter, counting the number of kernels for debugging.
    uint32_t n_kernels = 0;

    /// This function is called after recording and checks that the recording is
    /// valid i.e. that no variables where left uninitialized.
    void validate();
    /// Checks if all recorded kernels are still in the kernel cache.
    /// This might occur when calling dr.kernel_cache_flush between recording
    /// the function and replaying it.
    bool check_kernel_cache();
};

/**
 * \brief This struct is a wrapper arround a \ref ThreadState that records the
 * operations performed with it.
 */
struct RecordThreadState : ThreadState {

    RecordThreadState(ThreadState *internal) {
        this->context            = internal->context;
        this->stream             = internal->stream;
        this->event              = internal->event;
        this->sync_stream_event  = internal->sync_stream_event;
        this->device             = internal->device;
        this->compute_capability = internal->compute_capability;
        this->ptx_version        = internal->ptx_version;
        this->memory_pool        = internal->memory_pool;

        this->backend         = internal->backend;
        this->scope           = internal->scope;
        this->call_self_value = internal->call_self_value;
        this->call_self_index = internal->call_self_index;

#if defined(DRJIT_ENABLE_OPTIX)
        this->optix_pipeline = internal->optix_pipeline;
        this->optix_sbt      = internal->optix_sbt;
#endif

        this->m_internal = internal;

        this->m_recording.backend = internal->backend;

        this->scope = internal->scope;
    };

    void barrier() override {
        if (!paused()) {
            uint32_t start = this->m_recording.dependencies.size();

            Operation op;
            op.type             = OpType::Barrier;
            op.dependency_range = std::pair(start, start);
            this->m_recording.operations.push_back(op);
        }

        scoped_pause();
        return this->m_internal->barrier();
    }

    Task *launch(Kernel kernel, KernelKey *key, XXH128_hash_t hash,
                 uint32_t size, std::vector<void *> *kernel_params,
                 const std::vector<uint32_t> *kernel_param_ids) override {
        if (!paused()) {
            try {
                record_launch(kernel, key, hash, size, kernel_params,
                              kernel_param_ids);
            } catch (...) {
                record_exception();
            }
        }
        scoped_pause();
        return this->m_internal->launch(kernel, key, hash, size, kernel_params,
                                        kernel_param_ids);
    }

    /// Fill a device memory region with constants of a given type
    void memset_async(void *ptr, uint32_t size, uint32_t isize,
                      const void *src) override {
        if (!paused()) {
            try {
                record_memset_async(ptr, size, isize, src);
            } catch (...) {
                record_exception();
            }
        }
        scoped_pause();
        return this->m_internal->memset_async(ptr, size, isize, src);
    }

    /// Mask compression
    uint32_t compress(const uint8_t *in, uint32_t size,
                      uint32_t *out) override {
        if (!paused()) {
            try {
                record_compress(in, size, out);
            } catch (...) {
                record_exception();
            }
        }
        scoped_pause();
        return this->m_internal->compress(in, size, out);
    }

    /// Compute a permutation to reorder an integer array into discrete groups
    uint32_t mkperm(const uint32_t *values, uint32_t size,
                    uint32_t bucket_count, uint32_t *perm,
                    uint32_t *offsets) override {
        if (!paused()) {
            try {
                record_mkperm(values, size, bucket_count, perm, offsets);
            } catch (...) {
                record_exception();
            }
        }
        scoped_pause();
        return this->m_internal->mkperm(values, size, bucket_count, perm,
                                        offsets);
    }

    /// Perform a synchronous copy operation
    void memcpy(void *dst, const void *src, size_t size) override {
        jitc_log(LogLevel::Debug, "record(): memcpy(dst=%p, src=%p, size=%zu)",
                 dst, src, size);
        scoped_pause();
        return this->m_internal->memcpy(dst, src, size);
    }

    /// Perform an asynchronous copy operation
    void memcpy_async(void *dst, const void *src, size_t size) override {
        jitc_log(LogLevel::Debug,
                 "record(): memcpy_async(dst=%p, src=%p, size=%zu)", dst, src,
                 size);
        bool has_var = has_variable(src);
        if (!paused() && (has_var)) {

            uint32_t src_id;
            src_id = this->get_variable(src);

            RecordVariable rv;
            rv.last_memcpy  = this->m_recording.operations.size() + 1;
            uint32_t dst_id = this->add_variable(dst, rv);

            uint32_t start = this->m_recording.dependencies.size();
            add_in_param(src_id);
            add_out_param(dst_id,
                          this->m_recording.record_variables[src_id].type);
            uint32_t end = this->m_recording.dependencies.size();

            Operation op;
            op.type             = OpType::MemcpyAsync;
            op.dependency_range = std::pair(start, end);
            op.size             = size;
            this->m_recording.operations.push_back(op);
        }
        {
            scoped_pause();
            this->m_internal->memcpy_async(dst, src, size);
        }
        if (!paused() && !has_var) {
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
            if (call) {
                capture_call_offset(dst, size);
                jitc_log(LogLevel::Debug, "    captured call offset");
            }
        }
    }

    /// Sum over elements within blocks
    void block_reduce(VarType vt, ReduceOp op, uint32_t size,
                      uint32_t block_size, const void *in, void *out) override {
        if (!paused()) {
            try {
                record_block_reduce(vt, op, size, block_size, in, out);
            } catch (...) {
                record_exception();
            }
        }
        scoped_pause();
        return this->m_internal->block_reduce(vt, op, size, block_size, in,
                                              out);
    }

    void block_prefix_reduce(VarType vt, ReduceOp op, uint32_t size,
                             uint32_t block_size, bool exclusive, bool reverse,
                             const void *in, void *out) override {
        if (!paused()) {
            try {
                record_block_prefix_reduce(vt, op, size, block_size, exclusive,
                                           reverse, in, out);
            } catch (...) {
                record_exception();
            }
        }
        scoped_pause();
        return this->m_internal->block_prefix_reduce(
            vt, op, size, block_size, exclusive, reverse, in, out);
    }

    /// Compute a dot product of two equal-sized arrays
    void reduce_dot(VarType type, const void *ptr_1, const void *ptr_2,
                    uint32_t size, void *out) override {
        jitc_log(
            LogLevel::Warn,
            "RecordThreadState::reduce_dot(): unsupported function recording!");
        scoped_pause();
        return this->m_internal->reduce_dot(type, ptr_1, ptr_2, size, out);
    }

    /// Asynchronously update a single element in memory
    void poke(void *dst, const void *src, uint32_t size) override {
        jitc_log(LogLevel::Warn,
                 "RecordThreadState::poke(): unsupported function recording!");
        scoped_pause();
        return this->m_internal->poke(dst, src, size);
    }

    void aggregate(void *dst, AggregationEntry *agg, uint32_t size) override {
        if (!paused()) {
            try {
                record_aggregate(dst, agg, size);
            } catch (...) {
                record_exception();
            }
        }
        scoped_pause();
        this->m_internal->aggregate(dst, agg, size);
    }

    // Enqueue a function to be run on the host once backend computation is done
    void enqueue_host_func(void (*callback)(void *), void *payload) override {
        scoped_pause();
        return this->m_internal->enqueue_host_func(callback, payload);
    }

    /// LLVM: reduce a variable that was previously expanded due to
    /// dr.ReduceOp.Expand
    void reduce_expanded(VarType vt, ReduceOp reduce_op, void *data,
                         uint32_t exp, uint32_t size) override {
        if (!paused()) {
            try {
                record_reduce_expanded(vt, reduce_op, data, exp, size);
            } catch (...) {
                record_exception();
            }
        }
        scoped_pause();
        return this->m_internal->reduce_expanded(vt, reduce_op, data, exp,
                                                 size);
    }

    /**
     * This function is called every time a pointer is freed using \ref
     * jitc_free. It records the operation and removes the mapping from that
     * pointer to the recorded variable.
     * If the pointer is reused later by another call to \ref jitc_malloc, the
     * \ref RecordThreadState.add_variable function will create a new variable
     * and mapping from the pointer to it.
     */
    void notify_free(const void *ptr) override {
        if (has_variable(ptr)) {
            jitc_log(LogLevel::Debug, "record(): jitc_free(ptr=%p)", ptr);

            uint32_t start = this->m_recording.dependencies.size();
            add_in_param(ptr, VarType::Void, false);
            uint32_t end = this->m_recording.dependencies.size();

            Operation op;
            op.type             = OpType::Free;
            op.dependency_range = std::pair(start, end);

            this->ptr_to_slot.erase(ptr);
        }
    }

    ~RecordThreadState() {}

    /**
     * Adds an input of the recording.
     * This is adds the slot of that variable to the \ref Recording.inputs
     * vector.
     */
    void add_input(uint32_t input) {
        try {
            uint32_t input_index = this->m_recording.inputs.size();
            Variable *v          = jitc_var(input);
            RecordVariable rv;
            rv.state      = RecordVarState::Input;
            rv.init       = RecordVarInit::Input;
            rv.index      = input_index;
            rv.type       = (VarType) v->type;
            uint32_t slot = this->add_variable(v->data, rv);
            jitc_log(LogLevel::Info,
                     "record(): Adding variable %u <%p> input %u to slot s%u",
                     input, v->data, input_index, slot);
            this->m_recording.inputs.push_back(slot);
        } catch (...) {
            record_exception();
        }
    }
    /**
     * Adds an output to the recording.
     * The output can be seen as a final operation, which also has to infer the
     * size of it's input variables.
     * Therefore, we record the full \ref ParamInfo for each output variable.
     */
    void add_output(uint32_t output) {
        try {
            uint32_t output_index = this->m_recording.outputs.size();
            Variable *v           = jitc_var(output);
            uint32_t slot;
            if (!has_variable(v->data)) {
                slot = capture_variable(output);
            } else {
                slot = this->get_variable(v->data);
            }

            jitc_log(LogLevel::Trace,
                     "record(): Adding variable %u output %u to slot s%u",
                     output, output_index, slot);
            ParamInfo info;
            info.slot  = slot;
            info.vtype = (VarType) v->type;
            this->m_recording.outputs.push_back(info);
        } catch (...) {
            record_exception();
        }
    }

    bool pause() {
        bool tmp = m_paused;
        m_paused = true;
        return tmp;
    }
    bool resume() {
        bool tmp = m_paused;
        m_paused = false;
        return tmp;
    }

    /// A helper scope, pausing recording.
    struct pause_scope {
        RecordThreadState *rts;
        bool tmp;

        pause_scope(RecordThreadState *rts) : rts(rts), tmp(rts->pause()) {}
        ~pause_scope() { rts->m_paused = tmp; }
    };

    pause_scope scoped_pause() { return pause_scope(this); }

    /// Is recording paused or has an exception been thrown?
    /// Recording any operation should be gated by this function.
    inline bool paused() { return m_paused || m_exception; }

    /// Records an exception, thrown while recording an operation.
    /// This is necessary to gracefully fail finishing freezing the function.
    inline void record_exception() {
        if (!m_exception)
            m_exception = std::current_exception();
    }

    bool m_paused = false;

    std::exception_ptr m_exception = nullptr;

    ThreadState *m_internal;

    Recording m_recording;

private:
    // Mapping from data pointer of a variable to a index into the slot of the
    // recording.
    PtrToSlot ptr_to_slot;

    /**
     * Record the Expand operation, corresponding to the `jitc_var_expand` call,
     * with which the variable `index` has been expanded.
     *
     * Reductions in LLVM might be split into three operations.
     * First the variable is expanded by its size times the number of workers +
     * 1 Then the kernel writes into the expanded variable with some offset, and
     * finally the variable is reduced.
     * The expand operation allocates a new memory region and copies the old
     * content into it.
     * We catch this case if the input variable of a kernel has a reduce_op
     * associated with it.
     * The source variable is discovered by walking the operations to the last
     * memcpy and memset, which are then disabled by this function.
     */
    void record_expand(uint32_t index);

    /// Record a kernel launch
    void record_launch(Kernel kernel, KernelKey *key, XXH128_hash_t hash,
                       uint32_t size, std::vector<void *> *kernel_params,
                       const std::vector<uint32_t> *kernel_param_ids);
    void record_memset_async(void *ptr, uint32_t size, uint32_t isize,
                             const void *src);
    void record_compress(const uint8_t *in, uint32_t size, uint32_t *out);
    void record_mkperm(const uint32_t *values, uint32_t size,
                       uint32_t bucket_count, uint32_t *perm,
                       uint32_t *offsets);
    void record_block_reduce(VarType vt, ReduceOp op, uint32_t size,
                             uint32_t block_size, const void *in, void *out);
    void record_block_prefix_reduce(VarType vt, ReduceOp op, uint32_t size,
                                    uint32_t block_size, bool exclusive,
                                    bool reverse, const void *in, void *out);
    void record_aggregate(void *dst, AggregationEntry *agg, uint32_t size);
    void record_reduce_expanded(VarType vt, ReduceOp reduce_op, void *data,
                                uint32_t exp, uint32_t size);

    /**
     * This captures the offset buffer of a vcall in a kernel.
     * The offset buffer describes where in the data buffer of that vcall the
     * variables or pointers to variables, for that vcall are stored.
     * It should not change between invocations and we should therefore be able
     * to capture it and reuse it when replaying the kernel.
     */
    uint32_t capture_call_offset(const void *ptr, size_t dsize) {
        uint32_t size;
        size = dsize / type_size[(uint32_t) VarType::UInt64];

        AllocType atype = backend == JitBackend::CUDA ? AllocType::Device
                                                      : AllocType::HostAsync;
        uint64_t *data  = (uint64_t *) jitc_malloc(atype, dsize);
        jitc_memcpy(backend, data, ptr, dsize);

        uint32_t data_var =
            jitc_var_mem_map(backend, VarType::UInt64, data, size, true);

        RecordVariable rv;
        rv.ptr   = ptr;
        rv.state = RecordVarState::Captured;
        rv.init  = RecordVarInit::Captured;
        rv.index = data_var;

        uint32_t slot;
        auto it = this->ptr_to_slot.find(ptr);
        if (it == this->ptr_to_slot.end()) {
            slot = this->m_recording.record_variables.size();

            this->m_recording.record_variables.push_back(rv);

            this->ptr_to_slot.insert({ ptr, slot });
        } else {
            slot                = it.value();
            RecordVariable &old = this->m_recording.record_variables[slot];
            if (old.init != RecordVarInit::None)
                jitc_fail("record(): Tried to overwrite a initialized variable "
                          "with an offset buffer!");

            this->m_recording.record_variables[slot] = rv;
        }

        return slot;
    }

    /**
     * This function tries to capture a variable that is not known to the
     * recording \c ThreadState.
     * This is unsupported for now and raises an exception.
     */
    uint32_t capture_variable(uint32_t index, const void */*ptr*/ = nullptr,
                              bool /*remember*/ = true, bool test_scope = true,
                              bool /*overwrite*/ = false) {

        scoped_pause();
        Variable *v = jitc_var(index);
        if (v->scope < this->m_internal->scope && test_scope) {
            jitc_raise(
                "record(): Variable r%u[%u] -> %p, label=%s, was created "
                "before recording was started, but it was "
                "not speciefied as an input variable! This can happen if a "
                "input type is not fully traversavle, for example when not "
                "specifying a member in DRJIT_STRUCT, but using it in the "
                "frozen function.",
                index, v->size, v->data, jitc_var_label(index));
        }

        jitc_raise("record(): Variable r%u[%u] -> %p, label=%s, data=%s, of "
                   "size > 1 was created while recording.",
                   index, v->size, v->data, jitc_var_label(index),
                   jitc_var_str(index));

        return 0;
    }

    /**
     * Add information about a variable, deduplicating it and returning the slot
     * in the `variables` field of the recording.
     * Information is combined when the variable has already been added.
     * This is used by the input variables of a kernel.
     */
    uint32_t add_variable(const void *ptr, RecordVariable rv) {

        rv.ptr  = ptr;
        auto it = this->ptr_to_slot.find(ptr);

        if (it == this->ptr_to_slot.end()) {
            uint32_t slot = this->m_recording.record_variables.size();

            this->m_recording.record_variables.push_back(rv);

            this->ptr_to_slot.insert({ ptr, slot });

            return slot;
        } else {
            uint32_t slot = it.value();

            this->m_recording.record_variables[slot] |= rv;

            return slot;
        }
    }

    /// Return the slot index given the data pointer of a variable.
    /// This fails if the variable has not been added.
    uint32_t get_variable(const void *ptr) {
        auto it = this->ptr_to_slot.find(ptr);

        if (it == this->ptr_to_slot.end())
            jitc_fail("Failed to find the slot corresponding to the variable "
                      "with data at %p",
                      ptr);

        return it.value();
    }

    /// Test if the ThreadState knows this \c ptr
    bool has_variable(const void *ptr) {
        auto it = this->ptr_to_slot.find(ptr);

        return it != this->ptr_to_slot.end();
    }

    /**
     * Adds a parameter access to the \ref dependencies vector.
     * This also modifies the state of the \ref RecordVariable that was
     * accessed.
     */
    void add_param(ParamInfo info) {
        RecordVariable &rv = this->m_recording.record_variables[info.slot];
        if (info.type == ParamType::Output) {

            jitc_log(LogLevel::Debug, " <- param s%u", info.slot);

            if (info.vtype != VarType::Void)
                rv.type = info.vtype;

            rv.state = RecordVarState::OpOutput;

        } else if (info.type == ParamType::Input) {

            jitc_log(LogLevel::Debug, " -> param s%u", info.slot);

            if (info.test_uninit && rv.state == RecordVarState::Uninit)
                jitc_raise("record(): Varaible at slot s%u was read from by "
                           "operation o%u, but has not yet been initialized! "
                           "This can happen if the variable was not part of "
                           "the input but is used by an recorded operation.",
                           info.slot,
                           (uint32_t) this->m_recording.operations.size());

            if (info.vtype == VarType::Void)
                info.vtype = rv.type;
        }

        this->m_recording.dependencies.push_back(info);
    }
    /// Helper function for recording input parameters given the slot.
    void add_in_param(uint32_t slot, VarType vtype = VarType::Void,
                      bool test_uninit = true) {
        ParamInfo info;
        info.type        = ParamType::Input;
        info.slot        = slot;
        info.test_uninit = test_uninit;
        info.vtype       = vtype;
        add_param(info);
    }
    /// Helper function recording input access given the pointer.
    void add_in_param(const void *ptr, VarType vtype = VarType::Void,
                      bool test_uninit = true) {
        uint32_t slot = this->get_variable(ptr);
        add_in_param(slot, vtype, test_uninit);
    }
    /// Helper function recording an output access, given the slot and \ref
    /// VarType
    void add_out_param(uint32_t slot, VarType vtype) {
        ParamInfo info;
        info.type  = ParamType::Output;
        info.slot  = slot;
        info.vtype = vtype;
        add_param(info);
    }
    /// Helper function recording an output access, given the pointer and \ref
    /// VarType
    void add_out_param(const void *ptr, VarType vtype) {
        RecordVariable rv;
        uint32_t slot = this->add_variable(ptr, rv);
        add_out_param(slot, vtype);
    }
    /// Helper function recording an output access, given the pointer and the
    /// uint32_t representation of a \ref VarType
    void add_out_param(uint32_t slot, uint32_t vtype) {
        add_out_param(slot, (VarType) vtype);
    }
};

void jitc_freeze_start(JitBackend backend, const uint32_t *inputs,
                       uint32_t n_inputs);

Recording *jitc_freeze_stop(JitBackend backend, const uint32_t *outputs,
                            uint32_t n_outputs);

void jitc_freeze_abort(JitBackend backend);

void jitc_freeze_destroy(Recording *recording);

bool jitc_freeze_pause(JitBackend backend);

bool jitc_freeze_resume(JitBackend backend);

void jitc_freeze_replay(Recording *recording, const uint32_t *inputs,
                        uint32_t *outputs);

int jitc_freeze_dry_run(Recording *recording, const uint32_t *inputs,
                        uint32_t *outputs);
