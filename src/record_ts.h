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

void jitc_freeze_start(JitBackend backend, const uint32_t *inputs,
                       uint32_t n_inputs);

Recording *jitc_freeze_stop(JitBackend backend, const uint32_t *outputs,
                            uint32_t n_outputs);

void jitc_freeze_abort(JitBackend backend);

void jitc_freeze_destroy(Recording *recording);

int jitc_freeze_pause(JitBackend backend);

int jitc_freeze_resume(JitBackend backend);

void jitc_freeze_replay(Recording *recording, const uint32_t *inputs,
                        uint32_t *outputs);

int jitc_freeze_dry_run(Recording *recording, const uint32_t *inputs,
                        uint32_t *outputs);

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
            /// Whether a prefix sum operation is exclusive
            bool exclusive;
            bool reverse;
        } prefix_reduce;
        /// Bucket count for the mkperm operation.
        /// The function has to be re-recorded when the bucket count changes.
        /// Therefore this should not depend on the width of any variable.
        uint32_t bucket_count;
        /// Additional data such as the source of memset
        uint64_t data;
    };
    /// Records the size of the operation.
    size_t size;
    /// Records the size of the largest input variable (directly accessed or
    /// through a pointer if the kernel has no direct inputs).
    size_t input_size = 0;
    /// Whether this operation is enabled. We might have to disable some
    /// operations after the fact, and removing them from the Recording would be
    /// more complicated.
    bool enabled = true;
    /// Does this operation use optix?
    bool uses_optix = false;
    /// A copy of the shader binding table including a deepcopy of its hit- and
    /// miss- groups, used by the kernel. The callables are filled in by the \c
    /// CUDAThreadState::launch function.
    OptixShaderBindingTable *sbt;
};

/// Denotes the type of variable.
///
/// Output variables are only tracked through the outputs array, as this
/// information is only needed when constructing the output variables.
enum class RecordedVarState {
    /// This variable was not initialized
    Uninitialized,
    /// This variable has been created by an operation
    OpOutput,
    /// This variable is part of the function input
    Input,
    /// This variable has been captured i.e. it is copied and part of the
    /// recording.
    /// For example, the offset buffer of a vcall should not change between
    /// recording and replay and can be copied.
    /// Captured variables are immutable and copied when replaying, so that they
    /// are not changed by the replaying kernels.
    Captured,
};

/// Records how this variable was initialized
enum class RecordedVarInit {
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
 * The memory must have been allocated using `jit_malloc`, otherwise it cannot
 * be tracked.
 */
struct RecordedVariable {
    /// Stores index into input array if variable is input or index of captured
    /// variable
    uint32_t index = 0;
    /// Records how this variable has been initialized
    RecordedVarInit init = RecordedVarInit::None;

    /// Tracks the last memset and memcpy operations by indexing into the
    /// operations list, necessary for recording the expand operation.
    uint32_t last_memset = 0;
    uint32_t last_memcpy = 0;

    /// Tracks the current state of a variable
    RecordedVarState state = RecordedVarState::Uninitialized;
    /// Tracks the current type of the variable
    VarType type = VarType::Void;
#ifdef NDEBUG
    /// Tracks the pointer of the variable for debug purposes
    const void *ptr;
#endif

    RecordedVariable() {}

    /**
     * Not all information about variables might be known right away (see
     * memcpy). When new information about the variable is available, we can add
     * it to the already saved RecordVariable.
     */
    RecordedVariable &operator|=(const RecordedVariable &rhs) {
        if (this->state == RecordedVarState::Uninitialized) {
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
    /// Whether this recording requires a dryrun, to discover the size of
    /// certain operations.
    bool requires_dry_run = false;

    /// The variables used in this recording.
    /// Each variable refers to an allocation.
    /// If an allocation reuses a memory region, it is referred to by a separate
    /// variable.
    std::vector<RecordedVariable> recorded_variables;

    /// This vector maps the flat and deduplicated inputs to the frozen function
    /// to their variables in the \ref record_variables array.
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

#ifdef NDEBUG
    /// Counter, counting the number of kernels for debugging.
    uint32_t n_kernels = 0;
#endif

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
    
    /// The last exception thrown while recording.
    /// This is required to re-throw the exception at the end.
    std::exception_ptr m_exception = nullptr;

    /// The internal ThreadState, that is wrapped by this RecordThreadState.
    ThreadState *m_internal;

    /// The recording, produced when recording this ThreadState.
    Recording m_recording;

protected:
    /// Indicates that recording has been paused.
    bool m_paused = false;
    
    /// Mapping from data pointer of a variable to a index into the slot of the
    /// recording.
    PtrToSlot ptr_to_slot;

public:
    /**
     * Constructs a new RecordThreadState, wrapping an internal ThreadState.
     * This does not take ownership of the internal ThreadState
     * and it has to be kept alive until the RecordThreadState is destroyed.
     */
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

    void barrier() override;

    Task *launch(Kernel kernel, KernelKey *key, XXH128_hash_t hash,
                 uint32_t size, std::vector<void *> *kernel_params,
                 const std::vector<uint32_t> *kernel_param_ids) override;

    /// Fill a device memory region with constants of a given type
    void memset_async(void *ptr, uint32_t size, uint32_t isize,
                      const void *src) override;

    /// Mask compression
    uint32_t compress(const uint8_t *in, uint32_t size,
                      uint32_t *out) override;

    /// Compute a permutation to reorder an integer array into discrete groups
    uint32_t mkperm(const uint32_t *values, uint32_t size,
                    uint32_t bucket_count, uint32_t *perm,
                    uint32_t *offsets) override;

    /// Perform a synchronous copy operation
    void memcpy(void *dst, const void *src, size_t size) override;

    /// Perform an asynchronous copy operation
    void memcpy_async(void *dst, const void *src, size_t size) override;

    /// Sum over elements within blocks
    void block_reduce(VarType vt, ReduceOp op, uint32_t size,
                      uint32_t block_size, const void *in, void *out) override;

    void block_prefix_reduce(VarType vt, ReduceOp op, uint32_t size,
                             uint32_t block_size, bool exclusive, bool reverse,
                             const void *in, void *out) override;

    /// Compute a dot product of two equal-sized arrays
    void reduce_dot(VarType type, const void *ptr_1, const void *ptr_2,
                    uint32_t size, void *out) override;

    /// Asynchronously update a single element in memory
    void poke(void *dst, const void *src, uint32_t size) override;

    void aggregate(void *dst, AggregationEntry *agg, uint32_t size) override;

    // Enqueue a function to be run on the host once backend computation is done
    void enqueue_host_func(void (*callback)(void *), void *payload) override;

    /// LLVM: reduce a variable that was previously expanded due to
    /// dr.ReduceOp.Expand
    void reduce_expanded(VarType vt, ReduceOp reduce_op, void *data,
                         uint32_t exp, uint32_t size) override;

    /**
     * This function is called every time a pointer is freed using \ref
     * jitc_free. It records the operation and removes the mapping from that
     * pointer to the recorded variable.
     * If the pointer is reused later by another call to \ref jitc_malloc, the
     * \ref RecordThreadState.add_variable function will create a new variable
     * and mapping from the pointer to it.
     */
    void notify_free(const void *ptr) override;

    ~RecordThreadState() {}

    /**
     * Adds an input of the recording.
     * This is adds the slot of that variable to the \ref Recording.inputs
     * vector.
     */
    void add_input(uint32_t input);
    /**
     * Adds an output to the recording.
     * The output can be seen as a final operation, which also has to infer the
     * size of its input variables.
     * Therefore, we record the full \ref ParamInfo for each output variable.
     */
    void add_output(uint32_t output);

    bool pause();
    bool resume();

    /// A helper scope, pausing recording.
    struct pause_scope {
        RecordThreadState *rts;
        bool tmp;

        pause_scope(RecordThreadState *rts) : rts(rts), tmp(rts->pause()) {}
        ~pause_scope() { rts->m_paused = tmp; }
    };

    /// Is recording paused or has an exception been thrown?
    /// Recording any operation should be gated by this function.
    inline bool paused() { return m_paused || m_exception; }

    /// Records an exception, thrown while recording an operation.
    /// This is necessary to gracefully fail finishing freezing the function.
    inline void record_exception() {
        if (!m_exception)
            m_exception = std::current_exception();
    }

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
     *
     * \param ptr
     *      the pointer to the offset buffer
     *
     * \param dsize
     *      the size in bytes of the offset buffer
     *
     */
    uint32_t capture_call_offset(const void *ptr, size_t dsize);

    /**
     * This function tries to capture a variable that is not known to the
     * recording \c ThreadState.
     * This is unsupported for now and raises an exception.
     */
    uint32_t capture_variable(uint32_t index, const void * /*ptr*/ = nullptr,
                              bool /*remember*/ = true, bool test_scope = true,
                              bool /*overwrite*/ = false);

    /**
     * Add information about a variable, deduplicating it and returning the slot
     * in the `variables` field of the recording.
     * Information is combined when the variable has already been added.
     * This is used by the input variables of a kernel.
     */
    uint32_t add_variable(const void *ptr, RecordedVariable rv);

    /// Return the slot index given the data pointer of a variable.
    /// This fails if the variable has not been previously added.
    uint32_t get_variable(const void *ptr);

    /// Test if the ThreadState knows this \c ptr
    bool has_variable(const void *ptr);

    /**
     * Adds a parameter access to the \ref dependencies vector.
     * This also modifies the state of the \ref RecordVariable that was
     * accessed.
     */
    void add_param(ParamInfo info);
    /// Helper function for recording input parameters given the slot.
    void add_in_param(uint32_t slot, VarType vtype = VarType::Void,
                      bool test_uninit = true);
    /// Helper function recording input access given the pointer.
    void add_in_param(const void *ptr, VarType vtype = VarType::Void,
                      bool test_uninit = true);
    /// Helper function recording an output access, given the slot and \ref VarType
    void add_out_param(uint32_t slot, VarType vtype);
    /// Helper function recording an output access, given the pointer and \ref VarType
    void add_out_param(const void *ptr, VarType vtype);
    /// Helper function recording an output access, given the pointer and the
    /// uint32_t representation of a \ref VarType
    void add_out_param(uint32_t slot, uint32_t vtype);
};
