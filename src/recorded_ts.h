#include "eval.h"
#include "internal.h"
#include "log.h"

enum class OpType{
    KernelLaunch,
};
struct Operation{
    OpType type;
    // Indices into the dependencies vector
    std::pair<uint32_t, uint32_t> dependency_range;
};

// HashMap used to deduplicate variables
using VariableCache = tsl::robin_map<uint32_t, uint32_t>;

struct RecordedThreadState: ThreadState{
    
    RecordedThreadState(ThreadState *internal){
        this->internal = internal;
    }

    Task *launch(Kernel kernel, uint32_t size, 
                 std::vector<ScheduledVariable> &scheduled) override{

        // Record the operation
        uint32_t start = this->dependencies.size();
        for ( ScheduledVariable &sv : scheduled ){
            this->dependencies.push_back(this->get_or_insert_variable(sv.index));
        }
        uint32_t end = this->dependencies.size();
        this->operations.push_back({
            .type = OpType::KernelLaunch,
            .dependency_range = std::pair(start,end),
        });

        return this->internal->launch(kernel, size, scheduled);
    }

    /// Fill a device memory region with constants of a given type
    void memset_async(void *ptr, uint32_t size, uint32_t isize,
                      const void *src) override{
        return this->internal->memset_async(ptr, size, isize, src);
    }

    /// Reduce the given array to a single value
    void reduce(VarType type, ReduceOp rtype, const void *ptr, uint32_t size,
                void *out) override{
        return this->internal->reduce(type, rtype, ptr, size, out);
    }

    /// 'All' reduction for boolean arrays
    bool all(uint8_t *values, uint32_t size) override{
        return this->internal->all(values, size);
    }

    /// 'Any' reduction for boolean arrays
    bool any(uint8_t *values, uint32_t size) override{
        return this->internal->any(values, size);
    }

    /// Exclusive prefix sum
    void prefix_sum(VarType vt, bool exclusive, const void *in, uint32_t size,
                    void *out) override{
        return this->internal->prefix_sum(vt, exclusive, in, size, out);
    }

    /// Mask compression
    uint32_t compress(const uint8_t *in, uint32_t size, uint32_t *out) override{
        return this->internal->compress(in, size, out);
    }

    /// Compute a permutation to reorder an integer array into discrete groups
    uint32_t mkperm(const uint32_t *values, uint32_t size,
                    uint32_t bucket_count, uint32_t *perm,
                    uint32_t *offsets) override{
        return this->internal->mkperm(values, size, bucket_count, perm, offsets);
    }

    /// Perform a synchronous copy operation
    void memcpy(void *dst, const void *src, size_t size) override{
        return this->internal->memcpy(dst, src, size);
    }

    /// Perform an assynchronous copy operation
    void memcpy_async(void *dst, const void *src, size_t size) override{
        return this->internal->memcpy_async(dst, src, size);
    }

    /// Sum over elements within blocks
    void block_sum(enum VarType type, const void *in, void *out, uint32_t size,
                   uint32_t block_size) override{
        return this->internal->block_sum(type, in, out, size, block_size);
    }

    /// Asynchronously update a single element in memory
    void poke(void *dst, const void *src, uint32_t size) override{
        return this->internal->poke(dst, src, size);
    }

    void aggregate(void *dst, AggregationEntry *agg, uint32_t size) override{
        return this->internal->aggregate(dst, agg, size);
    }

    // Enqueue a function to be run on the host once backend computation is done
    void enqueue_host_func(void (*callback)(void *), void *payload) override{
        return this->internal->enqueue_host_func(callback, payload);
    }

    /// LLVM: reduce a variable that was previously expanded due to dr.ReduceOp.Expand
    void reduce_expanded(VarType vt, ReduceOp op, void *data, uint32_t exp,
                         uint32_t size) override {
        return this->internal->reduce_expanded(vt, op, data, exp, size);
    }

private:
    // Insert the variable index, deduplicating it and returning a index into the
    // `variables` field.
    uint32_t get_or_insert_variable(uint32_t index) {
        auto it = this->var_cache.find(index);

        if (it == this->var_cache.end()) {
            uint32_t id = this->variables.size();
            this->var_cache.insert({index, id});

            // TODO: store some information in the variables field
            this->variables.push_back(0);
            return id;
        } else {
            return it.value();
        }
    }

    ~RecordedThreadState(){}

    std::vector<Operation> operations;

    // Mapping from variable index in State to variable in this struct.
    VariableCache var_cache;
    std::vector<uint32_t> variables;
    
    std::vector<uint32_t> dependencies;

    ThreadState *internal;
};
