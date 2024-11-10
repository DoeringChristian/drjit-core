#include "drjit-core/array.h"
#include "drjit-core/jit.h"
#include "test.h"
#include <functional>
#include <tuple>
#include <type_traits>
#include <utility>

template<typename T>
using has_index_t = decltype(std::declval<T>().index());

template <typename T, typename = void> struct traversable{
    static constexpr bool value = false;
    static void traverse(T &v, std::vector<uint32_t> &indices){}
};

template <typename T>
struct traversable<T, std::void_t<decltype(std::declval<T>().index())>>
    : std::true_type {
    static constexpr bool value = true;
    static void traverse(T &v, std::vector<uint32_t> &indices){
        jit_log(LogLevel::Debug, "traverse indices[%u] = r%u", indices.size(), v.index());
        indices.push_back(v.index());
    }
};

template <typename Tuple, std::size_t... I>
static void traverse_tuple(Tuple& t, std::vector<uint32_t>& indices, std::index_sequence<I...>) {
    // Expands in left-to-right order
    (traversable<std::decay_t<decltype(std::get<I>(t))>>::traverse(std::get<I>(t), indices), ...);
}

template <typename... Ts>
struct traversable<std::tuple<Ts...>>{
    static constexpr bool value = true;
    static void traverse(std::tuple<Ts...> &v, std::vector<uint32_t> &indices){
        traverse_tuple(v, indices, std::index_sequence_for<Ts...>{});
    }
};

template<typename ...Args>
static void traverse_arguments(std::vector<uint32_t> &indices, Args&&... args){
    (traversable<std::decay_t<Args>>::traverse(args, indices), ...);
}

template<typename T>
using has_steal_t = decltype(T::steal(0));

template <typename T, typename = void> struct constructable{
    static constexpr bool value = false;
    static T construct(std::vector<uint32_t> &indices, uint32_t &counter) {
        static_assert(sizeof(T) == 0, "Could not construct type!");
    }
};

/// Construct any variable that has the \c borrow function
/// It is necessary to use \c borrow instead of \c steal, since we would split
/// ownership between outputs if they appeared twice.
template <typename T>
struct constructable<T, std::void_t<decltype(T::borrow(0))>> : std::true_type {
    static constexpr bool value = true;
    static T construct(std::vector<uint32_t> &indices, uint32_t &counter) {
        jit_log(LogLevel::Debug, "construct indices[%u] = r%u", counter, indices[counter]);
        return T::borrow(indices[counter++]);
    }
};

template<typename... Ts>
struct constructable<std::tuple<Ts...>>{
    static constexpr bool value = true;
    static std::tuple<Ts...> construct(std::vector<uint32_t> &indices,
                                       uint32_t &counter) {
        // NOTE: initializer list to guarantee order of construct evaluation
        return std::tuple{ constructable<Ts>::construct(indices, counter)... };
    }
};

static void log_vec(std::vector<uint32_t> &vec, const char *name){
    jit_log(LogLevel::Debug, "%s[%u]:", name, vec.size());
    uint32_t n = vec.size();
    for (uint32_t i = 0; i < n; i++) {
        jit_log(LogLevel::Debug, "    %s[%u]=r%u, (i=%u) < (n = %u) = %u", name,
                i, vec[i], i, n, (i < n));
        if(i > 10)
            abort();
    }
}


template <typename Func>
struct FrozenFunction {
    
    JitBackend m_backend;
    Func m_func;
    
    uint32_t m_outputs = 0;
    Recording *m_recording = nullptr;

    FrozenFunction(JitBackend backend, Func func)
        : m_backend(backend), m_func(func), m_outputs(0) {
        jit_log(LogLevel::Debug, "FrozenFunction()");
    }
    ~FrozenFunction(){
        jit_freeze_destroy(m_recording);
        m_recording = nullptr;
    }
    
    template<typename... Args>
    auto operator()(Args&&... args) {
        using Output = typename std::invoke_result<Func, Args...>::type;
        
        std::vector<uint32_t> input_vector;
        traverse_arguments(input_vector, args...);
        
        for (uint32_t i = 0; i < input_vector.size(); i++) {
            int rv;
            input_vector[i] = jit_var_schedule_force(input_vector[i], &rv);
        }
        jit_eval();

        Output output;
        if (!m_recording) {
            jit_log(LogLevel::Debug, "Record:");
            
            jit_freeze_start(m_backend, input_vector.data(),
                             input_vector.size());

            output = m_func(std::forward<Args>(args)...);

            std::vector<uint32_t> output_vector;
            traversable<Output>::traverse(output, output_vector);

            for (uint32_t i = 0; i < output_vector.size(); i++) {
                int rv;
                output_vector[i] = jit_var_schedule_force(output_vector[i], &rv);
            }
            jit_eval();

            m_recording = jit_freeze_stop(m_backend, output_vector.data(),
                                          output_vector.size());
            m_outputs = (uint32_t)output_vector.size();

            uint32_t counter = 0;
            output = constructable<Output>::construct(output_vector, counter);

            // Free the borrowed indices
            for(uint32_t index : output_vector)
                jit_var_dec_ref(index);
            
        } else {
            jit_log(LogLevel::Debug, "Replay:");
            
            std::vector<uint32_t> output_vector(1, 0);
            
            jit_freeze_replay(m_recording, input_vector.data(),
                              output_vector.data());
            

            uint32_t counter = 0;
            output = constructable<Output>::construct(output_vector, counter);
            
            // Free the borrowed indices
            for(uint32_t index : output_vector)
                jit_var_dec_ref(index);
        }

        return output;
    }
};

/**
 * Basic addition test.
 * Supplying a different input should replay the operation, with this input.
 * In this case, the input at replay is incremented and should result in an
 * incremented output.
 */
TEST_BOTH(01_basic_replay) {

    auto func = [](UInt32 input){
        return input + 1;
    };

    FrozenFunction frozen(Backend, func);

    for (uint32_t i = 0; i < 3; i++){
        auto input = arange<UInt32>(10+ i);

        auto result = frozen(input);

        auto reference = func(input);
        
        jit_assert(jit_var_all(jit_var_eq(result.index(), reference.index())));
    }
}

/**
 * This tests a single kernel with multiple unique inputs and outputs.
 */
TEST_BOTH(02_MIMO) {

    auto func = [](UInt32 x, UInt32 y) {
        return std::make_tuple(x + y, x * y);
    };
    
    FrozenFunction frozen(Backend, func);

    for (uint32_t i = 0; i < 3; i++){
        auto x = arange<UInt32>(10 + i);
        auto y = arange<UInt32>(10 + i) + 1;

        auto result = frozen(x, y);

        auto reference = func(x, y);

        jit_assert(jit_var_all(jit_var_eq(std::get<0>(result).index(), std::get<0>(reference).index())));
        jit_assert(jit_var_all(jit_var_eq(std::get<1>(result).index(), std::get<1>(reference).index())));
    }
}

/**
 * This tests if the recording feature works, when supplying the same variable
 * twice in the input. In the final implementation this test-case should never
 * occur, as variables would be deduplicated in beforehand.
 */
TEST_BOTH(03_deduplicating_input) {

    auto func = [](UInt32 x, UInt32 y){
        return std::tuple(x+y, x*y);
    };

    FrozenFunction frozen(Backend, func);

    for (uint32_t i = 0; i < 3; i++){
        auto x = arange<UInt32>(10 + i);

        auto result = frozen(x, x);

        auto reference = func(x, x);

        jit_assert(jit_var_all(jit_var_eq(std::get<0>(result).index(), std::get<0>(reference).index())));
        jit_assert(jit_var_all(jit_var_eq(std::get<1>(result).index(), std::get<1>(reference).index())));
    }
}

/**
 * This tests if the recording feature works, when supplying the same variable
 * twice in the output. In the final implementation this test-case should never
 * occur, as variables would be deduplicated in beforehand.
 */
TEST_BOTH(04_deduplicating_output) {
    // NOTE: removed this test, as this will never occur in the Python
    // implementation, but would require significant effort implementing the
    // FrozenFunction for the tests.
}

/**
 * This tests, Whether it is possible to record multiple kernels in sequence.
 * The input of the second kernel relies on the execution of the first.
 * On LLVM, the correctness of barrier operations is therefore tested.
 */
TEST_BOTH(05_sequential_kernels) {

    auto func = [](UInt32 x){
        auto y = x + 1;
        y.eval();
        return y + 1;
    };
    
    FrozenFunction frozen(Backend, func);

    for (uint32_t i = 0; i < 3; i++){
        auto x = arange<UInt32>(10 + i);

        auto result = frozen(x);

        auto reference = func(x);

        jit_assert(jit_var_all(jit_var_eq(result.index(), reference.index())));
    }
}

/**
 * This tests, Whether it is possible to record multiple independent kernels in
 * the same recording.
 * The variables of the kernels are of different size, therefore two kernels are
 * generated. At replay these can be executed in parallel (LLVM) or sequence
 * (CUDA).
 */
TEST_BOTH(06_parallel_kernels) {

    auto func = [](UInt32 x, UInt32 y){
        return std::tuple(x + 1, y + 1);
    };
    
    FrozenFunction frozen(Backend, func);

    for (uint32_t i = 0; i < 3; i++){
        auto x = arange<UInt32>(10 + i);
        auto y = arange<UInt32>(11 + i);

        auto result = frozen(x, y);

        auto reference = func(x, y);

        jit_assert(jit_var_all(jit_var_eq(std::get<0>(result).index(), std::get<0>(reference).index())));
        jit_assert(jit_var_all(jit_var_eq(std::get<1>(result).index(), std::get<1>(reference).index())));
    }
}

/**
 * This tests the recording and replay of a horizontal reduction operation
 * (hsum).
 */
TEST_BOTH(07_reduce_hsum) {

    auto func = [](UInt32 x){
        return hsum(x + 1);
    };
    
    FrozenFunction frozen(Backend, func);

    for (uint32_t i = 0; i < 3; i++){
        auto x = arange<UInt32>(10 + i);

        auto result = frozen(x);

        auto reference = func(x);

        jit_assert(jit_var_all(jit_var_eq(result.index(), reference.index())));
    }
}

/**
 * Tests recording of a prefix sum operation with different inputs at replay.
 */
TEST_BOTH(08_prefix_sum) {

    auto func = [](UInt32 x){
        return block_prefix_sum(x, x.size());
    };
    
    FrozenFunction frozen(Backend, func);

    for (uint32_t i = 0; i < 3; i++){
        auto x = arange<UInt32>(10 + i);
        x.make_opaque();

        auto result = frozen(x);

        auto reference = func(x);

        jit_assert(jit_var_all(jit_var_eq(result.index(), reference.index())));
    }
}

/**
 * Tests that it is possible to pass a single input to multiple outputs in a
 * frozen thread state without any use after free conditions.
 */
TEST_BOTH(10_input_passthrough) {
    
    auto func = [](UInt32 x){
        auto y = x + 1;
        return std::tuple(y, y);
    };
    
    FrozenFunction frozen(Backend, func);

    for (uint32_t i = 0; i < 3; i++){
        auto x = arange<UInt32>(10 + i);
        x.make_opaque();

        auto result = frozen(x);

        auto reference = func(x);

        jit_assert(jit_var_all(jit_var_eq(std::get<0>(result).index(), std::get<0>(reference).index())));
        jit_assert(jit_var_all(jit_var_eq(std::get<1>(result).index(), std::get<1>(reference).index())));
    }
}
