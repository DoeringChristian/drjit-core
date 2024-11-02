#include "drjit-core/array.h"
#include "drjit-core/jit.h"
#include "test.h"

/**
 * Basic addition test.
 * Supplying a different input should replay the operation, with this input.
 * In this case, the input at replay is incremented and should result in an
 * incremented output.
 */
TEST_BOTH(01_basic_replay) {
    Recording *recording;

    jit_log(LogLevel::Info, "Recording:");
    {
        UInt32 i0(0, 1, 2, 3, 4, 5, 6, 7, 8, 9);
        UInt32 r0(1, 2, 3, 4, 5, 6, 7, 8, 9, 10);

        uint32_t inputs[] = {i0.index()};

        jit_freeze_start(Backend, inputs, 1);

        UInt32 o0 = i0 + 1;
        o0.eval();

        uint32_t outputs[] = {o0.index()};

        recording = jit_freeze_stop(Backend, outputs, 1);

        jit_log(LogLevel::Info, "o0: %s", jit_var_str(outputs[0]));
        jit_assert(jit_var_all(jit_var_eq(r0.index(), outputs[0])));
    }

    jit_log(LogLevel::Info, "Replay:");
    {
        UInt32 i0(1, 2, 3, 4, 5, 6, 7, 8, 9, 10);
        UInt32 r0(2, 3, 4, 5, 6, 7, 8, 9, 10, 11);

        uint32_t inputs[] = {i0.index()};
        uint32_t outputs[1];

        jit_freeze_replay(recording, inputs, outputs);

        jit_log(LogLevel::Info, "o0: %s", jit_var_str(outputs[0]));
        jit_assert(jit_var_all(jit_var_eq(r0.index(), outputs[0])));
    }

    jit_freeze_destroy(recording);
}

/**
 * This tests a single kernel with multiple unique inputs and outputs.
 */
TEST_BOTH(02_MIMO) {
    Recording *recording;

    jit_log(LogLevel::Info, "Recording:");
    {
        UInt32 i0(0, 1, 2, 3, 4, 5, 6, 7, 8, 9);
        UInt32 i1(0, 1, 2, 3, 4, 5, 6, 7, 8, 9);
        UInt32 r0(0, 2, 4, 6, 8, 10, 12, 14, 16, 18);
        UInt32 r1(0, 1, 4, 9, 16, 25, 36, 49, 64, 81);

        uint32_t inputs[] = {
            i0.index(),
            i1.index(),
        };

        jit_freeze_start(Backend, inputs, 2);

        UInt32 o0 = i0 + i1;
        UInt32 o1 = i0 * i1;
        o0.schedule();
        o1.schedule();
        jit_eval();

        uint32_t outputs[] = {
            o0.index(),
            o1.index(),
        };

        recording = jit_freeze_stop(Backend, outputs, 2);

        jit_log(LogLevel::Info, "o0: %s", jit_var_str(outputs[0]));
        jit_log(LogLevel::Info, "o1: %s", jit_var_str(outputs[1]));
        jit_assert(jit_var_all(jit_var_eq(r0.index(), outputs[0])));
        jit_assert(jit_var_all(jit_var_eq(r1.index(), outputs[1])));
    }

    jit_log(LogLevel::Info, "Replay:");
    {
        UInt32 i0(1, 2, 3, 4, 5, 6, 7, 8, 9, 10);
        UInt32 i1(1, 2, 3, 4, 5, 6, 7, 8, 9, 10);
        UInt32 r0(2, 4, 6, 8, 10, 12, 14, 16, 18, 20);
        UInt32 r1(1, 4, 9, 16, 25, 36, 49, 64, 81, 100);

        uint32_t inputs[] = {
            i0.index(),
            i1.index(),
        };
        uint32_t outputs[2];

        jit_freeze_replay(recording, inputs, outputs);

        jit_log(LogLevel::Info, "o0: %s", jit_var_str(outputs[0]));
        jit_log(LogLevel::Info, "o1: %s", jit_var_str(outputs[1]));
        jit_assert(jit_var_all(jit_var_eq(r0.index(), outputs[0])));
        jit_assert(jit_var_all(jit_var_eq(r1.index(), outputs[1])));
    }

    jit_freeze_destroy(recording);
}

/**
 * This tests if the recording feature works, when supplying the same variable
 * twice in the input. In the final implementation this test-case should never
 * occur, as variables would be deduplicated in beforehand.
 */
TEST_BOTH(03_deduplicating_input) {
    Recording *recording;

    jit_log(LogLevel::Info, "Recording:");
    {
        UInt32 i0(0, 1, 2, 3, 4, 5, 6, 7, 8, 9);
        UInt32 r0(0, 2, 4, 6, 8, 10, 12, 14, 16, 18);
        UInt32 r1(0, 1, 4, 9, 16, 25, 36, 49, 64, 81);

        uint32_t inputs[] = {
            i0.index(),
            i0.index(),
        };

        jit_freeze_start(Backend, inputs, 2);

        UInt32 o0 = i0 + i0;
        UInt32 o1 = i0 * i0;
        o0.schedule();
        o1.schedule();
        jit_eval();

        uint32_t outputs[] = {
            o0.index(),
            o1.index(),
        };

        recording = jit_freeze_stop(Backend, outputs, 2);

        jit_log(LogLevel::Info, "o0: %s", jit_var_str(outputs[0]));
        jit_log(LogLevel::Info, "o1: %s", jit_var_str(outputs[1]));
        jit_assert(jit_var_all(jit_var_eq(r0.index(), outputs[0])));
        jit_assert(jit_var_all(jit_var_eq(r1.index(), outputs[1])));
    }

    jit_log(LogLevel::Info, "Replay:");
    {
        UInt32 i0(1, 2, 3, 4, 5, 6, 7, 8, 9, 10);
        UInt32 r0(2, 4, 6, 8, 10, 12, 14, 16, 18, 20);
        UInt32 r1(1, 4, 9, 16, 25, 36, 49, 64, 81, 100);

        uint32_t inputs[] = {
            i0.index(),
            i0.index(),
        };
        uint32_t outputs[2];

        jit_freeze_replay(recording, inputs, outputs);

        jit_log(LogLevel::Info, "o0: %s", jit_var_str(outputs[0]));
        jit_log(LogLevel::Info, "o1: %s", jit_var_str(outputs[1]));
        jit_assert(jit_var_all(jit_var_eq(r0.index(), outputs[0])));
        jit_assert(jit_var_all(jit_var_eq(r1.index(), outputs[1])));
    }

    jit_freeze_destroy(recording);
}

/**
 * This tests if the recording feature works, when supplying the same variable
 * twice in the output. In the final implementation this test-case should never
 * occur, as variables would be deduplicated in beforehand.
 */
TEST_BOTH(04_deduplicating_output) {
    Recording *recording;
    jit_set_log_level_stderr(LogLevel::Debug);

    jit_log(LogLevel::Info, "Recording:");
    {
        UInt32 i0(0, 1, 2, 3, 4, 5, 6, 7, 8, 9);
        UInt32 i1(0, 1, 2, 3, 4, 5, 6, 7, 8, 9);
        UInt32 r0(0, 2, 4, 6, 8, 10, 12, 14, 16, 18);
        UInt32 r1(0, 2, 4, 6, 8, 10, 12, 14, 16, 18);

        uint32_t inputs[] = {
            i0.index(),
            i1.index(),
        };

        jit_freeze_start(Backend, inputs, 2);

        UInt32 o0 = i0 + i1;
        UInt32 o1 = i0 + i1;
        o0.schedule();
        o1.schedule();
        jit_eval();

        uint32_t outputs[] = {
            o0.index(),
            o1.index(),
        };

        recording = jit_freeze_stop(Backend, outputs, 2);

        jit_log(LogLevel::Info, "o0: %s", jit_var_str(outputs[0]));
        jit_log(LogLevel::Info, "o1: %s", jit_var_str(outputs[1]));
        jit_assert(jit_var_all(jit_var_eq(r0.index(), outputs[0])));
        jit_assert(jit_var_all(jit_var_eq(r1.index(), outputs[1])));
    }

    jit_log(LogLevel::Info, "Replay:");
    {
        UInt32 i0(1, 2, 3, 4, 5, 6, 7, 8, 9, 10);
        UInt32 i1(1, 2, 3, 4, 5, 6, 7, 8, 9, 10);
        UInt32 r0(2, 4, 6, 8, 10, 12, 14, 16, 18, 20);
        UInt32 r1(2, 4, 6, 8, 10, 12, 14, 16, 18, 20);

        uint32_t inputs[] = {
            i0.index(),
            i1.index(),
        };
        uint32_t outputs[2];

        jit_freeze_replay(recording, inputs, outputs);

        jit_log(LogLevel::Info, "o0: %s", jit_var_str(outputs[0]));
        jit_log(LogLevel::Info, "o1: %s", jit_var_str(outputs[1]));
        jit_assert(jit_var_all(jit_var_eq(r0.index(), outputs[0])));
        jit_assert(jit_var_all(jit_var_eq(r1.index(), outputs[1])));
    }

    jit_freeze_destroy(recording);
}

/**
 * This tests, weather it is possible to record multiple kernels in sequence.
 * The input of the second kernel relies on the execution of the first.
 * On LLVM, the correctness of barrier operations is therefore tested.
 */
TEST_BOTH(05_sequential_kernels) {
    Recording *recording;

    jit_log(LogLevel::Info, "Recording:");
    {
        UInt32 i0(0, 1, 2, 3, 4, 5, 6, 7, 8, 9);
        UInt32 r0(2, 3, 4, 5, 6, 7, 8, 9, 10, 11);

        uint32_t inputs[] = {
            i0.index(),
        };

        jit_freeze_start(Backend, inputs, 1);

        UInt32 tmp = i0 + 1;
        tmp.schedule();
        jit_eval();
        UInt32 o0 = tmp + 1;
        o0.schedule();
        jit_eval();

        uint32_t outputs[] = {
            o0.index(),
        };

        recording = jit_freeze_stop(Backend, outputs, 1);

        jit_log(LogLevel::Info, "o0: %s", jit_var_str(outputs[0]));
        jit_assert(jit_var_all(jit_var_eq(r0.index(), outputs[0])));
    }

    jit_log(LogLevel::Info, "Replay:");
    {
        UInt32 i0(0, 1, 2, 3, 4, 5, 6, 7, 8, 9);
        UInt32 r0(2, 3, 4, 5, 6, 7, 8, 9, 10, 11);

        uint32_t inputs[] = {
            i0.index(),
        };
        uint32_t outputs[1];

        jit_freeze_replay(recording, inputs, outputs);

        jit_log(LogLevel::Info, "o0: %s", jit_var_str(outputs[0]));
        jit_assert(jit_var_all(jit_var_eq(r0.index(), outputs[0])));
    }

    jit_freeze_destroy(recording);
}

/**
 * This tests, weather it is possible to record multiple kernels in parallel.
 * The variables of the kernels are of different size, therefore two kernels are
 * generated. At replay these can be executed in parallel (LLVM) or sequence
 * (CUDA).
 */
TEST_BOTH(06_parallel_kernels) {
    Recording *recording;

    jit_log(LogLevel::Info, "Recording:");
    {
        UInt32 i0(0, 1, 2, 3, 4, 5, 6, 7, 8, 9);
        UInt32 i1(0, 1, 2, 3, 4, 5);
        UInt32 r0(1, 2, 3, 4, 5, 6, 7, 8, 9, 10);
        UInt32 r1(1, 2, 3, 4, 5, 6);

        uint32_t inputs[] = {
            i0.index(),
            i1.index(),
        };

        jit_freeze_start(Backend, inputs, 2);

        UInt32 o0 = i0 + 1;
        UInt32 o1 = i1 + 1;
        o0.schedule();
        o1.schedule();
        jit_eval();

        uint32_t outputs[] = {
            o0.index(),
            o1.index(),
        };

        recording = jit_freeze_stop(Backend, outputs, 2);

        jit_log(LogLevel::Info, "o0: %s", jit_var_str(outputs[0]));
        jit_log(LogLevel::Info, "o1: %s", jit_var_str(outputs[1]));
        jit_assert(jit_var_all(jit_var_eq(r0.index(), outputs[0])));
        jit_assert(jit_var_all(jit_var_eq(r1.index(), outputs[1])));
    }

    jit_log(LogLevel::Info, "Replay:");
    {
        UInt32 i0(1, 2, 3, 4, 5, 6, 7, 8, 9, 10);
        UInt32 i1(1, 2, 3, 4, 5, 6);
        UInt32 r0(2, 3, 4, 5, 6, 7, 8, 9, 10, 11);
        UInt32 r1(2, 3, 4, 5, 6, 7);

        uint32_t inputs[] = {
            i0.index(),
            i1.index(),
        };
        uint32_t outputs[2];

        jit_freeze_replay(recording, inputs, outputs);

        jit_log(LogLevel::Info, "o0: %s", jit_var_str(outputs[0]));
        jit_log(LogLevel::Info, "o1: %s", jit_var_str(outputs[1]));
        jit_assert(jit_var_all(jit_var_eq(r0.index(), outputs[0])));
        jit_assert(jit_var_all(jit_var_eq(r1.index(), outputs[1])));
    }

    jit_freeze_destroy(recording);
}

/**
 * This tests recording and replay of a horizontal reduction operation (hsum).
 */
TEST_BOTH(07_reduce_hsum) {
    Recording *recording;

    jit_log(LogLevel::Info, "Recording:");
    {
        UInt32 i0(0, 1, 2, 3, 4, 5, 6, 7, 8, 9);
        UInt32 r0 = opaque<UInt32>(45, 1);

        uint32_t inputs[] = {
            i0.index(),
        };

        jit_freeze_start(Backend, inputs, 1);

        UInt32 o0 = hsum(i0);
        o0.schedule();
        jit_eval();

        uint32_t outputs[] = {
            o0.index(),
        };

        recording = jit_freeze_stop(Backend, outputs, 1);

        jit_log(LogLevel::Info, "o0: %s", jit_var_str(outputs[0]));
        jit_assert(jit_var_all(jit_var_eq(r0.index(), outputs[0])));
    }

    jit_log(LogLevel::Info, "Replay:");
    {
        UInt32 i0(1, 2, 3, 4, 5, 6, 7, 8, 9, 10);
        UInt32 r0 = opaque<UInt32>(55, 1);

        uint32_t inputs[] = {
            i0.index(),
        };
        uint32_t outputs[1];

        jit_freeze_replay(recording, inputs, outputs);

        jit_log(LogLevel::Info, "o0: %s", jit_var_str(outputs[0]));
        jit_assert(jit_var_all(jit_var_eq(r0.index(), outputs[0])));
    }

    jit_freeze_destroy(recording);
}

/**
 * Tests recording of a prefix sum operation with different inputs at replay.
 */
TEST_BOTH(08_prefix_sum) {
    Recording *recording;

    jit_log(LogLevel::Info, "Recording:");
    {
        UInt32 i0(0, 1, 2, 3, 4, 5, 6, 7, 8, 9);
        UInt32 r0(0, 1, 3, 6, 10, 15, 21, 28, 36, 45);

        uint32_t inputs[] = {
            i0.index(),
        };

        jit_freeze_start(Backend, inputs, 1);

        uint32_t o0 = jit_var_block_prefix_reduce(
            ReduceOp::Add, i0.index(), jit_var_size(i0.index()), 0, 0);
        jit_var_schedule(o0);
        jit_eval();

        uint32_t outputs[] = {
            o0,
        };

        recording = jit_freeze_stop(Backend, outputs, 1);

        jit_log(LogLevel::Info, "o0: %s", jit_var_str(outputs[0]));
        jit_assert(jit_var_all(jit_var_eq(r0.index(), outputs[0])));
    }

    jit_log(LogLevel::Info, "Replay:");
    {
        UInt32 i0(1, 2, 3, 4, 5, 6, 7, 8, 9, 10);
        UInt32 r0(1, 3, 6, 10, 15, 21, 28, 36, 45, 55);

        uint32_t inputs[] = {
            i0.index(),
        };
        uint32_t outputs[1];

        jit_freeze_replay(recording, inputs, outputs);

        jit_log(LogLevel::Info, "o0: %s", jit_var_str(outputs[0]));
        jit_assert(jit_var_all(jit_var_eq(r0.index(), outputs[0])));
    }

    jit_freeze_destroy(recording);
}

/**
 * Basic addition test.
 * Supplying a different input should replay the operation, with this input.
 * In this case, the input at replay is incremented and should result in an
 * incremented output.
 */
TEST_BOTH(9_resized_input) {
    Recording *recording;

    jit_log(LogLevel::Info, "Recording:");
    {
        UInt32 i0(0, 1, 2);
        UInt32 r0(1, 2, 3);

        uint32_t inputs[] = {i0.index()};

        jit_freeze_start(Backend, inputs, 1);

        UInt32 o0 = i0 + 1;
        o0.eval();

        uint32_t outputs[] = {o0.index()};

        recording = jit_freeze_stop(Backend, outputs, 1);

        jit_log(LogLevel::Info, "o0: %s", jit_var_str(outputs[0]));
        jit_assert(jit_var_all(jit_var_eq(r0.index(), outputs[0])));
    }

    jit_log(LogLevel::Info, "Replay:");
    {
        UInt32 i0(1, 2, 3, 4);
        UInt32 r0(2, 3, 4, 5);

        uint32_t inputs[] = {i0.index()};
        uint32_t outputs[1];

        jit_freeze_replay(recording, inputs, outputs);

        jit_log(LogLevel::Info, "o0: %s", jit_var_str(outputs[0]));
        jit_assert(jit_var_all(jit_var_eq(r0.index(), outputs[0])));
    }

    jit_freeze_destroy(recording);
}

TEST_BOTH(10_input_passthrough) {
    Recording *recording;

    jit_log(LogLevel::Info, "Recording:");
    {
        UInt32 i0(0, 1, 2, 3, 4, 5, 6, 7, 8, 9);
        UInt32 r0(1, 2, 3, 4, 5, 6, 7, 8, 9, 10);

        uint32_t inputs[] = {i0.index()};

        jit_freeze_start(Backend, inputs, 1);

        UInt32 o0 = i0 + 1;
        o0.eval();

        uint32_t outputs[] = {o0.index(), i0.index()};

        recording = jit_freeze_stop(Backend, outputs, 2);

        jit_log(LogLevel::Info, "o0: %s", jit_var_str(outputs[0]));
        jit_log(LogLevel::Info, "o1: %s", jit_var_str(outputs[1]));
        jit_assert(jit_var_all(jit_var_eq(r0.index(), outputs[0])));
        jit_assert(jit_var_all(jit_var_eq(i0.index(), outputs[1])));
    }

    jit_log(LogLevel::Info, "Replay:");
    {
        UInt32 i0(1, 2, 3, 4, 5, 6, 7, 8, 9, 10);
        UInt32 r0(2, 3, 4, 5, 6, 7, 8, 9, 10, 11);

        uint32_t inputs[] = {i0.index()};
        uint32_t outputs[2];

        jit_freeze_replay(recording, inputs, outputs);

        jit_log(LogLevel::Info, "o0: %s", jit_var_str(outputs[0]));
        jit_log(LogLevel::Info, "o1: %s", jit_var_str(outputs[1]));
        jit_assert(jit_var_all(jit_var_eq(r0.index(), outputs[0])));
        jit_assert(jit_var_all(jit_var_eq(i0.index(), outputs[1])));
    }

    jit_freeze_destroy(recording);
}
