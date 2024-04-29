#include "drjit-core/jit.h"
#include "test.h"

TEST_BOTH(01_basic_replay) {
    Recording *recording;

    jit_log(LogLevel::Info, "Recording:");
    {
        UInt32 r(0, 1, 2, 3, 4, 5, 6, 7, 8, 9);
        UInt32 ref(1, 2, 3, 4, 5, 6, 7, 8, 9, 10);

        uint32_t inputs[] = {
            r.index()
        };

        jit_record_start(Backend, inputs, 1);

        UInt32 result = r + 1;
        result.eval();

        uint32_t outputs[] = {
            result.index()
        };

        recording = jit_record_stop(Backend, outputs, 1);

        jit_log(LogLevel::Info, "result: %s", jit_var_str(outputs[0]));
        jit_assert(jit_var_all(jit_var_eq(ref.index(), outputs[0])));
    }


    jit_log(LogLevel::Info, "Replay:");
    {
        UInt32 r(1, 2, 3, 4, 5, 6, 7, 8, 9, 10);
        UInt32 ref(2, 3, 4, 5, 6, 7, 8, 9, 10, 11);

        uint32_t inputs[] = {
            r.index()
        };
        uint32_t outputs[1];

        jit_record_replay(recording, inputs, outputs);

        jit_log(LogLevel::Info, "result: %s", jit_var_str(outputs[0]));
        jit_assert(jit_var_all(jit_var_eq(ref.index(), outputs[0])));
    }

    jit_record_destroy(recording);
}

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

        jit_record_start(Backend, inputs, 2);

        UInt32 o0 = i0 + i1;
        UInt32 o1 = i0 * i1;
        o0.schedule();
        o1.schedule();
        jit_eval();

        uint32_t outputs[] = {
            o0.index(),
            o1.index(),
        };

        recording = jit_record_stop(Backend, outputs, 2);

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

        jit_record_replay(recording, inputs, outputs);

        jit_log(LogLevel::Info, "o0: %s", jit_var_str(outputs[0]));
        jit_log(LogLevel::Info, "o1: %s", jit_var_str(outputs[1]));
        jit_assert(jit_var_all(jit_var_eq(r0.index(), outputs[0])));
        jit_assert(jit_var_all(jit_var_eq(r1.index(), outputs[1])));
    }

    jit_record_destroy(recording);
}
