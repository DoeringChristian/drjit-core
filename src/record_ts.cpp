#include "record_ts.h"
#include "eval.h"


Task *RecordedThreadState::launch(Kernel kernel, ScheduledVariable schedule[],
                                  ScheduledGroup group){
    
    uint32_t start = this->n_variables;
    for(uint32_t group_index = group.start; group_index < group.end; ++group_index){
        ScheduledVariable sv = schedule[group_index];
        this->n_variables++;
    }
    uint32_t end = this->n_variables;
    
    ScheduledGroup _group(group.size, start, end);

    Op op{
        OpType::Launch,
        ScheduledGroup(group.size, start, end),
        kernel,
    };
    this->operations.push_back(op);

    return this->internal->launch(kernel, schedule, group);
}

void RecordedThreadState::replay(ScheduledVariable vars[]){
    for(uint32_t i = 0; i < this->operations.size(); ++i){
        Op &operation = this->operations[i];
        switch (operation.type){
            case OpType::Launch:{
                this->internal->launch(operation.kernel, vars, operation.group);
            }
            break;
            default:{
                jitc_fail("An operation, recorded could not be implemented "
                          "because it was not implemented");
            }
            break;
        }
    }
}

