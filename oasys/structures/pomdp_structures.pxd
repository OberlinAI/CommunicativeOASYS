cdef class State():
    cdef readonly int index


cdef class InternalStates():
    cdef public int[:] values


cdef class PartialInternalStates(InternalStates):
    cdef public int[:] agent_nums


cdef class NestedPartialInternalStates(PartialInternalStates):
    cdef readonly int level
    cdef public list nested_internal_states


cdef class FactoredState(State):
    cdef public int[:] values


cdef class Action():
    cdef readonly int index


cdef class ActionComm(Action):
    cdef readonly int message


cdef class JointAction():
    cdef public int[:] actions


cdef class JointActionComm(JointAction):
    cdef public int[:] messages


cdef class PartialJointAction(JointAction):
    cdef public int[:] agent_nums


cdef class PartialJointActionComm(PartialJointAction):
    cdef public int[:] messages


cdef class Configuration():
    cdef public int[:] actions


cdef class Observation():
    cdef readonly int index


cdef class ObservationComm(Observation):
    cdef readonly int[:] messages

    cdef long calculate_message_index(self, int[:] possible_messages)
