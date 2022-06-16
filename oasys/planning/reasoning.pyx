cdef class ReasoningModel():
    cdef Action choose_action(self, State state, int internal_state):
        pass