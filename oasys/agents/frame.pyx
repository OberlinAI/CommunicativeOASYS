cdef class Frame():
    """Representation of an agent frame.

    Attributes:

    index -- the unique index of the frame
    possible_internal_states -- the possible internal states of agents with this frame
    """

    cdef int[:] get_all_available_actions(self):
        pass


    cdef int[:] get_available_actions(self, int internal_state):
        pass


    cdef int[:] get_available_actions_per_state(self, State state, int internal_state):
        pass


    cdef int[:] get_all_available_actions_comm(self):
        pass


    cdef int[:] get_available_actions_comm(self, int internal_state):
        pass


    cdef int[:] get_available_actions_comm_per_state(self, State state, int internal_state):
        pass