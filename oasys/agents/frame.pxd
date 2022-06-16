from oasys.structures.pomdp_structures import State

from oasys.structures.pomdp_structures cimport State


cdef class Frame():
    cdef readonly int index
    cdef readonly int[:] possible_internal_states

    cdef int[:] get_all_available_actions(self)
    cdef int[:] get_available_actions(self, int internal_state)
    cdef int[:] get_available_actions_per_state(self, State state, int internal_state)
    cdef int[:] get_all_available_actions_comm(self)
    cdef int[:] get_available_actions_comm(self, int internal_state)
    cdef int[:] get_available_actions_comm_per_state(self, State state, int internal_state)