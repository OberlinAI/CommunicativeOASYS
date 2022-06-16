from oasys.agents.frame import Frame
from .wildfire import Wildfire

from oasys.agents.frame cimport Frame
from .wildfire cimport Wildfire


cdef class WildfireFrame(Frame):
    cdef Wildfire domain
    cdef readonly double fire_reduction
    cdef readonly int loc_num
    cdef readonly int[:] all_available_actions
    cdef readonly int[:] all_available_actions_comm
    cdef readonly list available_actions_per_state
    cdef readonly list available_actions_comm_per_state
    cdef readonly int[:] noop_actions_comm

    cdef int[:] find_possible_internal_states(self)
    cdef int[:] find_available_actions(self)
    cdef int[:] find_available_actions_comm(self)
    cdef list find_available_actions_per_state(self)
    cdef list find_available_actions_comm_per_state(self)
    cdef int[:] find_noop_actions_comm(self)
