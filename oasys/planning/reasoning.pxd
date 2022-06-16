from oasys.agents.agent import Agent
from oasys.structures.pomdp_structures import State, Action

from oasys.agents.agent cimport Agent
from oasys.structures.pomdp_structures cimport State, Action


cdef class ReasoningModel():
    cdef Agent agent

    cdef Action choose_action(self, State state, int internal_state)