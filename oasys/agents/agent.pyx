from oasys.structures.pomdp_structures import State, Observation

from oasys.structures.pomdp_structures cimport Observation


cdef class Agent():
    def __cinit__(self, *argv):
        pass


    cpdef Action choose_action(self):
        pass


    cpdef list calculate_q_values(self):
        pass


    cdef void receive_reward(self, double reward):
        pass


    cdef void make_observation(self, State next_state, int next_internal_state, Observation observation):
        pass


    cdef Action sample_action(self, State state, int internal_state):
        pass


    cdef Action sample_action_with_configuration(self, State state, int internal_state, Configuration configuration):
        pass


    cdef Agent replicate_as_model(self):
        pass


    def __lt__(self, other):
        # for sorting neighborhoods in IPOMCP
        return True
