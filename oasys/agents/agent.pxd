from oasys.agents.frame import Frame
from oasys.domains.domain import Domain
from oasys.planning.reasoning import ReasoningModel
from oasys.structures.pomdp_structures import State, Action, Observation, Configuration

from oasys.agents.frame cimport Frame
from oasys.domains.domain cimport Domain
from oasys.planning.reasoning cimport ReasoningModel
from oasys.structures.pomdp_structures cimport State, Action, Observation, Configuration


cdef class Agent():
    cdef readonly int agent_num
    cdef readonly Frame frame
    cdef readonly Domain domain
    cdef readonly ReasoningModel reasoning
    cdef dict settings

    cpdef Action choose_action(self)
    cpdef list calculate_q_values(self)
    cdef void receive_reward(self, double reward)
    cdef void make_observation(self, State next_state, int next_internal_state, Observation observation)
    cdef Action sample_action(self, State state, int internal_state)
    cdef Action sample_action_with_configuration(self, State state, int internal_state, Configuration configuration)
    cdef Agent replicate_as_model(self)
