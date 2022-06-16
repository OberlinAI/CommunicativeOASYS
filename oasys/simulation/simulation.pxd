from oasys.domains.domain import Domain
from oasys.structures.pomdp_structures import State, InternalStates, JointAction, JointActionComm

from oasys.domains.domain cimport Domain
from oasys.structures.pomdp_structures cimport State, InternalStates, JointAction, JointActionComm

cdef class Simulation():
    cdef readonly Domain domain
    cdef list agents
    cdef str state_log_filename
    cdef str actions_observations_rewards_log_filename
    cdef str actions_messages_observations_rewards_log_filename
    cdef str neighborhoods_log_filename

    cdef str state_log_header
    cdef str actions_observations_rewards_log_header
    cdef str actions_messages_observations_rewards_log_header
    cdef str neighborhoods_log_header

    cpdef void run(self, int num_steps)

    cdef void log_state(self, int step, State state, InternalStates internal_states)
    cdef void log_actions_observations_rewards(self, int step, JointAction joint_action, list observations,
                                               double[:] rewards)
    cdef void log_actions_messages_observations_rewards(self, int step, JointActionComm joint_action, list observations,
                                                        double[:] rewards)
    cdef void log_neighborhoods(self)
