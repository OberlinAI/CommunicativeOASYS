from oasys.simulation.simulation import Simulation
import numpy as np

from .wildfire_structures import WildfireState, WildfireInternalStates, WildfireJointActionComm, WildfireJointAction, WildfireState, WildfireObservation

from oasys.simulation.simulation cimport Simulation
cimport numpy as np
from .wildfire_structures cimport WildfireState, WildfireInternalStates, WildfireJointActionComm, WildfireJointAction, WildfireState, WildfireObservation

cdef class WildfireSimulation(Simulation):
    pass

cdef class WildfireNetworkSimulation(Simulation):
    cdef int run_number
    cdef int setup
    cdef dict settings
    cdef int reasoning_type_value
    cpdef WildfireJointActionComm client_run(self, int client_num, WildfireState state, WildfireInternalStates internal_states, int total_clients)
    cpdef list server_run(self, WildfireState state, WildfireInternalStates internal_states, WildfireJointActionComm joint_action_comm)
    cpdef void update_reward_obs(self, np.ndarray[np.double_t, ndim=1]rewards, list observations, WildfireState next_state, WildfireInternalStates next_internal_states,  int client_num, int total_clients)
    cpdef void update_log_state(self,int step, WildfireState state, WildfireInternalStates internal_states)
    cpdef void update_log_actions_observations_rewards_messages(self, int step, WildfireJointActionComm joint_action_comm, list observations, np.ndarray[np.double_t, ndim=1] rewards)
