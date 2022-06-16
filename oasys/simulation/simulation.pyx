from oasys.domains.domain import Domain
from oasys.structures.pomdp_structures import State, FactoredState, InternalStates, JointAction, JointActionComm

from oasys.domains.domain cimport Domain
from oasys.structures.pomdp_structures cimport State, FactoredState, InternalStates, JointAction, JointActionComm


cdef class Simulation():
    def __cinit__(self):
        pass


    cpdef void run(self, int num_steps):
        pass


    cdef void log_state(self, int step, State state, InternalStates internal_states):
        # do we need to write the header?
        if step == 0:
            with open(self.state_log_filename, "w") as file:
                file.write(self.state_log_header + "\n")

        # append this step
        cdef str line
        with open(self.state_log_filename, "a") as file:
            line = str(step)
            if isinstance(state, FactoredState):
                line += "," + ",".join([str(state.values[i]) for i in range(len(state.values))])
            else:
                line += "," + str(state.index)
            line += "," + ",".join([str(internal_states.values[i]) for i in range(len(internal_states.values))])
            file.write(line + "\n")


    cdef void log_actions_observations_rewards(self, int step, JointAction joint_action, list observations,
                                          double[:] rewards):
        # do we need to write the header?
        if step == 1:
            with open(self.actions_observations_rewards_log_filename, "w") as file:
                file.write(self.actions_observations_rewards_log_header + "\n")

        # append this step
        cdef int i
        cdef str line
        with open(self.actions_observations_rewards_log_filename, "a") as file:

            for i in range(len(self.domain.agents)):
                #print(f' bad boi: observations {i} : { np.array(observations[i]) } \n' )
                line = str(step) + "," + str(i)
                line += "," + str(joint_action.actions[i])
                line += "," + str(observations[i])
                line += "," + str(rewards[i])
                file.write(line + "\n")


    cdef void log_actions_messages_observations_rewards(self, int step, JointActionComm joint_action, list observations,
                                                        double[:] rewards):
        # do we need to write the header?
        if step == 1:
            with open(self.actions_messages_observations_rewards_log_filename, "w") as file:
                file.write(self.actions_messages_observations_rewards_log_header + "\n")

        # append this step
        cdef int i
        cdef str line
        with open(self.actions_messages_observations_rewards_log_filename, "a") as file:

            for i in range(len(self.domain.agents)):
                line = str(step) + "," + str(i)
                line += "," + str(joint_action.actions[i])
                line += "," + str(joint_action.messages[i])
                line += "," + str(observations[i])
                line += "," + str(rewards[i])
                file.write(line + "\n")


    cdef void log_neighborhoods(self):
        # this is domain specific
        pass
