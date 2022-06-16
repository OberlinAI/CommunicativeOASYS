from oasys.simulation.simulation import Simulation
from oasys.structures.pomdp_structures import Action, ActionComm
from .wildfire import Wildfire
from .wildfire_agent import WildfireAgent
from .wildfire_reasoning import WildfireReasoningType
from .wildfire_structures import WildfireState, WildfireInternalStates, WildfireAction, WildfireActionComm, \
                                 WildfireJointAction, WildfireJointActionComm, WildfireConfiguration, \
                                 WildfireObservation, WildfireObservationComm
import numpy as np
from multiprocessing import Pool

from oasys.simulation.simulation cimport Simulation
from oasys.structures.pomdp_structures cimport Action, ActionComm
from .wildfire cimport Wildfire
from .wildfire_agent cimport WildfireAgent
from .wildfire_reasoning cimport WildfireReasoningType
from .wildfire_structures cimport WildfireState, WildfireInternalStates, WildfireAction, WildfireActionComm, \
                                  WildfireJointAction, WildfireJointActionComm, WildfireConfiguration, \
                                  WildfireObservation, WildfireObservationComm
cimport numpy as np

from libc.stdlib cimport srand


cdef class WildfireSimulation(Simulation):
    def __cinit__(self, int run, setup=None, reasoning_type=WildfireReasoningType.NOOP, dict settings={}):
        # set both of the random seeds
        np.random.seed(run + 1000 * setup)
        srand(run + 1000 * setup)
        # create the domain
        self.domain = Wildfire(setup, reasoning_type, settings)

        # create the filenames
        r_type = str(reasoning_type).replace("WildfireReasoningType.","")
        if reasoning_type == WildfireReasoningType.POMCPPF\
                or reasoning_type == WildfireReasoningType.IPOMCPPF\
                or reasoning_type == WildfireReasoningType.IPOMCPPFComm:
            r_type += "-" + str(int(round(settings["epsilon_p"] * 100))) + "e"

        if reasoning_type == WildfireReasoningType.IPOMCPPF\
                or reasoning_type == WildfireReasoningType.IPOMCPPFComm:
            r_type += "-" + str(settings["level"]) + "l"

        self.state_log_filename = "Wildfire_States_" \
                                  + r_type \
                                  + "_Setup" + str(self.domain.settings.SETUP) \
                                  + "_Traj" + str(settings["trajectories"]) \
                                  + "_Run" + str(run) \
                                  + ".csv"
        self.actions_observations_rewards_log_filename = "Wildfire_AOR_" \
                                                         + r_type \
                                                         + "_Setup" + str(self.domain.settings.SETUP) \
                                                         + "_Traj" + str(settings["trajectories"]) \
                                                         + "_Run" + str(run) \
                                                         + ".csv"
        self.actions_messages_observations_rewards_log_filename = "Wildfire_AMOR_" \
                                                                  + r_type \
                                                                  + "_Setup" + str(self.domain.settings.SETUP) \
                                                                  + "_Traj" + str(settings["trajectories"]) \
                                                                  + "_Run" + str(run) \
                                                                  + ".csv"
        self.neighborhoods_log_filename = "Wildfire_Neighborhoods_" \
                                          + r_type \
                                          + "_Setup" + str(self.domain.settings.SETUP) \
                                          + "_Traj" + str(settings["trajectories"]) \
                                          + "_Run" + str(run) \
                                          + ".csv"

        # create the log headers
        self.state_log_header = "Step," + ",".join(["f" + str(i) for i in range(self.domain.settings.FIRES)]
                                          + ["s" + str(i) for i in range(self.domain.num_agents())])
        self.actions_observations_rewards_log_header = "Step,Agent,Action,Observation,Reward"
        self.actions_messages_observations_rewards_log_header = "Step,Agent,Action,Message,Observation,Reward"

        # log the agents' neighborhoods
        if reasoning_type == WildfireReasoningType.POMCPPF\
                or reasoning_type == WildfireReasoningType.IPOMCPPF\
                or reasoning_type == WildfireReasoningType.IPOMCPPFComm:
            self.log_neighborhoods()


    cpdef void run(self, int num_steps):
        # get and log the start state
        cdef WildfireState state = self.domain.generate_start_state()
        cdef WildfireInternalStates internal_states = self.domain.generate_start_internal_states()
        self.log_state(0, state, internal_states)

        # give the initial state information to the agents
        cdef int agents_n = self.domain.num_agents()
        cdef int agent_num
        cdef WildfireAgent agent, neighbor
        for agent_num in range(agents_n):
            agent = self.domain.agents[agent_num]

            agent.set_current_state(state)
            agent.set_current_suppressant(internal_states.values[agent_num])

        # run the simulation
        cdef int step, neighbor_i, neighbors_n
        cdef list observations
        cdef Action action
        cdef ActionComm action_comm
        cdef WildfireJointAction joint_action
        cdef WildfireJointActionComm joint_action_comm
        cdef WildfireConfiguration configuration
        cdef WildfireState next_state
        cdef WildfireInternalStates next_internal_states
        cdef WildfireObservation observation
        cdef WildfireObservationComm observation_comm
        cdef np.ndarray[np.double_t, ndim=1] rewards
        cdef list neighbors

        cdef bint using_comm = self.domain.settings.COMMUNICATION

        for step in range(1, num_steps + 1):
            print("Step: ", step, "Fires:", np.array(state.values), "Suppressants:", np.array(internal_states.values))
            # get actions from each agent
            if using_comm:
                joint_action_comm = WildfireJointActionComm(agents_n)
            else:
                joint_action = WildfireJointAction(agents_n)

            for agent_num in range(agents_n):
                agent = self.domain.agents[agent_num]
                action = agent.choose_action()

                if using_comm:
                    action_comm = <ActionComm> action

                    joint_action_comm.actions[agent_num] = action_comm.index
                    joint_action_comm.messages[agent_num] = action_comm.message
                else:
                    joint_action.actions[agent_num] = action.index

            # create the corresponding configuration
            if using_comm:
                configuration = self.domain.create_configuration(-1, joint_action_comm, False)
            else:
                configuration = self.domain.create_configuration(-1, joint_action, False)
            print("Configuration:", np.array(configuration.actions))

            # sample a next state
            # NOTE: in the Wildfire domain, transitions are agent independent
            next_state = self.domain.sample_next_state_configuration(-1, state, configuration, None)

            if using_comm:
                next_internal_states = self.domain.sample_next_internal_states(-1, internal_states, joint_action_comm)
            else:
                next_internal_states = self.domain.sample_next_internal_states(-1, internal_states, joint_action)

            # calculate rewards and observations for each agent
            rewards = np.ndarray((agents_n,), dtype=np.double)
            observations = []
            for agent_num in range(agents_n):
                agent = self.domain.agents[agent_num]

                if using_comm:
                    action = WildfireAction(joint_action_comm.actions[agent_num])
                    action_comm = WildfireActionComm(joint_action_comm.actions[agent_num],
                                                     joint_action_comm.messages[agent_num])
                else:
                    action = WildfireAction(joint_action.actions[agent_num])

                observation = self.domain.sample_observation_configuration(agent_num, state, configuration, action,
                                                                           next_state)

                if using_comm:
                    neighbors = <list> self.domain.modeling_neighborhoods_per_agent[agent_num]
                    neighbors_n = len(neighbors)

                    # create the collection of messages that agent_num would receive
                    observation_comm = WildfireObservationComm(observation.fire_change, neighbors_n)
                    for neighbor_i in range(neighbors_n):
                        neighbor = <WildfireAgent> neighbors[neighbor_i]
                        observation_comm.messages[neighbor_i] = joint_action_comm.messages[neighbor.agent_num]

                    observations.append(observation_comm)

                    rewards[agent_num] = self.domain.reward_configuration_with_comm(agent_num, state, internal_states,
                                                                                    configuration, action_comm,
                                                                                    next_state, next_internal_states)
                else:
                    observations.append(observation)

                    rewards[agent_num] = self.domain.reward_configuration(agent_num, state, internal_states,
                                                                          configuration, action, next_state,
                                                                          next_internal_states)

            # give this information to the agents
            for agent_num in range(agents_n):
                agent = self.domain.agents[agent_num]
                agent.receive_reward(rewards[agent_num])
                agent.make_observation(next_state, next_internal_states.values[agent_num], observations[agent_num])

            # update the state
            state = next_state
            internal_states = next_internal_states

            # log this information
            self.log_state(step, next_state, next_internal_states)
            if using_comm:
                self.log_actions_messages_observations_rewards(step, joint_action_comm, observations, rewards)
            else:
                self.log_actions_observations_rewards(step, joint_action, observations, rewards)


    cdef void log_neighborhoods(self):
        # find the largest neighborhood
        cdef int largest_neighborhood = 0
        cdef int agents_n = self.domain.num_agents()
        cdef int agent_i, neighborhood_n, neighbor_i
        cdef list agent_models
        cdef WildfireAgent agent, neighbor

        for agent_i in range(agents_n):
            agent = <WildfireAgent> self.domain.agents[agent_i]
            agent_models = agent.reasoning.planner.agent_models

            neighborhood_n = len(agent_models)
            if neighborhood_n > largest_neighborhood:
                largest_neighborhood = neighborhood_n

        # create the header
        self.neighborhoods_log_header = "Agent," + ",".join(["Neighbor" + str(i) for i in range(largest_neighborhood)])

        cdef str line
        with open(self.neighborhoods_log_filename, "w") as file:
            # write the header
            file.write(self.neighborhoods_log_header + "\n")

            # save each agent's modeled neighborhood
            for agent_i in range(agents_n):
                agent = <WildfireAgent> self.domain.agents[agent_i]
                agent_models = agent.reasoning.planner.agent_models
                neighborhood_n = len(agent_models)

                # start the line with this agent
                line = str(agent.agent_num)

                # add each neighbor
                for neighbor_i in range(neighborhood_n):
                    neighbor = <WildfireAgent> agent_models[neighbor_i]

                    line += "," + str(neighbor.agent_num)

                # save the line
                file.write(line + "\n")




def rebuild_wildfire_network_simulation(run, setup, reasoning_type_value, settings,
                                        state_log_filename, actions_observations_rewards_log_filename ,
                                        actions_messages_observations_rewards_log_filename,
                                        neighborhoods_log_filename, state_log_header,
                                        actions_observations_rewards_log_header,
                                        actions_messages_observations_rewards_log_header):
        ns = WildfireNetworkSimulation(run, setup, WildfireReasoningType(reasoning_type_value), settings)
        ns.domain = Wildfire(setup, WildfireReasoningType(reasoning_type_value), settings)
        ns.state_log_filename = state_log_filename
        ns.actions_observations_rewards_log_header = actions_observations_rewards_log_header
        ns.actions_messages_observations_rewards_log_header = actions_messages_observations_rewards_log_header
        ns.neighborhoods_log_filename = neighborhoods_log_filename
        ns.state_log_header = state_log_header
        ns.actions_observations_rewards_log_header = actions_observations_rewards_log_header
        ns.actions_messages_observations_rewards_log_header = actions_messages_observations_rewards_log_header
        return ns


def calculate_action_agent_parallel_wrapper(args):
    simulator_instance = args[0]
    return simulator_instance.calculate_action_agent_parallel(args[1], args[2], args[3])


cdef class WildfireNetworkSimulation(Simulation):
    def __cinit__(self, int run_number, setup=None, reasoning_type=WildfireReasoningType.NOOP, dict settings={}):
        # set both of the random seeds
        np.random.seed(run_number + 1000 * setup)
        srand(run_number + 1000 * setup)

        # create the domain
        self.domain = Wildfire(setup, reasoning_type, settings)
        self.settings = settings
        self.setup = setup
        self.run_number = run_number
        self.reasoning_type_value = int(reasoning_type)

        # create the filenames
        r_type = str(reasoning_type).replace("WildfireReasoningType.","")
        if reasoning_type == WildfireReasoningType.POMCPPF\
                or reasoning_type == WildfireReasoningType.IPOMCPPF\
                or reasoning_type == WildfireReasoningType.IPOMCPPFComm:
            r_type += "-" + str(int(round(settings["epsilon_p"] * 100))) + "e"

        if reasoning_type == WildfireReasoningType.IPOMCPPF\
                or reasoning_type == WildfireReasoningType.IPOMCPPFComm:
            r_type += "-" + str(settings["level"]) + "l"

        self.state_log_filename = "Wildfire_States_" \
                                  + r_type \
                                  + "_Setup" + str(self.domain.settings.SETUP) \
                                  + "_Traj" + str(settings["trajectories"]) \
                                  + "_Run" + str(run_number) \
                                  + ".csv"
        self.actions_observations_rewards_log_filename = "Wildfire_AOR_" \
                                                         + r_type \
                                                         + "_Setup" + str(self.domain.settings.SETUP) \
                                                         + "_Traj" + str(settings["trajectories"]) \
                                                         + "_Run" + str(run_number) \
                                                         + ".csv"
        self.actions_messages_observations_rewards_log_filename = "Wildfire_AMOR_" \
                                                                  + r_type \
                                                                  + "_Setup" + str(self.domain.settings.SETUP) \
                                                                  + "_Traj" + str(settings["trajectories"]) \
                                                                  + "_Run" + str(run_number) \
                                                                  + ".csv"
        self.neighborhoods_log_filename = "Wildfire_Neighborhoods_" \
                                          + r_type \
                                          + "_Setup" + str(self.domain.settings.SETUP) \
                                          + "_Traj" + str(settings["trajectories"]) \
                                          + "_Run" + str(run_number) \
                                          + ".csv"

        # create the log headers
        self.state_log_header = "Step," + ",".join(["f" + str(i) for i in range(self.domain.settings.FIRES)]
                                          + ["s" + str(i) for i in range(self.domain.num_agents())])
        self.actions_observations_rewards_log_header = "Step,Agent,Action,Observation,Reward"
        self.actions_messages_observations_rewards_log_header = "Step,Agent,Action,Message,Observation,Reward"
        # log the agents' neighborhoods
        if reasoning_type == WildfireReasoningType.POMCPPF\
                or reasoning_type == WildfireReasoningType.IPOMCPPF\
                or reasoning_type == WildfireReasoningType.IPOMCPPFComm:
            self.log_neighborhoods()

    def __reduce__(self):
        return (rebuild_wildfire_network_simulation, (self.run_number, self.setup, self.reasoning_type_value, self.settings,
                                                      self.state_log_filename, self.actions_observations_rewards_log_filename ,
                                                      self.actions_messages_observations_rewards_log_filename,
                                                      self.neighborhoods_log_filename, self.state_log_header,
                                                      self.actions_observations_rewards_log_header,
                                                      self.actions_messages_observations_rewards_log_header))

    cpdef WildfireJointActionComm client_run(self, int client_num, WildfireState state, WildfireInternalStates internal_states, int total_clients):
        cdef bint using_comm = self.domain.settings.COMMUNICATION
        cdef int agents_n = self.domain.num_agents()
        cdef int agent_num, agent_i, result_i
        cdef list results
        cdef WildfireJointActionComm joint_action_comm_per_agent

        # Calcultaing the agents to model
        cdef int[:] client_agents = np.array_split(np.arange(self.domain.num_agents(), dtype=np.intc), total_clients)[client_num]

        # Updating states and internal_states for the all agents on client side
        print(f'Client_Agents:{np.array(client_agents)}')
        print(f'Run:{self.run_number}, Fires:{np.array(state.values)}, Suppressants:{np.array(internal_states.values)}')

        for agent_i in range(agents_n):
            agent = self.domain.agents[agent_i]
            agent.set_current_state(state)
            agent.set_current_suppressant(internal_states.values[agent_i])

        joint_action_comm = WildfireJointActionComm(agents_n)
        joint_action_comm.actions = np.full((agents_n), fill_value=-99, dtype=np.intc)
        joint_action_comm.messages = np.full((agents_n), fill_value=-99, dtype=np.intc)

        for agent_i in range(len(client_agents)):
            agent_num = client_agents[agent_i]
            agent = self.domain.agents[agent_num]
            action = agent.choose_action()

            if using_comm:
                action_comm = <ActionComm> action

                joint_action_comm.actions[agent_num] = action_comm.index
                joint_action_comm.messages[agent_num] = action_comm.message
            else:
                joint_action_comm.actions[agent_num] = action.index

        return joint_action_comm

    cpdef list server_run(self, WildfireState state, WildfireInternalStates internal_states, WildfireJointActionComm joint_action_comm):
        # give the initial state information to the agents
        cdef int agents_n = self.domain.num_agents()
        cdef int agent_num
        cdef WildfireAgent agent
        cdef bint using_comm = self.domain.settings.COMMUNICATION

        # run the simulation
        cdef int client_num
        cdef int step
        cdef list observations
        cdef WildfireAction action
        cdef WildfireObservation observation
        cdef WildfireObservationComm observation_comm
        cdef WildfireConfiguration configuration
        cdef WildfireState next_state
        cdef WildfireInternalStates next_internal_states
        cdef np.ndarray[np.double_t, ndim=1] rewards

        # create the corresponding configuration
        if using_comm:
            configuration = self.domain.create_configuration(-1, joint_action_comm, False)
            next_internal_states = self.domain.sample_next_internal_states(-1, internal_states, joint_action_comm)
        else:
            joint_action = WildfireJointAction(agents_n)
            joint_action.actions = joint_action_comm.actions
            configuration = self.domain.create_configuration(-1, joint_action, False)
            next_internal_states = self.domain.sample_next_internal_states(-1, internal_states, joint_action)
        print("Configuration:", np.array(configuration.actions))

        # sample a next state
        # NOTE: in the Wildfire domain, transitions are agent independent
        next_state = self.domain.sample_next_state_configuration(-1, state, configuration, None)
        
        # calculate rewards and observations for each agent
        rewards = np.ndarray((agents_n,), dtype=np.double)
        observations = []
        for agent_num in range(agents_n):
            agent = self.domain.agents[agent_num]
            if using_comm:
                action = WildfireAction(joint_action_comm.actions[agent_num])
                action_comm = WildfireActionComm(joint_action_comm.actions[agent_num],
                                                 joint_action_comm.messages[agent_num])
            else:
                action = WildfireAction(joint_action_comm.actions[agent_num])

            observation = self.domain.sample_observation_configuration(agent_num, state, configuration, action,
                                                                       next_state)

            if using_comm:
                neighbors = <list> self.domain.modeling_neighborhoods_per_agent[agent_num]
                neighbors_n = len(neighbors)

                # create the collection of messages that agent_num would receive
                observation_comm = WildfireObservationComm(observation.fire_change, neighbors_n)

                for neighbor_i in range(neighbors_n):
                    neighbor = <WildfireAgent> neighbors[neighbor_i]
                    observation_comm.messages[neighbor_i] = joint_action_comm.messages[neighbor.agent_num]

                observations.append(observation_comm)

                rewards[agent_num] = self.domain.reward_configuration_with_comm(agent_num, state, internal_states,
                                                                                configuration, action_comm,
                                                                                next_state, next_internal_states)
            else:
                observations.append(observation)

                rewards[agent_num] = self.domain.reward_configuration(agent_num, state, internal_states,
                                                                      configuration, action, next_state,
                                                                      next_internal_states)
        return [rewards, observations, next_state, next_internal_states]

    cpdef void update_reward_obs(self, np.ndarray[np.double_t, ndim=1] rewards, list observations,  WildfireState next_state, WildfireInternalStates next_internal_states, int client_num, int total_clients):
        cdef int agents_n = self.domain.num_agents()
        cdef WildfireAgent agent
        cdef int agent_i, agent_num
        cdef int[:] client_agents = np.array_split(np.arange(self.domain.num_agents(), dtype=np.intc), total_clients)[client_num]
        # give this information to the agents
        for agent_i in range(len(client_agents)):
            agent_num = client_agents[agent_i]
            agent = self.domain.agents[agent_num]
            agent.receive_reward(rewards[agent_num])
            agent.make_observation(next_state, next_internal_states.values[agent_num], observations[agent_num])

    cpdef void update_log_state(self,int step, WildfireState state, WildfireInternalStates internal_states):
        self.log_state(step,state,internal_states)

    cpdef void update_log_actions_observations_rewards_messages(self, int step, WildfireJointActionComm joint_action_comm, list observations, np.ndarray[np.double_t, ndim=1] rewards):
        cdef bint using_comm = self.domain.settings.COMMUNICATION
        cdef WildfireJointAction joint_action
        if using_comm:
            self.log_actions_messages_observations_rewards(step, joint_action_comm, observations, rewards)
        else:
            joint_action = WildfireJointAction(self.domain.num_agents())
            joint_action.actions = joint_action_comm.actions
            self.log_actions_observations_rewards(step, joint_action, observations, rewards)

    cdef void log_neighborhoods(self):
        # find the largest neighborhood
        cdef int largest_neighborhood = 0
        cdef int agents_n = self.domain.num_agents()
        cdef int agent_i, neighborhood_n, neighbor_i
        cdef list agent_models
        cdef WildfireAgent agent, neighbor

        for agent_i in range(agents_n):
            agent = <WildfireAgent> self.domain.agents[agent_i]
            agent_models = agent.reasoning.planner.agent_models

            neighborhood_n = len(agent_models)
            if neighborhood_n > largest_neighborhood:
                largest_neighborhood = neighborhood_n

        # create the header
        self.neighborhoods_log_header = "Agent," + ",".join(["Neighbor" + str(i) for i in range(largest_neighborhood)])

        cdef str line
        with open(self.neighborhoods_log_filename, "w") as file:
            # write the header
            file.write(self.neighborhoods_log_header + "\n")

            # save each agent's modeled neighborhood
            for agent_i in range(agents_n):
                agent = <WildfireAgent> self.domain.agents[agent_i]
                agent_models = agent.reasoning.planner.agent_models
                neighborhood_n = len(agent_models)

                # start the line with this agent
                line = str(agent.agent_num)

                # add each neighbor
                for neighbor_i in range(neighborhood_n):
                    neighbor = <WildfireAgent> agent_models[neighbor_i]

                    line += "," + str(neighbor.agent_num)

                # save the line
                file.write(line + "\n")
