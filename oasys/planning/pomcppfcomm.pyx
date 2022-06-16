from oasys.agents.frame import Frame
from oasys.structures.pomdp_structures import Observation
import numpy as np
import time

from oasys.agents.frame cimport Frame
from oasys.domains.domain cimport get_domain
from oasys.structures.pomdp_structures cimport Observation
cimport numpy as np
cimport cython

from libc.stdlib cimport rand, RAND_MAX
cdef extern from "math.h":
    double sqrt(double m)
    double log(double m)


def log_info(data, log_time=True):
    """Logs the data into a file

    Args:
        data:
            The information that needs to be logged into the file
        log_time:(bool)
            Logs the time of the information if true else just logs the
            information
    """
    with open("pomcppfcomm_logger.txt", "a") as file1:
        sec = time.time()
        local_time = time.ctime(sec)
        content = f'{data}\n\n'
        if log_time:
            content = f'{local_time}\n' + content
        # Writing data to a file
        file1.write(content)


cdef class POMCPPFCommPlanner:
    def __cinit__(self, Agent agent, int horizon, double gamma, double ucb_c, list agent_models, int trajectories,
                  bint fully_observable_environment, int[:] possible_messages_per_agent,
                  MessageEstimator message_estimator,list possible_next_states=None):
        self.agent = agent
        self.domain = get_domain()
        self.horizon = horizon
        self.gamma = gamma
        self.agent_models = agent_models
        self.trajectories = trajectories
        self.ucb_c = ucb_c
        self.q_sensitivity = 0.001
        self.last_action_node = None
        self.possible_next_states = possible_next_states
        self.possible_messages_per_agent = possible_messages_per_agent
        self.message_estimator = message_estimator
        self.fully_observable_environment = fully_observable_environment
        self.tree = BeliefNode(-1, -1, -1)
        self.last_action_observation_probabilities = dict()
        # count the number of agents modeled per each frame
        cdef Agent neighbor
        cdef int[:] total_agents_per_frame = self.domain.num_agents_per_frame()
        cdef int frames_n = len(total_agents_per_frame)
        cdef int modeled_agents_n = len(agent_models)
        cdef int agent_i

        self.agents_modeled_per_frame = np.zeros((frames_n,), dtype=np.intc)
        for agent_i in range(modeled_agents_n):
            neighbor = <Agent> agent_models[agent_i]
            self.agents_modeled_per_frame[neighbor.frame.index] += 1

        cdef Frame frame
        cdef int frame_i
        self.available_actions_per_frame = np.zeros((frames_n,), dtype=np.intc)
        for frame_i in range(frames_n):
            frame = <Frame> self.domain.frames[frame_i]
            self.available_actions_per_frame[frame_i] = len(frame.get_all_available_actions())


    @cython.boundscheck(False)
    @cython.nonecheck(False)
    @cython.wraparound(False)
    @cython.profile(True)
    @cython.cdivision(True)
    cpdef ActionComm create_plan(self, int[:] messages):
        cdef int trajectories = self.trajectories
        cdef int traj
        for traj in range(trajectories):
            # print("Traj:", traj)
            self.update_tree(self.tree.particle_filter, messages, 0, self.tree)

        # choose the best action
        cdef ActionNode best_action_node
        if self.agent.settings["level"] == 0:
            print("Agent:", self.agent.agent_num, "Internal State:", self.tree.particle_filter.particles[0].partial_internal_states.values[0], "Iterations:", trajectories)
            best_action_node = self.tree.argmax_q(self.q_sensitivity, self.ucb_c)
        else:
            best_action_node = self.tree.argmax_q(self.q_sensitivity)

        # save the action taken
        self.last_action_node = best_action_node
        self.last_action_observation_probabilities  = self.tree.calculate_last_action_observation_probabilities(self.agent.agent_num,
                                                                                                                self.last_action_node)

        # pick a message
        cdef int message = self.message_estimator.estimate_message(self.tree.particle_filter.particles[0].partial_internal_states.values[0],
                                                                   self.possible_messages_per_agent[0],
                                                                   self.agent)
        return ActionComm(best_action_node.action.index, message)


    @cython.boundscheck(False)
    @cython.nonecheck(False)
    @cython.wraparound(False)
    @cython.profile(True)
    @cython.cdivision(True)
    cpdef list root_q_values(self):
        cdef list q_values = []

        cdef int actions_n = len(self.domain.actions)
        cdef int action_i
        cdef double low_value = -999999999
        for action_i in range(actions_n):
            q_values.append(low_value)

        actions_n = len(self.tree.action_nodes)
        for action_i in range(actions_n):
            action_node = <ActionNode> self.tree.action_nodes[action_i]
            q = action_node.q_value

            if action_node.visits > 0:
                q_values[action_node.action.index] = q

        return q_values


    @cython.boundscheck(False)
    @cython.nonecheck(False)
    @cython.wraparound(False)
    @cython.profile(True)
    @cython.cdivision(True)
    cdef double update_tree(self, ParticleFilter particle_filter, int[:] messages, int h, BeliefNode belief_node):
        # update the particle filter
        if h > 0:
            belief_node.particle_filter.particles.extend(particle_filter.particles)
            belief_node.particle_filter.weights.extend(particle_filter.weights)
            belief_node.particle_filter.normalize()

        # have we reached the end of the tree?
        if h >= self.horizon:
            return 0.0

        # is belief_node a new leaf?
        if belief_node.visits == 0:
            belief_node.visits += 1 # the sum of weights in the particle filter is 1 (see note at end of function)
            return self.rollout_all(particle_filter, h)

        cdef bint first_sample = False
        cdef Particle particle
        if len(belief_node.action_nodes) == 0:
            if self.fully_observable_environment:
                particle = <Particle> particle_filter.particles[0]
                belief_node.create_action_nodes(self.agent, particle.state, particle.partial_internal_states.values[0])
            else:
                belief_node.create_action_nodes(self.agent)
            first_sample = True

        cdef int agent_num = self.agent.agent_num
        cdef ActionNode action_node
        if self.fully_observable_environment:
            particle = <Particle> particle_filter.particles[0]
            action_node = belief_node.argmax_ucb1(self.ucb_c, self.agent, particle.state,
                                               particle.partial_internal_states.values[0])
        else:
            action_node = belief_node.argmax_ucb1(self.ucb_c, self.agent)
        cdef Action action = action_node.action

        cdef list observations_particle_filters = list()
        cdef list observation_weights = list()
        cdef list all_messages_list = list()
        cdef list all_messages_weights = list()
        cdef list all_observations = self.domain.observations
        cdef int internal_states_n = len(self.agent.frame.possible_internal_states)
        cdef int obs_n = len(all_observations)
        cdef int obs_i, internal_state

        for internal_state in range(internal_states_n):
            for obs_i in range(obs_n):
                observations_particle_filters.append(ParticleFilter())
                observation_weights.append(0)

                all_messages_list.append(list())
                all_messages_weights.append(list())

        cdef int particles_n = len(particle_filter.particles)
        cdef double reward = 0.0
        cdef double obs_weight_sum = 0.0
        cdef int particle_i, next_internal_state, index, index_base
        cdef Particle new_particle
        cdef double weight, prob
        cdef State state, next_state
        cdef PartialInternalStates internal_states, next_internal_states
        cdef PartialJointActionComm partial_joint_action
        cdef Configuration configuration
        cdef Observation observation
        cdef ParticleFilter new_particle_filter
        cdef PartialInternalStates message_internal_states

        cdef PartialInternalStates next_internal_states_message_estimator
        cdef list all_next_internal_states_message_estimator = list()
        cdef list all_next_joint_actions = list()
        cdef list internal_state_stack = self.message_estimator.internal_state_stack
        cdef list messages_list, messages_weights

        for particle_i in range(particles_n):
            particle = <Particle> particle_filter.particles[particle_i]
            weight = <double> particle_filter.weights[particle_i]

            state = particle.state
            internal_states = particle.partial_internal_states

            # the joint action can differ for different particles because the f function can be stochastic
            partial_joint_action = self.sample_joint_action_with_comm(action.index, messages)

            # self.agent is the first spot in modeled_joint_action since its partial
            partial_joint_action.actions[0] = action.index
            partial_joint_action.agent_nums[0] = agent_num

            # save the joint action so we have the message we expect to be sent
            all_next_joint_actions.append(partial_joint_action)

            # save the next set of internal states just simulated by the message estimator during joint action sampling
            next_internal_states_message_estimator = <PartialInternalStates> internal_state_stack.pop(len(internal_state_stack) - 1)
            all_next_internal_states_message_estimator.append(next_internal_states_message_estimator)

            # randomly sample a configuration
            configuration = self.sample_configuration(partial_joint_action)

            # simulate the world
            next_state = None
            if self.possible_next_states is not None:
                next_state = self.domain.sample_next_state_configuration_from_possible(agent_num, state, configuration,
                                                                                       action,
                                                                                       self.possible_next_states[state.index])
            else:
                next_state = self.domain.sample_next_state_configuration(agent_num, state, configuration, action)

            next_internal_states = self.domain.sample_next_partial_internal_states(internal_states,
                                                                                   partial_joint_action)
            new_particle = Particle(next_state, next_internal_states)

            reward = reward + weight * self.domain.reward_configuration(agent_num, state, internal_states,
                                                                        configuration, action, next_state,
                                                                        next_internal_states)
            next_internal_state = next_internal_states.values[0]
            index_base = next_internal_state * obs_n

            for obs_i in range(obs_n):
                observation = <Observation> all_observations[obs_i]
                prob = weight * self.domain.observation_probability_configuration(agent_num, state, configuration,
                                                                                  action, next_state, observation)

                index = index_base + obs_i
                observation_weights[index] = (<double> observation_weights[index]) + prob
                obs_weight_sum += prob

                # since this is a level 0 agent, it maintains no beliefs about neighbors' internal states
                # so messages are not observations
                # this means we only need on particle filter per observation (shared by all messages)
                new_particle_filter = <ParticleFilter> observations_particle_filters[index]
                new_particle_filter.particles.append(new_particle)
                new_particle_filter.weights.append(prob)

                # Saving messages and weights for internal_state and observation pair combination
                messages_list = <list> all_messages_list[index]
                messages_list.append(particle_i) # particle_i is an index into all_next_joint_actions for the message

                messages_weights = <list> all_messages_weights[index]
                messages_weights.append(prob)

        # Normalizing observation weights
        for internal_state in range(internal_states_n):
            index_base = internal_state * obs_n
            for obs_i in range(obs_n):
                index = index_base + obs_i
                observation_weights[index] = (<double> observation_weights[index]) / obs_weight_sum

        # sampling an observation/internal state pair
        cdef double rand_val = rand() / (RAND_MAX + 1.0)
        cdef int indices = internal_states_n * obs_n
        for index in range(indices):
            prob = <double> observation_weights[index]

            if prob > rand_val:
                break
            else:
                rand_val -= prob

        obs_i = index % obs_n
        next_internal_state = index // obs_n
        observation = <Observation> all_observations[obs_i]

        # grab the corresponding particle filter and reweight it
        new_particle_filter = <ParticleFilter> observations_particle_filters[index]
        new_particle_filter.normalize()
        new_particle_filter.resample_particlefilter()

        # sample a message given the chosen observation/internal state pair
        messages_list = <list> all_messages_list[index]
        messages_weights = <list> all_messages_weights[index]
        cdef int weight_i, weights_n = len(messages_weights)
        cdef double weights_sum = 0.0

        cdef int next_messages_index
        if weights_n == 1:
            next_messages_index = <int> messages_list[0]
        else:
            # Normalizing weights for the messages sent with the sampled observation
            for weight_i in range(weights_n):
                weights_sum = weights_sum + <double> messages_weights[weight_i]

            for weight_i in range(weights_n):
                messages_weights[weight_i] = <double> (messages_weights[weight_i]/weights_sum)

            # Sampling a message
            rand_val = rand() / (RAND_MAX + 1.0)
            for weight_i in range(weights_n):
                prob = <double> messages_weights[weight_i]
                if prob > rand_val:
                    break
                else:
                    rand_val -= prob

            next_messages_index = <int> messages_list[weight_i]

        # grab the sampled next messages
        partial_joint_action = <PartialJointAction> all_next_joint_actions[next_messages_index]
        cdef int[:] next_messages = partial_joint_action.messages

        # push the corresponding next internal states into the message estimator's stack for the next step
        next_internal_states_message_estimator = <PartialInternalStates> all_next_internal_states_message_estimator[next_messages_index]
        self.message_estimator.internal_state_stack.append(next_internal_states_message_estimator)

        # create the observation comm so we can get the branch to the next belief node
        cdef int models_n = len(self.agent_models)
        cdef ObservationComm observation_comm = ObservationComm(observation.index, models_n)
        for index in range(models_n):
            observation_comm.messages[index] = next_messages[index + 1]

        cdef long message_index = observation_comm.calculate_message_index(self.possible_messages_per_agent)
        cdef BeliefNode next_belief_node = action_node.get_belief(next_internal_state, observation_comm.index,
                                                                  message_index)
        cdef double R = reward + self.gamma * self.update_tree(new_particle_filter, next_messages, h+1,
                                                               next_belief_node)

        belief_node.visits += 1 # as long as reward is a weighted sum above, this needs to be a 1 (total weight is 1)
        action_node.visits += 1 # so if these are not 1, then the average below for Q is broken

        # update action_node's Q value
        action_node.q_value += (R - action_node.q_value) / action_node.visits

        # pop the last internal states used for the message estimator
        self.message_estimator.pop_internal_state_stack()

        return R


    @cython.boundscheck(False)
    @cython.nonecheck(False)
    @cython.wraparound(False)
    @cython.profile(True)
    @cython.cdivision(True)
    cdef double rollout(self, Particle particle, int h):
        cdef Configuration configuration
        cdef Action action
        cdef State next_state
        cdef PartialInternalStates next_internal_states
        cdef PartialJointAction random_joint_action

        cdef int agent_num = self.agent.agent_num
        cdef double R = 0.0
        cdef double running_gamma = 1.0
        cdef State state = particle.state
        cdef PartialInternalStates internal_states = particle.partial_internal_states

        for h in range(h, self.horizon):
            # simulate the environment
            random_joint_action = self.sample_random_joint_action(state)
            configuration = self.sample_configuration(random_joint_action)
            action = self.agent.sample_action(state, internal_states.values[0])

            # self.agent is the first spot in modeled_joint_action since its partial
            random_joint_action.actions[0] = action.index
            random_joint_action.agent_nums[0] = agent_num

            if self.possible_next_states is not None:
                next_state = self.domain.sample_next_state_configuration_from_possible(agent_num, state, configuration,
                                                                                       action,
                                                                                       self.possible_next_states[state.index])
            else:
                next_state = self.domain.sample_next_state_configuration(agent_num, state, configuration, action)

            next_internal_states = self.domain.sample_next_partial_internal_states(internal_states,
                                                                           random_joint_action)

            # self.agent is the first spot in internal_states since its partial
            reward = self.domain.reward_configuration(agent_num, state, internal_states, configuration, action,
                                                      next_state, next_internal_states)

            # update the cumulative reward
            R = R + running_gamma * reward

            # prepare for the next iteration
            state = next_state
            internal_states = next_internal_states
            running_gamma = running_gamma * self.gamma

        return R

    @cython.boundscheck(False)
    @cython.nonecheck(False)
    @cython.wraparound(False)
    @cython.profile(True)
    cdef PartialJointAction sample_random_joint_action(self, State state):
        # creating a partial joint action
        cdef int models_n = len(self.agent_models)
        cdef PartialJointAction random_joint_action = PartialJointAction(models_n + 1)
        random_joint_action.agent_nums[0] = self.agent.agent_num

        cdef Agent neighbor
        cdef int[:] available_actions
        cdef int model_i, available_actions_n, agent_i, rand_i, random_action, neighbor_i

        # pick a random action for each agent that is modeled at higher level of reasoning
        for model_i in range(models_n):
            neighbor = <Agent> self.agent_models[model_i]
            neighbor_i = model_i + 1 # self.agent is always first

            available_actions = neighbor.frame.get_all_available_actions()
            available_actions_n = len(available_actions)
            if available_actions_n > 1:
                # Considering neighbors will always fight if they have suppressant
                # TODO: move this into WildfireFrame so that this planner works with CyberSec
                available_actions_n = available_actions_n - 1

            # pick a random available action for that agent
            rand_i = rand() % available_actions_n
            random_action = available_actions[rand_i]

            random_joint_action.actions[neighbor_i] = random_action
            random_joint_action.agent_nums[neighbor_i] = neighbor.agent_num

        return random_joint_action

    @cython.boundscheck(False)
    @cython.nonecheck(False)
    @cython.wraparound(False)
    @cython.profile(True)
    cdef PartialJointActionComm sample_joint_action_with_comm(self,  int action_index, int[:] messages):
        cdef int models_n = len(self.agent_models)

        # creating a partial joint action
        cdef PartialJointActionComm joint_action = PartialJointActionComm(models_n + 1)
        joint_action.actions[0] = action_index
        joint_action.agent_nums[0] = self.agent.agent_num

        cdef Agent neighbor
        cdef int[:] available_actions
        cdef int model_i, neighbor_i, internal_state, message, action
        cdef int message_i, messages_n = len(messages)

        # pick an action for each agent based on the received message
        for model_i in range(models_n):
            neighbor = <Agent> self.agent_models[model_i]
            neighbor_i = model_i + 1  # self.agent is always first

            message = messages[neighbor_i]
            action = self.message_estimator.estimate_action(message, neighbor)

            joint_action.actions[neighbor_i] = action
            joint_action.agent_nums[neighbor_i] = neighbor.agent_num

        # predict the next messages for these agents
        self.message_estimator.next_messages(messages, joint_action, self.agent_models,
                                             self.possible_messages_per_agent, self.domain)

        return joint_action


    @cython.boundscheck(False)
    @cython.nonecheck(False)
    @cython.wraparound(False)
    @cython.profile(True)
    @cython.cdivision(True)
    cdef double rollout_all(self, ParticleFilter particle_filter, int h):
        cdef double R = 0.0
        cdef int particles_n = len(particle_filter.particles)

        cdef Particle particle
        cdef double weight
        cdef int particle_i

        for particle_i in range(particles_n):
            particle = <Particle> particle_filter.particles[particle_i]
            weight = <double> particle_filter.weights[particle_i]
            R = R + weight * self.rollout(particle, h)

        return R


    @cython.boundscheck(False)
    @cython.nonecheck(False)
    @cython.wraparound(False)
    @cython.profile(True)
    @cython.cdivision(True)
    cdef Configuration sample_configuration(self, PartialJointAction joint_action):
        cdef Configuration small_configuration = self.domain.create_configuration_from_partial(self.agent.agent_num,
                                                                                               joint_action)
        if len(joint_action.actions) == self.domain.num_agents():
            return small_configuration

        # calculate the proportions for each frame/action
        cdef int frames_n = len(self.agents_modeled_per_frame)
        cdef int config_n = len(small_configuration.actions)
        cdef list prop_per_action = list()
        cdef int frame_offset = 0
        cdef int actions_n, config_i, frame_i, action_i
        cdef double prob, frame_size

        for frame_i in range(frames_n):
            frame_size = self.agents_modeled_per_frame[frame_i]
            actions_n = self.available_actions_per_frame[frame_i]

            for action_i in range(actions_n):
                config_i = action_i  + frame_offset
                prob = small_configuration.actions[config_i] / frame_size
                prop_per_action.append(prob)

            frame_offset += actions_n

        # print(np.array(prop_per_action), np.sum(prop_per_action), np.sum(small_configuration.actions))

        # randomly sample a configuration based on those proportions
        cdef Configuration configuration = Configuration(config_n)

        cdef int[:] agents_per_frame = self.domain.num_agents_per_frame()
        cdef int agent_i, last, agents_n
        cdef double rand_val
        frame_offset = 0
        for frame_i in range(frames_n):
            actions_n = self.available_actions_per_frame[frame_i]

            agents_n = agents_per_frame[frame_i]
            if frame_i == self.agent.frame.index:
                # don't include ourselves in the configuration
                agents_n -= 1

            # randomly sample an action for each agent of this frame
            for agent_i in range(agents_n):
                rand_val = rand() / (RAND_MAX + 1.0)

                for action_i in range(actions_n):
                    config_i = action_i  + frame_offset
                    prob = <double> prop_per_action[config_i]

                    if prob > 0.0:
                        last = config_i

                        if prob > rand_val:
                            break
                        else:
                            rand_val -= prob

                # increase the count of the chosen action
                configuration.actions[last] += 1

            frame_offset += actions_n

        return configuration


    cdef void make_observation(self, State next_state, int next_internal_state, ObservationComm observation):
        # update the root of the tree
        cdef long message_index = observation.calculate_message_index(self.possible_messages_per_agent)
        self.tree = self.last_action_node.get_belief(next_internal_state, observation.index, message_index)
        self.tree.parent_action_node = None # for garbage collection
        self.last_action_node = None


    cdef void reset(self):
        self.tree = BeliefNode(-1, -1, -1)
        self.last_action_node = None


cdef class BeliefNode:
    def __cinit__(self, int observation_index, long message_index, int internal_state,
                  ActionNode parent_action_node=None):
        self.observation_index = observation_index
        self.message_index = message_index
        self.internal_state = internal_state
        self.visits = 0
        self.particle_filter = ParticleFilter()
        self.action_nodes = list()
        self.parent_action_node = parent_action_node


    @cython.boundscheck(False)
    @cython.nonecheck(False)
    @cython.wraparound(False)
    @cython.profile(True)
    cdef void create_action_nodes(self, Agent agent, State state=None, int internal_state=0):
        cdef int[:] available_actions
        if state is not None:
            available_actions = agent.frame.get_available_actions_per_state(state, internal_state)
        else:
            available_actions = agent.frame.get_all_available_actions()

        cdef int actions_n = len(available_actions)
        cdef int actions_i
        cdef ActionNode action_node
        cdef Action action

        for action_i in range(actions_n):
            action = <Action> agent.domain.actions[available_actions[action_i]]
            action_node = ActionNode(action, action_i, self)
            #action_node.create_belief_nodes(self.max_internal_state, self.max_observation)
            self.action_nodes.append(action_node)


    @cython.boundscheck(False)
    @cython.nonecheck(False)
    @cython.wraparound(False)
    @cython.profile(True)
    cdef ActionNode argmax_ucb1(self, double ucb_c, Agent agent, State state = None, int internal_state = -1):
        # find the ActionNode with the highest UCB-1 value
        cdef ActionNode best_action_node = None
        cdef ActionNode action_node
        cdef int[:] available_actions

        if state is not None:
            available_actions = agent.frame.get_available_actions_per_state(state, internal_state)
        else:
            available_actions = agent.frame.get_all_available_actions()

        cdef int n_visits = self.visits
        cdef int actions_n = len(self.action_nodes)
        cdef int available_actions_n = len(available_actions)
        cdef int action_i, available_action_i, rand_i

        cdef double ucb1
        cdef double best_ucb1 = -99999999
        cdef bint available

        cdef list best_action_nodes = []
        cdef int best_action_nodes_n = 0

        for action_i in range(actions_n):
            action_node = <ActionNode> self.action_nodes[action_i]

            # is this action possible in (state, internal_state)?
            available = state is None # if state is None, then all actions have action nodes
            if not available:
                for available_action_i in range(available_actions_n):
                    if action_node.action.index == available_actions[available_action_i]:
                        available = True
                        break
            if available:
                if action_node.visits > 0:
                    ucb1 = action_node.q_value + ucb_c * sqrt(log(n_visits) / action_node.visits)
                else:
                    ucb1 = 10000000

                # print("A:", action_node.action.index, "V:", action_node.visits, "Q:", action_node.q_value, "UCB:", ucb1)

                if ucb1 > best_ucb1:
                    best_ucb1 = ucb1
                    best_action_node = action_node
                    best_action_nodes_n = 1
                elif ucb1 == best_ucb1:
                    if best_action_nodes_n == 1:
                        best_action_nodes = list()
                        best_action_nodes.append(best_action_node)

                    best_action_nodes.append(action_node)
                    best_action_nodes_n = best_action_nodes_n + 1

        if best_action_nodes_n == 1:
            return best_action_node
        else:
            rand_i = rand() % best_action_nodes_n
            return best_action_nodes[rand_i]


    @cython.boundscheck(False)
    @cython.nonecheck(False)
    @cython.wraparound(False)
    @cython.profile(True)
    cdef ActionNode argmax_q(self, double q_sensitivity, double ucb_c=0.0):
        # find the ActionNode with the highest Q value
        cdef ActionNode action_node
        cdef list best_action_nodes = []

        cdef int best_action_nodes_n = 0
        cdef int actions_n = len(self.action_nodes)
        cdef int action_i

        cdef double q, ucb1
        cdef double best_q = -99999999

        for action_i in range(actions_n):
            action_node = <ActionNode> self.action_nodes[action_i]
            q = action_node.q_value

            if action_node.visits > 0:
                if q > best_q + q_sensitivity:
                    best_q = q
                    best_action_nodes = [action_node]
                    best_action_nodes_n = 1
                elif q > best_q - q_sensitivity:
                    best_action_nodes.append(action_node)
                    best_action_nodes_n += 1

            if ucb_c > 0.0:
                if action_node.visits == 0:
                    ucb1 = 10000000
                else:
                    ucb1 = q + ucb_c * sqrt(log(self.visits) / action_node.visits)
                print("Action:", action_node.action.index, "q:", action_node.q_value,
                      "visits:", action_node.visits, "ucb1:", ucb1)

        # randomly break ties, if needed
        cdef int rand_i
        if best_action_nodes_n == 1:
            return best_action_nodes[0]
        elif best_action_nodes_n >1:
            # one or more best actions are present
            rand_i = rand() % (best_action_nodes_n)
            return best_action_nodes[rand_i]


    @cython.boundscheck(False)
    @cython.nonecheck(False)
    @cython.wraparound(False)
    @cython.profile(True)
    cdef dict calculate_last_action_observation_probabilities(self, int agent_num, ActionNode action_node):
        """Calculates the probability of an observation in the best action_node.
        These probabilities are used to compute probability of message received by an agent

        Args:
            agent_num(int):
                Current Agent number
            action_node(action_node):
                Best action node to which observation probabilities are to be computed
        Returns:
            last_action_all_belief_node_visits(dict):
                The probabilities of all observations in the given action node
        """
        cdef BeliefNode bn
        cdef list belief_nodes
        cdef int belief_node_i
        cdef dict last_action_all_belief_node_visits = dict()
        cdef int action_index
        cdef tuple tup
        belief_nodes = action_node.all_belief_nodes
        action_index = action_node.action.index

        for belief_node_i in range(len(belief_nodes)):
            bn = belief_nodes[belief_node_i]
            tup = tuple((agent_num, action_index, bn.observation_index, bn.message_index))
            if tup in last_action_all_belief_node_visits:
                last_action_all_belief_node_visits[tup] += <double> (bn.visits/ action_node.visits)
            else:
                last_action_all_belief_node_visits[tup] = <double> (bn.visits/ action_node.visits)
        return last_action_all_belief_node_visits


cdef class ActionNode:
    def __cinit__(self, Action action, int available_action_i, BeliefNode parent_belief_node):
        self.action = action
        self.available_action_index = available_action_i
        self.visits = 0
        self.q_value = 0.0
        self.belief_nodes = {}
        self.all_belief_nodes = list() # for iterating over quickly in all_belief_particles
        self.parent_belief_node = parent_belief_node


    cdef BeliefNode get_belief(self, int internal_state, int observation_index, long message_index):
        cdef tuple t = tuple((internal_state, observation_index, message_index))
        cdef BeliefNode belief_node
        if t in self.belief_nodes:
            return self.belief_nodes[t]
        else:
            belief_node = BeliefNode(observation_index, message_index, internal_state, self)
            self.belief_nodes[t] = belief_node
            self.all_belief_nodes.append(belief_node)
            return belief_node


    cdef list all_beliefs(self):
        cdef list all_particles = list()
        cdef list all_visits = list()
        cdef list all_branches = list()
        cdef list internal_state_list, visits_list, branches_list

        # get the maximum internal state
        cdef int children_n = len(self.all_belief_nodes)
        cdef int child_i, internal_state
        cdef int max_internal_state = -1
        cdef BeliefNode node
        for child_i in range(children_n):
            node = <BeliefNode> self.all_belief_nodes[child_i]
            internal_state = node.internal_state
            if internal_state > max_internal_state:
                max_internal_state = internal_state

        # add lists for each internal state
        for internal_state in range(max_internal_state + 1):
            all_particles.append(list())
            all_visits.append(list())
            all_branches.append(list())
        all_particles.append(all_visits)
        all_particles.append(all_branches)

        cdef tuple branch
        for child_i in range(children_n):
            node = <BeliefNode> self.all_belief_nodes[child_i]
            internal_state = node.internal_state
            branch = tuple((node.observation_index, node.message_index, node.internal_state))

            internal_state_list = <list> all_particles[internal_state]
            internal_state_list.append(node.particle_filter)

            visits_list = <list> all_visits[internal_state]
            visits_list.append(node.visits)

            branches_list = <list> all_branches[internal_state]
            branches_list.append(branch)

        return all_particles


cdef class MessageEstimator:
    def __cinit__(self):
        self.internal_state_stack = list()


    @cython.boundscheck(False)
    @cython.nonecheck(False)
    @cython.wraparound(False)
    cdef void start_internal_state_stack(self, PartialInternalStates internal_states):
        self.internal_state_stack.clear()
        self.internal_state_stack.append(internal_states)


    cdef void pop_internal_state_stack(self):
        self.internal_state_stack.pop()


    cdef int estimate_message(self, int internal_state, int possible_messages_n, Agent agent):
        # this is domain specific, so we cannot implement it here
        pass


    cdef int estimate_action(self, int message, Agent agent):
        # this is domain specific, so we cannot implement it here
        pass


    @cython.boundscheck(False)
    @cython.nonecheck(False)
    @cython.wraparound(False)
    cdef void next_messages(self, int[:] messages, PartialJointActionComm joint_action, list agent_models,
                              int[:] possible_messages_per_agent, Domain domain):
        cdef int models_n = len(agent_models)

        # grab the current internal state
        cdef int stack_n = len(self.internal_state_stack)
        cdef PartialInternalStates internal_states = <PartialInternalStates> self.internal_state_stack[stack_n - 1]

        # pick messages for each agent
        cdef int model_i, neighbor_i
        cdef Agent neighbor
        for model_i in range(models_n):
            neighbor = <Agent> agent_models[model_i]
            neighbor_i = model_i + 1
            joint_action.messages[neighbor_i] = self.estimate_message(internal_states.values[neighbor_i],
                                                                      possible_messages_per_agent[neighbor_i],
                                                                      neighbor)

        # simulate a next set of internal states based on the predicted actions
        cdef PartialInternalStates next_internal_states = domain.sample_next_partial_internal_states(internal_states,
                                                                                                     joint_action)
        self.internal_state_stack.append(next_internal_states)


    cdef list update_internal_state_prob(self, int message, int possible_messages_n, list prior_probs):
        # this is domain specific, so we cannot implement it here
        pass
