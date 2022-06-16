from oasys.agents.frame import Frame
import numpy as np
import time

from oasys.agents.frame cimport Frame
from oasys.domains.domain cimport get_domain
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
    with open("pomcp_logger.txt", "a") as file1:
        sec = time.time()
        local_time = time.ctime(sec)
        content = f'{data}\n\n'
        if log_time:
            content = f'{local_time}\n' + content
        # Writing data to a file
        file1.write(content)


cdef class POMCPPFPlanner:
    def __cinit__(self, Agent agent, int horizon, double gamma, double ucb_c, list agent_models,
                  bint fully_observable_environment, list possible_next_states=None):
        self.agent = agent
        self.domain = get_domain()
        self.horizon = horizon
        self.gamma = gamma
        self.agent_models = agent_models
        self.ucb_c = ucb_c
        self.q_sensitivity = 0.001
        self.last_action_node = None
        self.possible_next_states = possible_next_states
        self.fully_observable_environment = fully_observable_environment
        self.tree = BeliefNode(-1, -1)

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
    cpdef Action create_plan(self, int trajectories):
        # print('POMCPPFPLanner got called: and particles:',len(self.tree.particle_filter.particles))
        cdef int traj
        for traj in range(trajectories):
            # if traj % 25 == 0:
            #     print("Trajectories:",traj)
            self.update_tree(self.tree.particle_filter, 0, self.tree)

        # choose the best action
        cdef ActionNode best_action_node
        if self.agent.settings["level"] == 0:
            print("Agent:", self.agent.agent_num, "Internal State:", self.tree.particle_filter.particles[0].partial_internal_states.values[0], "Iterations:", trajectories)
            best_action_node = self.tree.argmax_q(self.q_sensitivity, self.ucb_c)
        else:
            best_action_node = self.tree.argmax_q(self.q_sensitivity)

        # save the action taken
        self.last_action_node = best_action_node
        return best_action_node.action


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
    cdef double update_tree(self, ParticleFilter particle_filter, int h, BeliefNode belief_node):
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
        cdef list all_observations = self.domain.observations
        cdef int internal_states_n = len(self.agent.frame.possible_internal_states)
        cdef int obs_n = len(all_observations)
        cdef int obs_i, internal_state
        for internal_state in range(internal_states_n):
            for obs_i in range(obs_n):
                observations_particle_filters.append(ParticleFilter())
                observation_weights.append(0)

        cdef int particles_n = len(particle_filter.particles)
        cdef double reward = 0.0
        cdef double obs_weight_sum = 0.0
        cdef int particle_i, next_internal_state, index, index_base
        cdef Particle new_particle
        cdef double weight, prob
        cdef State state, next_state
        cdef PartialInternalStates internal_states, next_internal_states
        cdef PartialJointAction random_joint_action
        cdef Configuration configuration
        cdef Observation observation
        cdef ParticleFilter new_particle_filter

        for particle_i in range(particles_n):
            particle = <Particle> particle_filter.particles[particle_i]
            weight = <double> particle_filter.weights[particle_i]

            state = particle.state
            internal_states = particle.partial_internal_states
            random_joint_action = self.sample_random_joint_action(state)
            configuration = self.sample_configuration(random_joint_action)

            # self.agent is the first spot in modeled_joint_action since its partial
            random_joint_action.actions[0] = action.index
            random_joint_action.agent_nums[0] = agent_num

            next_state = None
            if self.possible_next_states is not None:
                next_state = self.domain.sample_next_state_configuration_from_possible(agent_num, state, configuration,
                                                                                       action,
                                                                                       self.possible_next_states[state.index])
            else:
                next_state = self.domain.sample_next_state_configuration(agent_num, state, configuration, action)

            next_internal_states = self.domain.sample_next_partial_internal_states(internal_states, random_joint_action)
            new_particle = Particle(next_state, next_internal_states)

            reward = reward + weight * self.domain.reward_configuration(agent_num, state, internal_states,
                                                                        configuration, action, next_state,
                                                                        next_internal_states)
            next_internal_state = next_internal_states.values[0]
            index_base = next_internal_state * obs_n
            for obs_i in range(obs_n):
                observation = <Observation> all_observations[obs_i]
                index = index_base + obs_i
                new_particle_filter = <ParticleFilter> observations_particle_filters[index]

                prob = weight * self.domain.observation_probability_configuration(agent_num, state, configuration,
                                                                                  action, next_state, observation)
                new_particle_filter.particles.append(new_particle)
                new_particle_filter.weights.append(prob)
                observation_weights[index] = (<double> observation_weights[index]) + prob
                obs_weight_sum += prob

        # Normalizing observation weights
        for internal_state in range(internal_states_n):
            index_base = internal_state * obs_n
            for obs_i in range(obs_n):
                index = index_base + obs_i
                observation_weights[index] = (<double> observation_weights[index]) / obs_weight_sum

        # sampling an observation/internal state pair
        cdef double rand_val = rand() / (RAND_MAX + 1.0)
        cdef int last
        cdef int indices = internal_states_n * obs_n
        for index in range(indices):
            prob = <double> observation_weights[index]
            last = index

            if prob > rand_val:
                break
            else:
                rand_val -= prob

        obs_i = last % obs_n
        next_internal_state = last // obs_n
        observation = <Observation> all_observations[obs_i]

        # grab the corresponding particle filter and reweight it
        new_particle_filter = <ParticleFilter> observations_particle_filters[last]
        new_particle_filter.normalize()
        new_particle_filter.resample_particlefilter()

        cdef BeliefNode next_belief_node = action_node.get_belief(observation.index, next_internal_state)
        cdef double R = reward + self.gamma * self.update_tree(new_particle_filter, h+1, next_belief_node)
        belief_node.visits += 1 # as long as reward is a weighted sum above, this needs to be a 1 (total weight is 1)
        action_node.visits += 1 # so if these are not 1, then the average below for Q is broken

        # update action_node's Q value
        action_node.q_value += (R - action_node.q_value) / action_node.visits
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


    cdef void make_observation(self, State next_state, int next_internal_state, Observation observation):
        # update the root of the tree
        self.tree = self.last_action_node.get_belief(observation.index, next_internal_state)
        self.tree.parent_action_node = None # for garbage collection
        self.last_action_node = None


    cdef void reset(self):
        self.tree = BeliefNode(-1, -1)
        self.last_action_node = None


cdef class BeliefNode:
    def __cinit__(self, int observation_index, int internal_state, ActionNode parent_action_node=None):
        self.observation_index = observation_index
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


cdef class ActionNode:
    def __cinit__(self, Action action, int available_action_i, BeliefNode parent_belief_node):
        self.action = action
        self.available_action_index = available_action_i
        self.visits = 0
        self.q_value = 0.0
        self.belief_nodes = {}
        self.all_belief_nodes = list() # for iterating over quickly in all_belief_particles
        self.parent_belief_node = parent_belief_node


    cdef BeliefNode get_belief(self, int observation_index, int internal_state):
        cdef tuple t = tuple((internal_state, observation_index))
        cdef BeliefNode belief_node
        if t in self.belief_nodes:
            return self.belief_nodes[t]
        else:
            belief_node = BeliefNode(observation_index, internal_state, self)
            self.belief_nodes[t] = belief_node
            self.all_belief_nodes.append(belief_node)
            return belief_node


    cdef list all_belief_particles(self):
        cdef list all_particles = list()
        cdef list internal_list

        cdef int children_n = len(self.all_belief_nodes)
        cdef int child_i, particles_n, particle_i, internal_state, obs_i
        cdef int max_internal_state = -1
        cdef int max_obs = -1
        cdef BeliefNode node
        for child_i in range(children_n):
            node = <BeliefNode> self.all_belief_nodes[child_i]
            if node.internal_state > max_internal_state:
                max_internal_state = node.internal_state
            if node.observation_index > max_obs:
                max_obs = node.observation_index

        for internal_state in range(max_internal_state + 1):
            internal_list = list()
            all_particles.append(internal_list)

            # these will be replaced by particle filters
            for obs_i in range(max_obs + 1):
                internal_list.append(None)

            # the eventual list for valid obs_i for this internal_state
            internal_list.append(list())

        cdef int list_i = max_obs + 1
        cdef list valid_obs_i
        for child_i in range(children_n):
            node = <BeliefNode> self.all_belief_nodes[child_i]
            internal_list = <list> all_particles[node.internal_state]
            internal_list[node.observation_index] = node.particle_filter

            valid_obs_i = <list> internal_list[list_i]
            valid_obs_i.append(node.observation_index)

        return all_particles
