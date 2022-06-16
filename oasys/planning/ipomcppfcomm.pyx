from oasys.agents.frame import Frame
from oasys.planning.pomcppfcomm import BeliefNode as BeliefNode_pf, ActionNode as ActionNode_pf
from oasys.structures.planning_structures import Particle, ParticleFilter, NestedParticle
from oasys.structures.pomdp_structures import Action, Observation, ActionComm, ObservationComm
import numpy as np
import time

from oasys.agents.frame cimport Frame
from oasys.domains.domain cimport get_domain
from oasys.planning.pomcppfcomm cimport BeliefNode as BeliefNode_pf, ActionNode as ActionNode_pf
from oasys.structures.pomdp_structures cimport Action, Observation, ActionComm, ObservationComm
from oasys.structures.planning_structures cimport Particle, ParticleFilter, NestedParticle

cimport numpy as np
cimport cython

from libc.stdlib cimport rand, RAND_MAX
from libc.math cimport round
cdef extern from "math.h":
    double sqrt(double m)
    double log(double m)


global filename

def set_filename(run_number):
    global filename
    if run_number:
        filename = "ipomcppfcomm_logger_run"+str(run_number)+".txt"
    else:
        filename = "ipomcppfcomm_logger.txt"


def log_info(data, log_time=True):
    """Logs the data into a file
    Args:
        data:
            The information that needs to be logged into the file
        log_time:(bool)
            Logs the time of the information if true else just logs the
            information
    """
    global filename

    with open(filename, "a") as file1:
        sec = time.time()
        local_time = time.ctime(sec)
        content = f'{data}\n\n'
        if log_time:
            content = f'{local_time}\n' + content
        # Writing data to a file
        file1.write(content)


cdef class IPOMCPPFCommPlanner:
    def __cinit__(self, Agent agent, int level, int horizon, double gamma, double ucb_c,
                  list agent_models, int trajectories, bint fully_observable_environment,
                  int[:] possible_messages_per_agent, MessageEstimator message_estimator,
                  list possible_next_states=None, dict global_pomcppfcomm_policy=None):
        self.agent = agent
        self.domain = get_domain()
        self.horizon = horizon
        self.gamma = gamma
        self.agent_models = agent_models
        self.ucb_c = ucb_c
        self.q_sensitivity = 0.001
        self.last_action_node = None
        self.message_estimator = message_estimator
        self.possible_next_states = possible_next_states
        self.trajectories = trajectories
        self.level = level
        self.fully_observable_environment = fully_observable_environment
        self.possible_messages_per_agent = possible_messages_per_agent
        self.neighbor_next_full_particle_sets = list()
        self.tree = BeliefNode(-1, -1, -1)
        self.use_premodel = self.agent.settings["premodel"]
        self.reuse_tree = False

        self.neighbor_particle_filters = dict()
        if global_pomcppfcomm_policy == None:
            self.global_pomcppfcomm_policy = self.agent.settings["POMCPPFComm_global_policy"]
        else:
            self.global_pomcppfcomm_policy = global_pomcppfcomm_policy
        self.neighbor_planners = self.create_neighbor_planners()
        self.last_action_observation_probabilities = dict()
        set_filename(self.agent.settings['run'])

        # count the number of agents modeled per each frame
        cdef Agent neighbor
        cdef int[:] total_agents_per_frame = self.domain.num_agents_per_frame()
        cdef int frames_n = len(total_agents_per_frame)
        cdef int modeled_agents_n = len(agent_models)
        cdef int agent_i

        self.agents_modeled_per_frame = np.zeros((frames_n,), dtype=np.int32)
        for agent_i in range(modeled_agents_n):
            neighbor = agent_models[agent_i]
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
    @cython.cdivision(True)
    @cython.profile(True)
    cpdef ActionComm create_plan(self, int[:] messages):
        cdef int trajectories = self.trajectories
        cdef int traj

        cdef bint print_traj = not self.agent.settings["premodel"] or \
                               (self.agent.settings["level"] > 1 and self.level == self.agent.settings["level"])

        # print('Level', self.level,'IPOMCPPFPLanner got called and particles:',len(self.tree.particle_filter.particles))
        for traj in range(trajectories):
            self.update_tree(self.tree.particle_filter, messages, 0, self.tree)

            if print_traj and traj % 100 == 0:
                print("Trajectories: ", traj)

        # choose the best action
        cdef ActionNode best_action_node
        if self.level == self.agent.settings["level"]:
            print("Agent:", self.agent.agent_num, "Iterations:", trajectories)
            log_info(f"Agent:{self.agent.agent_num}, Iterations: {trajectories}")
            best_action_node = self.tree.argmax_q(self.q_sensitivity, self.ucb_c)
        else:
            best_action_node = self.tree.argmax_q(self.q_sensitivity)

        # save the action taken
        self.last_action_node = best_action_node
        self.last_action_observation_probabilities  = self.tree.calculate_last_action_observation_probabilities(self.agent.agent_num, self.last_action_node)
        return best_action_node.action


    @cython.boundscheck(False)
    @cython.nonecheck(False)
    @cython.wraparound(False)
    @cython.profile(True)
    @cython.cdivision(True)
    cdef double update_tree(self, ParticleFilter particle_filter, int[:] messages, int h, BeliefNode belief_node):
        # update the particle filter
        if h == 1 or (self.reuse_tree and h > 0):
            belief_node.particle_filter.particles.extend(particle_filter.particles)
            belief_node.particle_filter.weights.extend(particle_filter.weights)

        # have we reached the end of the tree?
        if h >= self.horizon:
            return 0.0

        # is belief_node a new leaf?
        if belief_node.visits == 0:
            belief_node.visits += 1  # the sum of weights in the particle filter is 1 (see note at end of function)
            return self.rollout_all(particle_filter, h)

        cdef bint first_sample = False
        cdef NestedParticle particle
        if len(belief_node.action_nodes) == 0:
            if self.fully_observable_environment:
                particle = <NestedParticle> particle_filter.particles[0]
                belief_node.create_action_nodes(self.agent, particle.state, particle.partial_internal_states.values[0])
            else:
                belief_node.create_action_nodes(self.agent)
                first_sample = True

        cdef int agent_num = self.agent.agent_num
        cdef ActionNode action_node
        if self.fully_observable_environment:
            particle = <NestedParticle> particle_filter.particles[0]
            action_node = belief_node.argmax_ucb1(self.ucb_c, self.agent, particle.state,
                                                  particle.partial_internal_states.values[0])
        else:
            action_node = belief_node.argmax_ucb1(self.ucb_c, self.agent)
        cdef ActionComm action = <ActionComm> action_node.action

        cdef dict observations_particle_filters = dict()
        cdef list observation_weights = list()
        cdef list all_messages_list = list()
        cdef list all_messages_weights = list()
        cdef list all_observations = self.domain.observations
        cdef int internal_states_n = len(self.agent.frame.possible_internal_states)
        cdef int obs_n = len(all_observations)
        cdef int obs_i, internal_state

        for internal_state in range(internal_states_n):
            for obs_i in range(obs_n):
                observation_weights.append(0)

                all_messages_list.append(list())
                all_messages_weights.append(list())

        cdef int models_n = len(self.agent_models)
        cdef int particles_n = len(particle_filter.particles)
        cdef double reward = 0.0
        cdef double obs_weight_sum = 0.0
        cdef int particle_i, next_internal_state, index, index_base
        cdef long message_index
        cdef Particle new_particle
        cdef double weight, prob
        cdef State state, next_state
        cdef PartialInternalStates internal_states, next_internal_states
        cdef PartialJointActionComm modeled_joint_action
        cdef Configuration configuration
        cdef Observation observation
        cdef ObservationComm observation_comm = ObservationComm(-1, models_n) # this is only for message index, so obs index doesn't matter
        cdef ParticleFilter new_particle_filter

        cdef list all_next_joint_actions = list()
        cdef list messages_list, messages_weights
        cdef tuple tup

        for particle_i in range(particles_n):
            particle = <NestedParticle> particle_filter.particles[particle_i]
            weight = <double> particle_filter.weights[particle_i]

            state = particle.state
            internal_states = particle.partial_internal_states

            modeled_joint_action = self.sample_modeled_joint_action(particle, h, messages)

            # self.agent is the first spot in modeled_joint_action since its partial
            modeled_joint_action.actions[0] = action.index
            modeled_joint_action.agent_nums[0] = agent_num
            modeled_joint_action.messages[0] = action.message

            # save the joint action so we have the message we expect to be sent
            all_next_joint_actions.append(modeled_joint_action)

            # randomly sample a configuration
            configuration = self.sample_configuration(modeled_joint_action)

            # simulate the world
            next_state = None
            if self.possible_next_states is not None:
                next_state = self.domain.sample_next_state_configuration_from_possible(agent_num, state, configuration,
                                                                                       action,
                                                                                       self.possible_next_states[state.index])
            else:
                next_state = self.domain.sample_next_state_configuration(agent_num, state, configuration, action)

            next_internal_states = self.domain.sample_next_partial_internal_states(internal_states, modeled_joint_action)

            new_particle = None
            if h < self.horizon - 1:  # don't update beyond the horizon
                new_particle = self.update_nested_beliefs(particle, next_state, next_internal_states,
                                                          modeled_joint_action)

            reward = reward + weight * self.domain.reward_configuration_with_comm(agent_num, state, internal_states,
                                                                                  configuration, action, next_state,
                                                                                  next_internal_states)

            # get the message index
            for index in range(models_n):
                observation_comm.messages[index] = modeled_joint_action.messages[index + 1]
            message_index = observation_comm.calculate_message_index(self.possible_messages_per_agent)

            # update for every observation
            next_internal_state = next_internal_states.values[0]
            index_base = next_internal_state * obs_n

            for obs_i in range(obs_n):
                observation = <Observation> all_observations[obs_i]
                prob = weight * self.domain.observation_probability_configuration(agent_num, state, configuration,
                                                                                  action, next_state, observation)

                index = index_base + obs_i
                observation_weights[index] = (<double> observation_weights[index]) + prob
                obs_weight_sum += prob

                # we have different particle filters per observation and message set pair
                tup = tuple((next_internal_state, obs_i, message_index))
                if tup in observations_particle_filters:
                    new_particle_filter = <ParticleFilter> observations_particle_filters[tup]
                else:
                    new_particle_filter = ParticleFilter()
                    observations_particle_filters[tup] = new_particle_filter

                new_particle_filter.particles.append(new_particle)
                new_particle_filter.weights.append(prob)

                # Saving messages and weights for internal_state and observation pair combination
                messages_list = <list> all_messages_list[index]
                messages_list.append(particle_i)  # particle_i is an index into all_next_joint_actions for the message

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

        # create the observation comm so we can get the sampled particle filter and the branch to the next belief node
        observation_comm = ObservationComm(observation.index, models_n)
        for index in range(models_n):
            observation_comm.messages[index] = next_messages[index + 1]
        message_index = observation_comm.calculate_message_index(self.possible_messages_per_agent)

        tup = (next_internal_state, obs_i, message_index)
        new_particle_filter = <ParticleFilter> observations_particle_filters[tup]
        new_particle_filter.normalize()
        new_particle_filter.resample_particlefilter()

        cdef BeliefNode next_belief_node = action_node.get_belief(next_internal_state, observation.index, message_index)

        cdef double R = reward + self.gamma * self.update_tree(new_particle_filter, next_messages, h+1,
                                                               next_belief_node)
        belief_node.visits += 1 # as long as reward is a weighted sum above, this needs to be a 1 (total weight is 1)
        action_node.visits += 1 # so if these are not 1, then the average below for Q is broken

        # update action_node's Q value
        action_node.q_value += (R - action_node.q_value) / action_node.visits

        return R


    @cython.boundscheck(False)
    @cython.nonecheck(False)
    @cython.wraparound(False)
    @cython.profile(True)
    cdef NestedParticle update_nested_beliefs(self, NestedParticle particle, State next_state,
                                              PartialInternalStates next_internal_states,
                                              PartialJointActionComm modeled_joint_action):

        cdef NestedParticle next_particle = NestedParticle(next_state, next_internal_states, particle.level)

        cdef int models_n = len(self.agent_models)
        cdef int neighbor_level = self.level - 1
        cdef int desired_particles_n = <int> self.agent.settings["particles_n"]
        cdef int model_i, neighbor_i, next_internal_state, n, sets_n, total_visits, set_i, visits, rand_visit, last, \
                 particle_i, particles_n, nested_model_i, nested_neighbor_i, internal_states_n, \
                 neighbor_internal_state, neighbor_message
        cdef double prob, prob_per_particle, rand_val, new_weight
        cdef long next_message_index, branch_message_index
        cdef tuple branch
        cdef list all_neighbor_next_particle_sets, all_visits, all_branches, visits_list, branches_list, \
                  internal_state_particle_filters, all_neighbor_probs, neighbor_probs
        cdef ParticleFilter neighbor_filter, bayes_filter, neighbor_next_filter=ParticleFilter()
        cdef NestedParticle neighbor_particle, nested_particle
        cdef Particle nested_particle_l0
        cdef PartialInternalStates sampled_internal_states, sampled_next_internal_states
        cdef Agent neighbor

        cdef PartialJointActionComm neighbor_joint_action = PartialJointActionComm(models_n + 1)
        cdef ObservationComm our_messages = ObservationComm(-1, models_n)
        for model_i in range(models_n):
            our_messages.messages[model_i] = modeled_joint_action.messages[model_i + 1]
        cdef int[:] neighbors_possible_messages_per_agent = cython.view.array(shape=(models_n + 1,), itemsize=sizeof(int), format="i")
        cdef ObservationComm their_messages = ObservationComm(-1, models_n)

        for model_i in range(models_n):
            neighbor_i = model_i + 1

            # get the neighbor's new belief particles
            next_internal_state = next_internal_states.values[neighbor_i]

            if self.level == 1 and self.fully_observable_environment:
                # the neighbor knows its own internal state
                sampled_next_internal_states = PartialInternalStates(1)
                sampled_next_internal_states.values[0] = next_internal_state
                nested_particle_l0 = Particle(next_state, sampled_next_internal_states)

                # we only need one particle since the environment is fully observable
                neighbor_next_filter = ParticleFilter()
                neighbor_next_filter.particles.append(nested_particle_l0)
                neighbor_next_filter.weights.append(1.0)

                next_particle.nested_particles.append(neighbor_next_filter)

                continue # skip the rest -- this saves us some indentation

            all_neighbor_next_particle_sets = <list> self.neighbor_next_full_particle_sets[model_i]
            n = len(all_neighbor_next_particle_sets)

            if n - 2 > next_internal_state:
                all_visits = <list> all_neighbor_next_particle_sets[n - 2]
                all_branches = <list> all_neighbor_next_particle_sets[n - 1]
                visits_list = <list> all_visits[next_internal_state]
                branches_list = <list> all_branches[next_internal_state]

                internal_state_particle_filters = <list> all_neighbor_next_particle_sets[next_internal_state]
                sets_n = len(internal_state_particle_filters)

                # create their set of messages
                their_messages.messages[...] = our_messages.messages
                their_messages.messages[model_i] = modeled_joint_action.messages[0]

                neighbors_possible_messages_per_agent[...] = self.possible_messages_per_agent
                neighbors_possible_messages_per_agent[0] = self.possible_messages_per_agent[neighbor_i]
                neighbors_possible_messages_per_agent[neighbor_i] = self.possible_messages_per_agent[0]

                next_message_index = their_messages.calculate_message_index(neighbors_possible_messages_per_agent)

                # randomly pick a particle filter based on the number of times its belief node was visited
                total_visits = 0
                for set_i in range(sets_n):
                    visits = <int> visits_list[set_i]
                    branch = <tuple> branches_list[set_i]
                    branch_message_index = <long> branch[1]

                    # did this branch have a matching message?
                    if branch_message_index == next_message_index:
                        total_visits = total_visits + visits

                if total_visits > 0:
                    # randomly sample an observation that occurred with this message
                    rand_visit = rand() % total_visits
                    for set_i in range(sets_n):
                        visits = <int> visits_list[set_i]
                        branch = <tuple> branches_list[set_i]
                        branch_message_index = <long> branch[1]

                        # did this branch have a matching message?
                        if branch_message_index == next_message_index:
                            if rand_visit < visits:
                                last = set_i
                                break
                            else:
                                rand_visit = rand_visit - visits

                    neighbor_next_filter = <ParticleFilter> internal_state_particle_filters[last]
                    neighbor_next_filter.normalize()
                    neighbor_next_filter.resample_particlefilter(desired_particles_n)

                elif self.level == 2:
                    neighbor = <Agent> self.agent_models[model_i]
                    neighbor_filter = <ParticleFilter> particle.nested_particles[model_i]
                    particles_n = len(neighbor_filter.particles)

                    all_neighbor_probs = list()
                    for nested_model_i in range(models_n):
                        internal_states_n = neighbors_possible_messages_per_agent[nested_model_i + 1] - 1 # -1 for No Message
                        neighbor_probs = list()
                        all_neighbor_probs.append(neighbor_probs)

                        for neighbor_internal_state in range(internal_states_n):
                            neighbor_probs.append(0.0)

                    # factor the beliefs per nested neighbor
                    for particle_i in range(particles_n):
                        nested_particle = <NestedParticle> neighbor_filter.particles[particle_i]
                        prob_per_particle = <double> neighbor_filter.weights[particle_i]

                        for nested_model_i in range(models_n):
                            nested_neighbor_i = nested_model_i + 1

                            neighbor_internal_state = nested_particle.partial_internal_states.values[nested_neighbor_i]
                            neighbor_probs = <list> all_neighbor_probs[nested_model_i]

                            prob = <double> neighbor_probs[neighbor_internal_state]
                            neighbor_probs[neighbor_internal_state] = prob + prob_per_particle

                    # update the beliefs using the messages
                    for nested_model_i in range(models_n):
                        nested_neighbor_i = nested_model_i + 1

                        neighbor_probs = <list> all_neighbor_probs[nested_model_i]
                        neighbor_message = their_messages.messages[nested_model_i]

                        messages_n = neighbors_possible_messages_per_agent[nested_neighbor_i]

                        neighbor_probs = self.message_estimator.update_internal_state_prob(neighbor_message, messages_n,
                                                                                           neighbor_probs)
                        all_neighbor_probs[nested_model_i] = neighbor_probs

                    # re-weight the existing particle filter using Bayes rule
                    # TODO: this doesn't account for neighbor's beliefs over states...
                    bayes_filter = ParticleFilter()
                    for particle_i in range(particles_n):
                        nested_particle = <NestedParticle> neighbor_filter.particles[particle_i]
                        new_weight = 1.0

                        for nested_model_i in range(models_n):
                            nested_neighbor_i = nested_model_i + 1

                            neighbor_internal_state = nested_particle.partial_internal_states.values[nested_neighbor_i]
                            neighbor_probs = <list> all_neighbor_probs[nested_model_i]

                            prob = <double> neighbor_probs[neighbor_internal_state]
                            new_weight = new_weight * prob

                        bayes_filter.particles.append(nested_particle)
                        bayes_filter.weights.append(new_weight)

                    bayes_filter.normalize()

                    # randomly generate next internal states
                    neighbor_next_filter = ParticleFilter()

                    neighbor_joint_action.actions[...] = modeled_joint_action.actions
                    neighbor_joint_action.actions[0] = modeled_joint_action.actions[neighbor_i]
                    neighbor_joint_action.actions[neighbor_i] = modeled_joint_action.actions[0]
                    neighbor_joint_action.agent_nums[...] = modeled_joint_action.agent_nums
                    neighbor_joint_action.agent_nums[0] = modeled_joint_action.agent_nums[neighbor_i]
                    neighbor_joint_action.agent_nums[neighbor_i] = modeled_joint_action.agent_nums[0]

                    for particle_i in range(particles_n):
                        nested_particle = <NestedParticle> bayes_filter.particles[particle_i]
                        prob_per_particle = <double> bayes_filter.weights[particle_i]

                        sampled_next_internal_states = self.domain.sample_next_partial_internal_states(nested_particle.partial_internal_states,
                                                                                                      neighbor_joint_action)
                        sampled_next_internal_states.values[0] = next_internal_states.values[neighbor_i]

                        # TODO: this doesn't account for neighbor's beliefs over states...
                        nested_particle = NestedParticle(next_state, sampled_next_internal_states, neighbor_level)
                        neighbor_next_filter.particles.append(nested_particle)
                        neighbor_next_filter.weights.append(prob_per_particle)

                    neighbor_next_filter.normalize()
                    neighbor_next_filter.resample_particlefilter(desired_particles_n)

                else:
                    print("No beliefs for", next_internal_state, "for agent", next_internal_states.agent_nums[model_i + 1], flush=True)

                    # TODO should we exit?
                    neighbor_next_filter = ParticleFilter()
            else:
                print("neighbor:", next_internal_states.agent_nums[model_i + 1],
                      "internal state:", next_internal_state,
                      "planning:", len(all_neighbor_next_particle_sets))

            next_particle.nested_particles.append(neighbor_next_filter)

        return next_particle


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
    cdef double rollout(self, NestedParticle particle, int h):
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
            random_joint_action = self.sample_random_joint_action(state, internal_states)
            configuration = self.sample_configuration(random_joint_action)
            action = self.agent.sample_action(state, internal_states.values[0])

            # self.agent is the first spot in random_joint_action since its partial
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
    cdef list create_neighbor_planners(self):
        cdef list neighbor_planners = list()
        cdef Agent model = self.agent.replicate_as_model()
        cdef Agent neighbor

        cdef int models_n = len(self.agent_models)
        cdef int model_i, neighbor_i
        cdef list neighbor_models

        cdef IPOMCPPFCommPlanner ipomcppf_planner
        cdef POMCPPFCommPlanner pomcppf_planner

        cdef int[:] neighbors_possible_messages_per_agent

        for model_i in range(models_n):
            neighbor = <Agent> self.agent_models[model_i]
            neighbor_i = model_i + 1

            # deleting the model of neighbor and creating the IPOMCPPF model for the agent
            neighbor_models = self.agent_models[:] # make a deep copy so we can edit it below
            neighbor_models[model_i] = model

            neighbors_possible_messages_per_agent = cython.view.array(shape=(models_n + 1,), itemsize=sizeof(int),
                                                                      format="i")
            neighbors_possible_messages_per_agent[...] = self.possible_messages_per_agent
            neighbors_possible_messages_per_agent[0] = self.possible_messages_per_agent[neighbor_i]
            neighbors_possible_messages_per_agent[neighbor_i] = self.possible_messages_per_agent[0]

            if self.level > 1:
                ipomcppf_planner = IPOMCPPFCommPlanner(neighbor, self.level - 1, self.horizon, self.gamma,
                                                       self.ucb_c, neighbor_models, self.trajectories,
                                                       self.fully_observable_environment,
                                                       neighbors_possible_messages_per_agent,
                                                       self.message_estimator,
                                                       self.possible_next_states,
                                                       self.global_pomcppfcomm_policy)
                neighbor_planners.append(ipomcppf_planner)
            else:
                pomcppf_planner = POMCPPFCommPlanner(neighbor, self.horizon, self.gamma, self.ucb_c, neighbor_models,
                                                     self.trajectories, self.fully_observable_environment,
                                                     neighbors_possible_messages_per_agent,
                                                     self.message_estimator,
                                                     self.possible_next_states)
                neighbor_planners.append(pomcppf_planner)

        return neighbor_planners


    @cython.boundscheck(False)
    @cython.nonecheck(False)
    @cython.wraparound(False)
    @cython.profile(True)
    cdef PartialJointActionComm sample_modeled_joint_action(self, NestedParticle particle, int h, int[:] messages):
        cdef int models_n = len(self.agent_models)

        # clear the particle sets found for our neighbors in this step
        self.neighbor_next_full_particle_sets.clear()

        # creating a partial joint action
        cdef PartialJointActionComm modeled_joint_action = PartialJointActionComm(models_n + 1)
        modeled_joint_action.agent_nums[0] = self.agent.agent_num

        cdef Action act
        cdef ActionComm neighbor_action
        cdef int model_i, neighbor_i, best_actions_n, rand_i

        # plan actions for our neighbors using lower level reasoning
        cdef PartialInternalStates internal_states, ipomcp_internal_states
        cdef IPOMCPPFCommPlanner ipomcppf_planner
        cdef POMCPPFCommPlanner pomcppf_planner
        cdef ActionNode action_node
        cdef ActionNode_pf action_node_pf
        cdef int internal_state, frame_index, neighbor_horizon, state_index
        cdef tuple msgs_tuple
        cdef list next_particle_sets
        cdef int[:,:,:,:,:] policy
        cdef int[:] neighbor_messages = cython.view.array(shape=(models_n + 1,), itemsize=sizeof(int), format="i")

        for model_i in range(models_n):
            neighbor = <Agent> self.agent_models[model_i]
            neighbor_i = model_i + 1 # self.agent is always first
            neighbor_horizon = self.horizon - h

            neighbor_messages[...] = messages
            neighbor_messages[0] = messages[neighbor_i]
            neighbor_messages[neighbor_i] = messages[0]

            if self.level > 1:
                ipomcppf_planner = <IPOMCPPFCommPlanner> self.neighbor_planners[model_i]
                ipomcppf_planner.horizon = neighbor_horizon
                ipomcppf_planner.tree.particle_filter = <ParticleFilter> particle.nested_particles[model_i]
                ipomcppf_planner.tree.particle_filter.resample_particlefilter(<int> self.agent.settings["particles_n"]//2)

                # get the neighbor's action
                neighbor_action = ipomcppf_planner.create_plan(neighbor_messages)

                # get the neighbor's new belief particle set
                action_node = ipomcppf_planner.last_action_node
                next_particle_sets = action_node.all_beliefs()
                self.neighbor_next_full_particle_sets.append(next_particle_sets)
                self.neighbor_particle_filters[neighbor.agent_num] = ipomcppf_planner.tree.particle_filter
                ipomcppf_planner.reset()
            else:
                if self.use_premodel:
                    msgs_tuple = tuple(neighbor_messages[1:]) # leave out the agent's own message in position 0
                    state_index = particle.state.index
                    internal_state = particle.partial_internal_states.values[neighbor_i]
                    frame_index = neighbor.frame.index

                    # Using a global offline policy
                    policy = self.global_pomcppfcomm_policy[msgs_tuple]
                    best_actions_n =  policy[frame_index][state_index][internal_state][neighbor_horizon - 1][0]

                    if best_actions_n == 1:
                        global_policy_action_index = policy[frame_index][state_index][internal_state][neighbor_horizon - 1][1]
                    else:
                        rand_i = rand() % best_actions_n
                        global_policy_action_index = policy[frame_index][state_index][internal_state][neighbor_horizon - 1][rand_i + 1]

                    #global_policy_action_index = policy[frame_index][state_index][internal_state][neighbor_horizon-1]

                    message = self.message_estimator.estimate_message(internal_state,
                                                                      self.possible_messages_per_agent[neighbor_i],
                                                                      neighbor)
                    neighbor_action = ActionComm(global_policy_action_index, message)
                else:
                    pomcppf_planner = <POMCPPFCommPlanner> self.neighbor_planners[model_i]
                    pomcppf_planner.horizon = self.horizon - h
                    pomcppf_planner.tree.particle_filter = <ParticleFilter> particle.nested_particles[model_i]
                    pomcppf_planner.tree.particle_filter.resample_particlefilter(<int> self.agent.settings["particles_n"]//2)

                    # create the level one internal states particle to use to simulate communication
                    internal_states = particle.partial_internal_states
                    ipomcp_internal_states = PartialInternalStates(models_n + 1)

                    ipomcp_internal_states.values[...] = internal_states.values
                    ipomcp_internal_states.values[0] = internal_states.values[neighbor_i]
                    ipomcp_internal_states.values[neighbor_i] = internal_states.values[0]

                    ipomcp_internal_states.agent_nums[...] = internal_states.agent_nums
                    ipomcp_internal_states.agent_nums[0] = neighbor.agent_num
                    ipomcp_internal_states.agent_nums[neighbor_i] = internal_states.agent_nums[0]

                    pomcppf_planner.message_estimator.start_internal_state_stack(ipomcp_internal_states)

                    # get the neighbor's action
                    neighbor_action = pomcppf_planner.create_plan(neighbor_messages)

                    # get the neighbor's new belief particle set
                    action_node_pf = pomcppf_planner.last_action_node
                    next_particle_sets = action_node_pf.all_beliefs()
                    self.neighbor_next_full_particle_sets.append(next_particle_sets)
                    self.neighbor_particle_filters[neighbor.agent_num] = pomcppf_planner.tree.particle_filter
                    pomcppf_planner.reset()

            modeled_joint_action.actions[neighbor_i] = neighbor_action.index
            modeled_joint_action.agent_nums[neighbor_i] = neighbor.agent_num
            modeled_joint_action.messages[neighbor_i] = neighbor_action.message

        return modeled_joint_action


    @cython.boundscheck(False)
    @cython.nonecheck(False)
    @cython.wraparound(False)
    @cython.profile(True)
    cdef PartialJointAction sample_random_joint_action(self, State state, PartialInternalStates internal_states):
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

            internal_state = internal_states.values[neighbor_i]
            available_actions = neighbor.frame.get_available_actions(internal_state)
            available_actions_n = len(available_actions)

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
        cdef BeliefNode next_bn = self.last_action_node.get_belief(next_internal_state, observation.index,
                                                                   message_index)

        # did we already calculate a belief for this node?
        if len(next_bn.particle_filter.particles) == 0:
            next_bn.particle_filter = self.belief_update(next_state, next_internal_state, self.tree.particle_filter,
                                                         self.last_action_node.action, observation)

        # update the tree
        self.tree = next_bn
        self.tree.parent_action_node = None # for garbage collection
        self.last_action_node = None


    @cython.boundscheck(False)
    @cython.nonecheck(False)
    @cython.wraparound(False)
    @cython.profile(True)
    @cython.cdivision(True)
    cdef ParticleFilter belief_update(self, State next_state, int next_internal_state, ParticleFilter particle_filter,
                                      ActionComm action_comm, ObservationComm obs_comm):
        """ Implements the nested belief update for the given observation comm

        Args:
            next_state:
                The state at next time step
            particle_filter:
                The currnent belief of the subject agent
            action_comm:
                Subject agent's action and message pair
            obs_comm:
                The observation that the subject sees in the domain.
                It also has a set of messages that it receives from
                its neighbors

        Returns:
            new_particle_filter:
                The updated belief of the subject agent
        """
        cdef tuple tup
        cdef State state
        cdef Agent neighbor
        cdef Observation observation
        cdef int model_i, models_n = len(self.agent_models)
        cdef int action_i, actions_n, action_ind
        cdef int message_i, messages_n
        cdef int particle_i, particle_o, particles_n = len(particle_filter.particles)
        cdef int horizon = 0
        cdef list joint_actions_particle = list()
        cdef list all_observations = self.domain.observations
        cdef int obs_i, obs_n = len(all_observations)

        cdef ParticleFilter new_neighbor_particle_filter, new_particle_filter = ParticleFilter()
        cdef ParticleFilter neighbor_particle_filter, neighbor_obs_particle_filter

        cdef ActionComm neighbor_action_comm
        cdef Action neighbor_action
        cdef dict neighbor_probs_act_msg = dict()
        cdef int neighbor_i, neighbor_message, neighbor_internal_state
        cdef ObservationComm neighbor_obs_comm
        cdef IPOMCPPFCommPlanner neighbor_planner

        cdef double weight, new_weight, transition_prob, observation_prob
        cdef double sum_action_message_prob, neighbor_obs_prob

        cdef NestedParticle particle, new_particle, neighbor_particle
        cdef PartialJointActionComm modeled_joint_action
        cdef PartialInternalStates internal_states, next_internal_states, neighbor_internal_states
        cdef Configuration configuration

        cdef int[:] messages = cython.view.array(shape=(models_n + 1,), itemsize=sizeof(int), format="i")
        messages[0] = action_comm.message
        cdef int[:] neighbor_messages, neighbor_actions

        # Create a placeholder with zero value for each possible action message pair of neighbors
        for model_i in range(models_n):
            neighbor = <Agent> self.agent_models[model_i]
            neighbor_i = model_i+1

            neighbor_actions = neighbor.frame.get_all_available_actions()
            actions_n = len(neighbor_actions)

            messages_n  = self.possible_messages_per_agent[neighbor_i]
            messages[neighbor_i] = obs_comm.messages[model_i]

            for action_i in range(actions_n):
                action_ind = neighbor_actions[action_i]

                for message_i in range(messages_n):
                    tup = tuple((neighbor.agent_num, action_ind, message_i-1))
                    neighbor_probs_act_msg[tup] = 0.0000001 # small value to eliminate zero division error

        # Update the neighbor_probabilities of possible action message pair that are
        # obtained from planning for neighbors using subject agent's current belief
        for particle_i in range(particles_n):
            particle = <NestedParticle> particle_filter.particles[particle_i]
            weight = <double> particle_filter.weights[particle_i]

            modeled_joint_action = self.sample_modeled_joint_action(particle, horizon, messages)
            modeled_joint_action.actions[0] = action_comm.index
            modeled_joint_action.agent_nums[0] = self.agent.agent_num
            modeled_joint_action.messages[0] = action_comm.message
            joint_actions_particle.append(modeled_joint_action)

            for model_i in range(models_n):
                neighbor = <Agent> self.agent_models[model_i]
                neighbor_i = model_i+1
                tup = tuple((neighbor.agent_num, modeled_joint_action.actions[neighbor_i],
                             modeled_joint_action.messages[neighbor_i]))
                neighbor_probs_act_msg[tup] = (<double> neighbor_probs_act_msg[tup]) + weight

        # Update the belief of subject agent
        for particle_i in range(particles_n):
            particle = <NestedParticle> particle_filter.particles[particle_i]
            weight = <double> particle_filter.weights[particle_i]

            state = particle.state
            internal_states = <PartialInternalStates> particle.partial_internal_states
            modeled_joint_action = <PartialJointActionComm> joint_actions_particle[particle_i]

            configuration = self.sample_configuration(modeled_joint_action)
            next_internal_states = self.domain.sample_next_partial_internal_states(internal_states, modeled_joint_action)
            next_internal_states.values[0] = next_internal_state

            new_particle = NestedParticle(next_state, next_internal_states, particle.level)
            transition_prob = self.domain.state_transition_probability_configuration(self.agent.agent_num, state,
                                                                                     configuration, action_comm,
                                                                                     next_state)
            observation = <Observation> all_observations[obs_comm.index]
            observation_prob = self.domain.observation_probability_configuration(self.agent.agent_num, state,
                                                                                 configuration, action_comm, next_state,
                                                                                 observation)
            new_weight = transition_prob * observation_prob

            # in case this particle wasn't consistent with the observation and transition
            if new_weight == 0.0:
                continue

            # get the Prod_{all_neighbors}(Sum_{all_actions}(neighbor, action, given message))
            for model_i in range(models_n):
                neighbor_i = model_i + 1
                neighbor_message = messages[neighbor_i]

                neighbor_actions = neighbor.frame.get_all_available_actions()
                actions_n = len(neighbor_actions)

                sum_action_message_prob = 0.0000001 # small value to eliminate zero division error
                for action_i in range(actions_n):
                    action_ind = neighbor_actions[action_i]
                    tup = tuple((neighbor.agent_num, action_ind, neighbor_message))
                    sum_action_message_prob = sum_action_message_prob + (<double> neighbor_probs_act_msg[tup])
                new_weight = new_weight * sum_action_message_prob

            if particle.level == 1:
                for model_i in range(models_n):
                    neighbor = <Agent> self.agent_models[model_i]
                    neighbor_i = model_i + 1

                    neighbor_internal_states = PartialInternalStates(1)
                    neighbor_internal_states.agent_nums[0] = neighbor.agent_num
                    neighbor_internal_states.values[0] = next_internal_states.values[neighbor_i]

                    neighbor_particle = NestedParticle(next_state, neighbor_internal_states, 0)

                    neighbor_particle_filter = ParticleFilter()
                    neighbor_particle_filter.particles.append(neighbor_particle)
                    neighbor_particle_filter.weights.append(1.0)

                    new_particle.nested_particles.append(neighbor_particle_filter)
            else:
                for model_i in range(models_n):
                    neighbor = <Agent> self.agent_models[model_i]
                    neighbor_i = model_i + 1

                    neighbor_action_comm = ActionComm(modeled_joint_action.actions[neighbor_i],
                                                      modeled_joint_action.messages[neighbor_i])

                    neighbor_messages = cython.view.array(shape=(models_n + 1,), itemsize=sizeof(int), format="i")
                    neighbor_messages[...] = messages
                    neighbor_messages[0] = messages[neighbor_i]
                    neighbor_messages[neighbor_i] = action_comm.message

                    neighbor_planner = self.neighbor_planners[model_i]
                    neighbor_particle_filter = <ParticleFilter> particle.nested_particles[model_i]

                    # Getting a new particle filter for each observation
                    for obs_i in range(obs_n):
                        observation = <Observation> all_observations[obs_i]
                        neighbor_obs_prob = self.domain.observation_probability_configuration(neighbor.agent_num,
                                                                                              state,
                                                                                              configuration,
                                                                                              neighbor_action_comm,
                                                                                              next_state,
                                                                                              observation)

                        if neighbor_obs_prob > 0.0:
                            # recursively find the next belief for this observation
                            neighbor_obs_comm = ObservationComm(observation.index, models_n)
                            neighbor_obs_comm.messages[...] = neighbor_messages[1:]

                            neighbor_obs_particle_filter = neighbor_planner.belief_update(next_state,
                                                                                          next_internal_states.values[neighbor_i],
                                                                                          neighbor_particle_filter,
                                                                                          neighbor_action_comm,
                                                                                          neighbor_obs_comm)

                            # Multiplying the weights in observation particlefilter with observation probability
                            for particle_o in range(len(neighbor_obs_particle_filter.particles)):
                                neighbor_obs_particle_filter.weights[particle_o] = (<double> neighbor_obs_particle_filter.weights[particle_o]) * neighbor_obs_prob

                            neighbor_particle_filter.particles.extend(neighbor_obs_particle_filter.particles)
                            neighbor_particle_filter.weights.extend(neighbor_obs_particle_filter.weights)

                    neighbor_particle_filter.normalize()
                    neighbor_particle_filter.resample_particlefilter(particles_n)

                    new_particle.nested_particles.append(neighbor_particle_filter)

            new_particle_filter.particles.append(new_particle)
            new_particle_filter.weights.append(new_weight)

        new_particle_filter.normalize()
        new_particle_filter.resample_particlefilter()

        return new_particle_filter

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
        cdef int[:] available_actions_comm
        if state is not None:
            available_actions_comm = agent.frame.get_available_actions_comm_per_state(state, internal_state)
        else:
            available_actions_comm = agent.frame.get_all_available_actions_comm()

        cdef int actions_n = len(available_actions_comm)
        cdef int actions_i
        cdef ActionNode action_node
        cdef ActionComm action

        for action_i in range(actions_n):
            action = <ActionComm> agent.domain.actions_with_comm[available_actions_comm[action_i]]
            action_node = ActionNode(action, action_i, self)
            self.action_nodes.append(action_node)


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
        # tup = tuple((agent_num, action_index))
        # last_action_all_belief_node_visits[tup] = <double> action_node.visits/self.visits
        for belief_node_i in range(len(belief_nodes)):
            bn = belief_nodes[belief_node_i]
            tup = tuple((agent_num, action_index, bn.observation_index, bn.message_index))
            if tup in last_action_all_belief_node_visits:
                last_action_all_belief_node_visits[tup] += <double> (bn.visits/ action_node.visits)
            else:
                last_action_all_belief_node_visits[tup] = <double> (bn.visits/ action_node.visits)
        return last_action_all_belief_node_visits


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
            available = False
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
                print("Action:", action_node.action.index,"Message:", action_node.action.message, "q:", action_node.q_value,
                      "visits:", action_node.visits, "ucb1:", ucb1)
                log_info(f"Action:{ action_node.action.index}, q:{action_node.q_value},visits:{action_node.visits}, ucb1:{ucb1}", False)


        # randomly break ties, if needed
        cdef int rand_i
        if best_action_nodes_n == 1:
            return best_action_nodes[0]
        else:
            rand_i = rand() % best_action_nodes_n
            return best_action_nodes[rand_i]


cdef class ActionNode:
    def __cinit__(self, ActionComm action, int available_action_i, BeliefNode parent_belief_node):
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
