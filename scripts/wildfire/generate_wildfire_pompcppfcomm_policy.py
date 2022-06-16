import sys; sys.path.append("../..")
from multiprocessing import Pool
import time
import numpy as np
import pickle

from oasys.domains.wildfire.wildfire import Wildfire
from oasys.domains.wildfire.wildfire_reasoning import WildfireReasoningType

# PROCESSES = 8

Q_SENSITIVITY = 0.01

def log(data):
    with open("comm_policy_gen.txt",'a') as f:
        f.write(f"{time.ctime()}: {data} \n")


def create_all_messages(neighbor_i, neighbors_n, suppressant_states, messages, all_messages):
    if neighbor_i == neighbors_n:
        all_messages.append(messages)
    else:
        for message in range(-1, suppressant_states):
            new_messages = list(messages)
            new_messages.append(message)

            create_all_messages(neighbor_i + 1, neighbors_n, suppressant_states, new_messages, all_messages)


def main():
    # parse the command line arguments and create the settings
    settings = {}

    setup = int(sys.argv[1])
    settings["epsilon_p"] = float(sys.argv[2])
    settings["trajectories"] = int(sys.argv[3])
    max_horizon = int(sys.argv[4])
    settings["horizon"] = max_horizon
    settings["ucb_c"] = float(sys.argv[5])
    settings["particles_n"] = int(sys.argv[6])
    settings["cliques_n"] = int(sys.argv[7])
    settings["level"] = 0
    processes = int(sys.argv[8])

    open("comm_policy_gen.txt", "w")

    # create the domain and grab useful properties
    wildfire = Wildfire(setup, WildfireReasoningType.POMCPPFComm, settings)
    states_n = len(wildfire.states)
    actions_n = len(wildfire.actions)
    suppressant_states = len(wildfire.settings.INITIAL_SUPPRESSANT_DISTRIBUTION)
    agents_per_frame = np.asarray(wildfire.settings.N_AGENTS_PER_FRAME)
    frames_n = len(agents_per_frame)
    neighbors_n = len(wildfire.modeling_neighborhoods[0]) - 1 # assuming this is the same for all agents

    # create all of the possible message sets
    all_messages = list()
    create_all_messages(0, neighbors_n, suppressant_states, [-1], all_messages)

    # split the work over multiple processes
    messages_n = len(all_messages)
    print("messages_n:", messages_n)

    params = []
    arr_shape = (frames_n, states_n, suppressant_states, max_horizon, actions_n + 1)
    agent_num = 0
    process_counter = 0
    for frame_i in range(frames_n):
        start_index = 0
        processes_per_frame = processes // frames_n
        for process in range(processes_per_frame):
            if process == processes_per_frame - 1:
                end_index = messages_n
            else:
                end_index = int((process +  1) * messages_n / processes_per_frame)

            process_messages = list()
            for messages_i in range(start_index, end_index):
                msg_vector = np.asarray(all_messages[messages_i], dtype=np.intc)
                policy_arr = -1 * np.ones(shape=arr_shape, dtype=np.intc)
                process_messages.append((msg_vector, messages_i, policy_arr))

            params.append((setup, frame_i, states_n, agents_per_frame, max_horizon, process_messages, settings, process_counter, agent_num))

            start_index = end_index
            process_counter += 1

        agents_n_frame = agents_per_frame[frame_i]
        agent_num = agent_num + agents_n_frame

    with Pool(processes) as pool:
        workers_data = pool.map(run, params)

    results = [None] * len(workers_data)
    for tup in workers_data:
        results[tup[0]] = tup[1]

    write_policies(results, frames_n, setup, settings["trajectories"], settings["ucb_c"])


def write_policies(results, frames_n, setup, trajectories, ucb_c):
    policies = {}
    index = 0
    processes = len(results)
    for frame_i in range(frames_n):
        for process in range(processes // frames_n):
            dictionary = results[index]
            print(index, list(dictionary.keys()))
            for key in dictionary:
                if key not in policies:
                    policies[key] = dictionary[key]
                else:
                    existing_table = policies[key]
                    update_table = dictionary[key]

                    print(existing_table.shape, update_table.shape)

                    for state_i in range(existing_table.shape[1]):
                        for internal_state in range(existing_table.shape[2]):
                            for horizon in range(existing_table.shape[3]):
                                for action_i in range(existing_table.shape[4]):
                                    existing_table[frame_i][state_i][internal_state][horizon][action_i] = update_table[frame_i][state_i][internal_state][horizon][action_i]
            index += 1

    setup_number = setup // 100
    filename = f'POMCPPFComm_policy_Setup{setup_number}_t{trajectories}_u{ucb_c}.pkl'
    with open(filename, 'wb') as f:
        pickle.dump(policies, f)


def run(params):
    # grab the info from the params
    setup = params[0]
    frame_i = params[1]
    states_n = params[2]
    agents_per_frame = params[3]
    max_horizon = params[4]
    messages_list = params[5]
    settings = params[6]
    process = params[7]
    agent_num = params[8]

    dictionary = dict()

    for tup in messages_list:
        message_vector = tup[0]
        message_i = tup[1]
        policy_arr = tup[2]

        start_time = time.time()
        log(f"{message_i}: {message_vector}")

        # agent_num = 0
        # for frame_i in range(frames_n):
        for horizon in range(1, max_horizon+1):
            settings["horizon"] = horizon
            wildfire = Wildfire(setup, WildfireReasoningType.POMCPPFComm, settings)

            agent = wildfire.agents[agent_num]
            suppressant_states = len(wildfire.settings.INITIAL_SUPPRESSANT_DISTRIBUTION)

            for state_i in range(states_n):
                state = wildfire.states[state_i]

                for internal_state in range(suppressant_states):
                    agent.set_current_state(state)
                    agent.set_current_suppressant(internal_state)

                    q_values = agent.calculate_q_values(message_vector)

                    best_actions = []
                    best_q = -99999999999999
                    actions_n = len(q_values)
                    for action_i in range(actions_n):
                        q = q_values[action_i]

                        if q > best_q:
                            best_q = q

                    best_lower = best_q - Q_SENSITIVITY
                    for action_i in range(actions_n):
                        q = q_values[action_i]

                        if q >= best_lower:
                            best_actions.append(action_i)

                    best_actions_n = len(best_actions)
                    policy_arr[frame_i][state_i][internal_state][horizon - 1][0] = best_actions_n

                    for best_action_i in range(best_actions_n):
                        policy_arr[frame_i][state_i][internal_state][horizon - 1][best_action_i + 1] = best_actions[best_action_i]

                    # act = agent.choose_action(message_vector)
                    # act_index = act.index
                    # policy_arr[frame_i][state_i][internal_state][horizon-1] = act_index
                    print(f"messages:{message_vector}, F:{frame_i}, " +
                          f"H:{horizon}, S:{state_i}, IST:{internal_state}, " +
                          f"A:{best_actions}")

            print("Agent:", agent_num, "Horizon:", horizon, "State:", state_i, "Messages:", [np.asarray(m) for m in messages_list])

            # agents_n_frame = agents_per_frame[frame_i]
            # agent_num = agent_num + agents_n_frame

        # save the policy under the message
        dictionary[tuple(message_vector[1:])] = policy_arr  # drop the agent's own message since it won't be part of the observation

        end_time = time.time()
        time_elapsed = end_time-start_time
        data = f"{message_i}: {message_vector}, {round(time_elapsed,3)} seconds"
        print(data)
        log(data)

    # Create the separate policy for the internal_state
    setup_number = setup // 100
    filename = f'separate_policies/POMCPPFComm_policy_Setup{setup_number}_t{settings["trajectories"]}_u{settings["ucb_c"]}-{process}.pkl'
    with open(filename, "wb")as f1:
        pickle.dump(dictionary, f1)

    # return dictionary
    return (process, dictionary)


def test():
    settings = {}

    setup = int(sys.argv[1])
    settings["epsilon_p"] = float(sys.argv[2])
    settings["trajectories"] = int(sys.argv[3])
    max_horizon = int(sys.argv[4])
    settings["horizon"] = max_horizon
    settings["ucb_c"] = float(sys.argv[5])
    settings["particles_n"] = int(sys.argv[6])
    settings["cliques_n"] = int(sys.argv[7])
    settings["level"] = 0
    processes = int(sys.argv[8])
    frames_n = int(sys.argv[9])

    results = list()
    setup_number = setup // 100
    for process in range(processes):
        filename = f'separate_policies/POMCPPFComm_policy_Setup{setup_number}_t{settings["trajectories"]}_u{settings["ucb_c"]}-{process}.pkl'
        with open(filename, "rb") as f:
            dictionary = pickle.load(f)
            results.append(dictionary)

    write_policies(results, frames_n, setup, settings["trajectories"], settings["ucb_c"])


if __name__ == "__main__":
    main()
    # test()
