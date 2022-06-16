import sys; sys.path.append("../..")
from multiprocessing import Pool
import time
import numpy as np

from oasys.domains.wildfire.wildfire import Wildfire
from oasys.domains.wildfire.wildfire_reasoning import WildfireReasoningType

Q_SENSITIVITY = 0.01

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
    settings["level"] = 0
    settings["cliques_n"] = int(sys.argv[7])
    processes = int(sys.argv[8])

    # create the domain and grab useful properties
    wildfire = Wildfire(setup, WildfireReasoningType.POMCPPF, settings)
    states_n = len(wildfire.states)
    actions_n = len(wildfire.actions)
    suppressant_states = len(wildfire.settings.INITIAL_SUPPRESSANT_DISTRIBUTION)
    agents_per_frame = np.asarray(wildfire.settings.N_AGENTS_PER_FRAME)
    frames_n = len(agents_per_frame)

    shape = (frames_n, states_n, suppressant_states, max_horizon, actions_n + 1)
    policy_arr = np.empty(shape=shape, dtype=np.intc)

    agent_i = 0
    params = []
    for frame_i in range(frames_n):
        frame_policy = -1 * np.ones(shape=shape[1:], dtype=np.intc)
        params.append((setup, agent_i, frame_i, settings, max_horizon, frame_policy))

        agent_i = agent_i + agents_per_frame[frame_i]

    with Pool(processes) as pool:
        worker_data = pool.map(run, params)

    for worker in worker_data:
        worker_agent_num = worker[0]
        policy_arr[worker_agent_num] = worker[1]

    # saving the global policy in a *npy file
    setup_number = setup // 100
    filename = f'POMCPPF_policy_Setup{setup_number}.npy'
    with open(filename, 'wb') as f:
        np.save(f, policy_arr)


def run(params):
    setup = params[0]
    agent_num = params[1]
    frame_num = params[2]
    settings = params[3]
    max_horizon = params[4]
    agent_policy = params[5]

    for horizon in range(1, max_horizon+1):
        start_time = time.time()
        settings["horizon"] = horizon
        wildfire = Wildfire(setup, WildfireReasoningType.POMCPPF, settings)
        agent = wildfire.agents[agent_num]
        states_n =  len(wildfire.states)
        suppressant_states = len(wildfire.settings.INITIAL_SUPPRESSANT_DISTRIBUTION)

        for state_i in range(states_n):
            state = wildfire.states[state_i]

            for internal_state in range(suppressant_states):
                agent.set_current_state(state)
                agent.set_current_suppressant(internal_state)

                q_values = agent.calculate_q_values()

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
                agent_policy[state_i][internal_state][horizon - 1][0] = best_actions_n

                for best_action_i in range(best_actions_n):
                    agent_policy[state_i][internal_state][horizon - 1][best_action_i + 1] = best_actions[best_action_i]

                print(best_actions, agent_policy[state_i][internal_state][horizon-1])

            print("Agent:", agent_num, "Horizon:", horizon, "State:", state_i)

    return (frame_num, agent_policy)

if __name__ == "__main__":
    main()
