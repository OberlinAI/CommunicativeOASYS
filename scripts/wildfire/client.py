import sys; sys.path.append("../..")
from oasys.domains.wildfire.wildfire import Wildfire
from oasys.domains.wildfire.wildfire_simulation import WildfireNetworkSimulation
from oasys.domains.wildfire.wildfire_reasoning import WildfireReasoningType
from oasys.domains.wildfire.wildfire_structures import (WildfireState,
                                                        WildfireAction,
                                                        WildfireInternalStates,
                                                        WildfireObservation,
                                                        WildfireObservationComm,
                                                        WildfireJointAction,
                                                        WildfireConfiguration)
import socket
import pickle
import numpy as np
import time

SETUP = 1
REASONING_TYPES = [WildfireReasoningType.Heuristic, WildfireReasoningType.POMCPPF,
                   WildfireReasoningType.IPOMCPPF, WildfireReasoningType.IPOMCPPFComm]
STEPS = 15

START_RUN = 1
STOP_RUN = 100


class NetworkClient:
    """Class for the client to receive data and simulate the domain"""
    def __init__(self):
        self.reasoning = REASONING_TYPES[int(sys.argv[1])]
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.host = str(sys.argv[2])  # server ip address
        self.client_num = int(sys.argv[3])
        self.port = 9990
        self.s.connect((self.host, self.port))
        # Creating a log file and clearing if it is not empty
        print(f"Connection to {self.host} from client {self.client_num} established")
        open(f"client_logger.txt", "w").close()
        # open(f"ipomcpcomm_logger.txt", "w").close()
        # open(f"pomcpcomm_logger.txt", "w").close()

    def send_data(self, data):
        """Sends the data to the server in pickle format
        """
        print('sending data')
        pickl = pickle.dumps(data)
        self.s.send(pickl)

    def recv_data(self):
        """Receives the data to the server in pickle format
        """
        pickl = self.s.recv(8192)
        data = [0]
        if pickl:
            data = pickle.loads(pickl)
        return data

    def log_info(self, data, log_time=True):
        """Logs the data into a file

        Args:
            data:
                The information that needs to be logged into the file
            log_time:(bool)
                Logs the time of the information if true else just logs the
                information
        """
        with open("client_logger.txt", "a") as file1:
            sec = time.time()
            local_time = time.ctime(sec)
            content = f'{data}\n\n'
            if log_time:
                content = f'{local_time}\n' + content
            # Writing data to a file
            file1.write(content)

    def wildfire_task(self):
        """Simulates the action and retuns the action taken by agent
        """
        # Saving Client number and server address
        # self.log_info(f'Client: {self.client_num}\n' +
        #               f'Server Address: {self.host}\n' +
        #             f'Port: {self.port}\n')

        data = [0]
        simulator = WildfireNetworkSimulation(1, 1)
        while data[-1] != 'Terminate Connection':
            data = self.recv_data()

            if data[-1] == 'Initial Data':
                settings = data[0]
                run = data[1]
                setup = data[2]
                total_clients = data[3]
                print('Initial settings and setup number received')
                # Global pomcppf policy
                if 'filename' in settings:
                    # Default policy is already in settings,
                    # load the global policy if the filename is given by server
                    if self.reasoning == WildfireReasoningType.IPOMCPPF:
                        settings["POMCPPF_global_policy"] = np.load(settings["filename"])
                    elif self.reasoning == WildfireReasoningType.IPOMCPPFComm:
                        with open(settings["filename"],"rb") as f:
                            settings["POMCPPFComm_global_policy"] = pickle.load(f)
                # self.log_info(f'settings:{settings}, run:{run}, setup: {setup},client_num: {self.client_num}, total_clients:{total_clients}')
                simulator = WildfireNetworkSimulation(run, setup, self.reasoning, settings)

            elif data[-1] == 'Start Simulation':
                sim_start_time = time.time()
                state = data[0]
                internal_states = data[1]
                print('State and Internal States are received')
                # self.log_info(f'state:{np.array(state.values)}, internal states: {np.array(internal_states.values)}')
                joint_action = simulator.client_run(self.client_num, state,
                                                    internal_states,
                                                    total_clients)
                sim_end_time = time.time()
                print("Step simulation time: ", (sim_end_time - sim_start_time))
                # self.log_info(f'Action:{np.array(joint_action.actions)}, Messages:{np.array(joint_action.messages)}')

                # sending the joint_action
                self.send_data([joint_action, 'Simulation Done'])

            elif data[-1] == 'Reward Obs':
                rewards = data[0]
                observations = data[1]
                next_state = data[2]
                next_internal_states = data[3]

                print('Rewards, Observations, next_state and next_internal_states received')
                # self.log_info(f'rewards:{rewards}, next_state:{np.array(next_state.values)}, '+
                #               f'next_internal_states:{np.array(next_internal_states.values)}')

                simulator.update_reward_obs(rewards, observations, next_state,
                                            next_internal_states, self.client_num, total_clients)
                # self.log_info('--------------------------------------------', False)


        print('Closing the connection')
        self.s.close()

if __name__ == "__main__":
    nc = NetworkClient()
    nc.wildfire_task()
