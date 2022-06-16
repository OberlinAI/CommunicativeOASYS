import sys; sys.path.append("../..")
from oasys.domains.wildfire.wildfire import Wildfire
from oasys.domains.wildfire.wildfire_simulation import WildfireNetworkSimulation
from oasys.domains.wildfire.wildfire_reasoning import WildfireReasoningType
from oasys.domains.wildfire.wildfire_structures import (WildfireState,
                                                        WildfireAction,
                                                        WildfireInternalStates,
                                                        WildfireJointAction,
                                                        WildfireJointActionComm,
                                                        WildfireObservation,
                                                        WildfireObservationComm,
                                                        WildfireConfiguration)
from multiprocessing import Queue, Process
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


class NetworkServer:
    """Class for the server that oversees remote clients, stores the joint action,
    records next environment state, intrenal states, rewards. Also broadcasts both
    initial and next internalstates environment states, rewards"""
    def __init__(self):
        """ Args:
                num_clients(int):
                    Number of clients
        """
        self.reasoning = REASONING_TYPES[int(sys.argv[1])]
        self.num_clients = int(sys.argv[2])
        self.connections = []
        self.addresses = []
        self.host = ''
        self.port = 9990
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    def bind_socket(self):
        """ Binds the socket to address
        """
        print("Binding the Port: " + str(self.port))
        # Re-use the address if it is already in use
        self.s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        # Binds the socket
        self.s.bind((self.host, self.port))
        self.s.listen(5)

    def accepting_connections(self):
        """Accepts the connections from all the clients
        """
        while len(self.connections) != self.num_clients:
            print(f'Expecting {self.num_clients-len(self.connections)} client(s) to connect')
            conn, addr = self.s.accept()
            self.connections.append(conn)
            self.addresses.append(addr)
            self.s.setblocking(True)
            print('Connection Succesful to ', addr[0])

    def log_info(self, data, log_time=True):
        """Logs the data into a file

        Args:
            data:
                The information that needs to be logged into the file
            log_time:(bool)
                Logs the time of the information if true else just logs the
                information
        """
        with open("server_logger.txt", "a") as file1:
            sec = time.time()
            local_time = time.ctime(sec)
            content = f'{data}\n\n'
            if log_time:
                content = f'{local_time}\n' + content
            # Writing data to a file
            file1.write(content)

    def parallel_receive(self, num_agents):
        """Receives data from clients in parallel

        Args:
            num_agents(int):
                Total number of agents in the system
        Returns:
            joint_action(Wildfire.JointAction)
                A JointAction object containing the actions taken by each of the agent
        """
        output = Queue()
        processes = [Process(target=self.receive_data_client,
                     args=(conn, output)) for conn in self.connections]
        for p in processes:
            p.start()
        for p in processes:
            p.join(timeout=None)

        joint_action_list = [output.get() for p in processes]
        joint_action = self.combine_joint_action(joint_action_list, num_agents)
        # self.log_info(f'Joint Actions: {list(joint_action.actions)}')
        return joint_action

    def receive_data_client(self, conn, output):
        """Receives data from a particular client

        Args:
            conn(socket):
                A connection socket of between client and server
            output(queue):
                a queue object that is used to store joint action data from clients
        """
        data = [0]
        while(data[-1] != 'Simulation Done'):
            data = pickle.loads(conn.recv(8192))
            if data:
                print(f'Received joint action from client {self.connections.index(conn)}')
                # self.log_info(f'Received joint action from client {self.connections.index(conn)}')
                output.put(data[0])

    def combine_joint_action(self, joint_action_list, num_agents):
        """ Combines all the joint action objects given by the clients

        Args:
            joint_action_list(list):
                A list of joint action objects returned by clients
            num_agents(int):
                Total number of Agents

        Returns:
            joint_action(Wildfire.JointAction)
                A JointAction object containing the actions taken by each of the agent
        """
        joint_action_comm = WildfireJointActionComm(num_agents)
        for joint_action_subset in joint_action_list:
            for num_agent in range(num_agents):
                if joint_action_subset.actions[num_agent] != -99:
                    joint_action_comm.actions[num_agent] = np.intc(joint_action_subset.actions[num_agent])
                # Checking if messages are relevant
                # if messages are irrelevant they are initialised to -1
                if self.reasoning in [WildfireReasoningType.IPOMCPPFComm]:
                    # Consider only messages of those agents that are modelled by clients
                    if joint_action_subset.messages[num_agent] != -99:
                        joint_action_comm.messages[num_agent] = np.intc(joint_action_subset.messages[num_agent])
        return joint_action_comm

    def broadcast_data(self, data):
        """Converts the data into pickle format and then sends the data
        to all the clients
        Args:
            pickl(pickle):
                The data received from server in client format.
        """
        if len(self.connections) == self.num_clients:
            pickl = pickle.dumps(data)
            for conn in self.connections:
                conn.send(pickl)
                time.sleep(0.05)

    def wildfire_task(self):
        """Starts and executes the simulation on the server"""

        self.bind_socket()
        self.accepting_connections()

        # Creating a log file and clearing if it is not empty
        open("server_logger.txt", "w").close()
        # Saving Client number and server address
        # self.log_info(f'Clients: {self.num_clients}\n Port: {self.port}\n' +
        #               f'Clients Addresses: {self.addresses}\n' +
        #               f'Connections: {self.connections}\n')

        setup = int(sys.argv[3])
        start_run = int(sys.argv[4])
        stop_run = int(sys.argv[5])

        settings = {}
        settings["epsilon_p"] = float(sys.argv[6])
        settings["trajectories"] = int(sys.argv[7])
        settings["horizon"] = int(sys.argv[8])
        settings["ucb_c"] = float(sys.argv[9])
        settings["level"] = int(sys.argv[10])
        settings["particles_n"] = int(sys.argv[11])
        settings["cliques_n"] = int(sys.argv[12])
        if self.reasoning in [WildfireReasoningType.IPOMCPPF, WildfireReasoningType.IPOMCPPFComm]:
            settings["premodel"] = bool(int(sys.argv[13]))
        if settings["premodel"]:
            if self.reasoning == WildfireReasoningType.IPOMCPPF:
                setup_number = setup // 100
                settings["filename"] = f'POMCPPF_policy_Setup{setup_number}.npy'
            elif self.reasoning == WildfireReasoningType.IPOMCPPFComm:
                setup_number = setup // 100
                settings["filename"] = f'POMCPPFComm_policy_Setup{setup_number}.pkl'

        # Loading a default policies on the server side
        settings["POMCPPF_global_policy"] = np.zeros((1,1,1,1), dtype = np.intc)
        settings["POMCPPFComm_global_policy"] = dict()

        # self.log_info(f'Settings: {settings}')

        for run in range(start_run, stop_run + 1):
             # Adding run number to the settings dictionary
            settings["run"] = run

            # Logging the run times
            start_time = time.time()

            self.run_simulations(run, setup, settings, self.num_clients)

            run_time = time.time() - start_time
            with open(f'Runtimes.txt', 'a') as file:
                file.write((str(run_time)[:8])+'\n')

        # Close connection
        self.broadcast_data(['Terminate Connection'])
        self.s.close()

    def run_simulations(self, run, setup, settings, num_clients):
        """ Runs the simulation upto the STEPS specified. Calculates the
        rewards, observations state and internal_states and communicates
         them to clients

        Args:
            run (int):
                number denoting the nth run
            setup (int):
                Setup number for the domain. Wildfire Domain:{3,4,5}
            settings (dict):
                Contains all the required input parameters for model
            num_clients (int):
                Number of clients in the network setup
        """

        # self.log_info(f'Sending the data to clients')
        self.broadcast_data([settings, run, setup, self.num_clients, 'Initial Data'])
        simulator = WildfireNetworkSimulation(run, setup, self.reasoning, settings)
        state = simulator.domain.generate_start_state()
        internal_states = simulator.domain.generate_start_internal_states()
        num_agents = len(internal_states.values)
        time.sleep(0.5)
        run_start_time = time.time()

        # Saving the initial state
        simulator.update_log_state(0, state, internal_states)

        # run the simulation
        for step in range(1, STEPS + 1):
            step_start_time = time.time()
            # self.log_info(f"Step: {step}\n Fires:{np.array(state.values)}\n"+
            #               f" Suppressants:{np.array(internal_states.values)}")
            print("Step: ", step, "Fires:", np.array(state.values), "Suppressants:", np.array(internal_states.values))

            # self.log_info('Broadcasting internal states')
            print('Broadcasting state and internal states')
            self.broadcast_data([state, internal_states, 'Start Simulation'])
            print('Waiting for clients to send the joint action')
            # Receiving joint action from clients
            joint_action_comm = self.parallel_receive(num_agents)
            received_joint_action = np.array(joint_action_comm.actions)
            print("Received Joint Action: ", received_joint_action)
            # self.log_info(f'Received Joint action: {received_joint_action}')

            # Logging the messages received
            if self.reasoning in [WildfireReasoningType.IPOMCPPFComm]:
                received_messages = np.array(joint_action_comm.messages)
                print("Received Messages: ", received_messages)
                # self.log_info(f'Received Messages: {received_messages}')


            rewards, observations, next_state, next_internal_states = simulator.server_run(state, internal_states, joint_action_comm)
            self.broadcast_data([rewards, observations, next_state, next_internal_states, 'Reward Obs'])
            # self.log_info(f'broadcasting observations, rewards:{rewards}\n'+
            #               f'next_state:{np.array(next_state.values)}\n'+
            #               f'next_internal_states:{np.array(next_internal_states.values)}')

            # update the state
            state = next_state
            internal_states = next_internal_states

            step_end_time = time.time()

            print("Time elapsed for previous step: ", round(step_end_time - step_start_time), "seconds")
            print("Total time elapsed in current run: ", round(step_end_time - run_start_time), "seconds")
            # log this information
            simulator.update_log_state(step, next_state,
                                       next_internal_states)
            simulator.update_log_actions_observations_rewards_messages(step,
                                                                       joint_action_comm,
                                                                       observations,
                                                                       rewards)
            # self.log_info('-------------------------------',False)


if __name__ == "__main__":
    ns = NetworkServer()
    ns.wildfire_task()
