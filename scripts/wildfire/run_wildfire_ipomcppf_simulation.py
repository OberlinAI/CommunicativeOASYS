import sys; sys.path.append("../..")
import time
import numpy as np
import pickle

from oasys.domains.wildfire.wildfire import Wildfire
from oasys.domains.wildfire.wildfire_simulation import WildfireSimulation
from oasys.domains.wildfire.wildfire_reasoning import WildfireReasoningType

REASONING_TYPE = WildfireReasoningType.IPOMCPPF
STEPS = 15

START_RUN = 1
STOP_RUN = 100


def main():
    # parse the command line arguments
    setup = int(sys.argv[1])
    start_run = int(sys.argv[2])
    stop_run = int(sys.argv[3])

    settings = {}
    settings["epsilon_p"] = float(sys.argv[4])
    settings["trajectories"] = int(sys.argv[5])
    settings["horizon"] = int(sys.argv[6])
    settings["ucb_c"] = float(sys.argv[7])
    settings["level"] = int(sys.argv[8])
    settings["particles_n"] = int(sys.argv[9])
    settings["cliques_n"] = int(sys.argv[10])
    settings["premodel"] = bool(int(sys.argv[11]))

    # Loading global policy into the agent settings
    if settings["premodel"]:
        # Policy setup numbers are either 1, 2, or 3.
        # They correspond to actual setup numbers 1XX, 2XX, and 3XX, respectively
        setup_number = setup // 100
        settings["POMCPPF_global_policy"] = np.load(f'POMCPPF_policy_Setup{setup_number}.npy')
    else:
        settings["POMCPPF_global_policy"] = np.zeros((1,1,1,1), dtype = np.intc)

    for run in range(start_run, stop_run + 1):
        # Adding run number to the settings dictionary
        settings["run"] = run

        # Logging the run times
        start_time = time.time()

        run_simulations(run, setup, settings)

        run_time = time.time() - start_time
        print("Run Time: ", run_time)
        with open(f'ipomcppf_level{settings["level"]}_Runtimes.txt', 'a') as file:
            file.write((str(run_time)[:8])+'\n')



def run_simulations(run, setup, settings):
    # run the simulation
    simulator = WildfireSimulation(run, setup, REASONING_TYPE, settings)
    simulator.run(STEPS)


if __name__ == "__main__":
    main()
