import sys; sys.path.append("../..")

from oasys.domains.wildfire.wildfire import Wildfire
from oasys.domains.wildfire.wildfire_simulation import WildfireSimulation
from oasys.domains.wildfire.wildfire_reasoning import WildfireReasoningType


SETUPS = [100]
START_RUN = 1
STOP_RUN = 50
STEPS = 15


def main():
    for reasoning_type in [WildfireReasoningType.Coordination, WildfireReasoningType.Heuristic, WildfireReasoningType.NOOP]:
        for setup in [SETUPS]:
            for run in range(START_RUN, STOP_RUN + 1):
                settings = {}
                settings["trajectories"] = 0

                simulator = WildfireSimulation(run, setup, reasoning_type, settings)
                simulator.run(STEPS)


if __name__ == "__main__":
    main()
