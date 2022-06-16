import sys; sys.path.append("../..")

from oasys.domains.wildfire.wildfire import Wildfire
import oasys.domains.wildfire.wildfire_structures as wildfire_structures
from oasys.domains.wildfire.wildfire_settings import WildfireSettings


def main():
    test2()
    testAll()


def test2():
    settings = WildfireSettings(100)
    state1 = wildfire_structures.WildfireState(settings.FIRES)
    state1.values[0] = 0
    state1.values[1] = 0
    state1.values[2] = 0
    state1.calculate_index(settings)

    state2 = wildfire_structures.WildfireState(settings.FIRES)
    state2.values[0] = 1
    state2.values[1] = 2
    state2.values[2] = 3
    state2.calculate_index(settings)

    print(state1.index, state2.index)


def testAll():
    settings = WildfireSettings(100)
    n = settings.FIRE_STATES ** settings.FIRES
    m = settings.FIRE_STATES

    states = []
    for i in range(n):
        s = i

        state = wildfire_structures.WildfireState(settings.FIRES)
        states.append(state)
        for f in range(settings.FIRES):
            state.values[f] = s % m
            s /= m
        state.calculate_index(settings)

    allOK = True
    for i in range(n):
        state = states[i]

        if i != state.index:
            print(state.index, [v for v in state.values])
            allOK = False

    if allOK:
        print("All", n, "states correct!")




if __name__ == "__main__":
    main()
