# name : yuval saadati
# id: 205956634
import sys

from pddlsim.executors.plan_dispatch import PlanDispatcher
from pddlsim.local_simulator import LocalSimulator
from my_agent import Executor
with open(str(sys.argv[2])) as fp:
    line = fp.readline()
    planner = False
    while line:
        if "probabilistic" in line:
            print (LocalSimulator().run(str(sys.argv[2]), str(sys.argv[3]), Executor(str(sys.argv[1]))))
            planner = True
            break
        line = fp.readline()
    if not planner:
        print (LocalSimulator().run(str(sys.argv[2]), str(sys.argv[3]), PlanDispatcher()))