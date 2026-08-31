from my_agents.dynamic_crop_agent_v0 import agent
from kaggle_environments import make
from pprint import pprint

env = make("kaggriculture",
           configuration={"episodeSteps": 720,
                          "randomSeed": 0,},
                          debug=True,)

steps = env.run([agent, "starter",])
pprint(steps)


# from eval import evaluate_and_log
# module = runpy.run_path("agent_v4.py")
# fresh_agent = module["agent"]
# evaluate_and_log(fresh_agent, "starter", 10, "agent_v4", "starter")


