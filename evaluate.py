from my_agents.carrot_agent import agent
from kaggle_environments import make

env = make("kaggriculture",
           configuration={"episodeSteps": 720,
                          "randomSeed": 0,},
                          debug=True,)

steps = env.run([agent, "starter",])


# from eval import evaluate_and_log
# module = runpy.run_path("agent_v4.py")
# fresh_agent = module["agent"]
# evaluate_and_log(fresh_agent, "starter", 10, "agent_v4", "starter")


