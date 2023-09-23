import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

PATH = 'results/sacred/'
RUN_ID = '1'

with open(PATH+RUN_ID+'/metrics.json', 'r') as f:

    metrics_dict = json.load(f)

    agent_0_returns = metrics_dict['agent0/episode_reward']['values']
    agent_1_returns = metrics_dict['agent0/episode_reward']['values']
    total_returns = np.array(agent_0_returns) + np.array(agent_1_returns)

    steps = np.array(metrics_dict['agent0/episode_reward']['steps']) * 20

sns.set_theme()
plt.plot(steps, total_returns, label='SEAC', c='g')
plt.xlabel('Environment Steps')
plt.ylabel('Returns')
plt.legend(loc='lower right')
plt.title('LBF: (8x8), two agents, two foods, cooperative')
# plt.savefig(PATH+RUN_ID+'/plot.svg', format="svg")
plt.savefig(PATH+RUN_ID+'/plot.png')