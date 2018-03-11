import numpy as np
import matplotlib.pyplot as plt

clipped_agent = {'Nonclipped': [1752.0, 1949.0, 1987.0, 1312.0, 1943.0, 1850.0, 1938.0, 1636.0, 1994.0, 2083.0], 'Clipped': [181, 186, 187, 106, 185, 181, 182, 159, 184, 189]}
nonclipped_agent = {'Clipped': [86, 84, 86, 84, 84, 88, 88, 86, 88, 86], 'Nonclipped': [907.0, 912.0, 961.0, 955.0, 905.0, 963.0, 918.0, 915.0, 917.0, 958.0]}

avg_clipped_agent_nonclipped_score = np.average(clipped_agent.get('Nonclipped'))
avg_clipped_agent_clipped_score = np.average(clipped_agent.get('Clipped'))
avg_nonclipped_agent_nonclipped_score = np.average(nonclipped_agent.get('Nonclipped'))
avg_nonclipped_agent_clipped_score = np.average(nonclipped_agent.get('Clipped'))

N = 2

ind = np.arange(N)  # the x locations for the groups
width = 0.35       # the width of the bars

fig, ax = plt.subplots()

clipped_agent_means = (avg_clipped_agent_clipped_score, avg_clipped_agent_nonclipped_score)
nonclipped_agent_means = (avg_nonclipped_agent_clipped_score, avg_nonclipped_agent_nonclipped_score)

rects1 = ax.bar(ind, clipped_agent_means, width, color='r')
rects2 = ax.bar(ind + width, nonclipped_agent_means, width, color='y')

# add some text for labels, title and axes ticks
ax.set_xlabel('Type of Reward/Env')
ax.set_ylabel('Average Scores (n = 10)')
ax.set_title('Scores by Clipped Total Reward and Clipped Agent')
ax.set_xticks(ind + width / 2)
ax.set_xticklabels(('Clipped Total Reward', 'Actual Total Reward'))

ax.legend((rects1[0], rects2[0]), ('Clipped Agent', 'Nonclipped Agent'))


def autolabel(rects):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 10+height,
                '%d' % int(height),
                ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)

plt.show()