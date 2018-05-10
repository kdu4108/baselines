import numpy as np
import matplotlib.pyplot as plt
import json
def autolabel(rects, ax):
    """
    Attach a text label above each bar displaying its height
    """
    ax_height = np.absolute(ax.get_ylim()[1] - ax.get_ylim()[0])
    shift = ax_height * 0.03
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., shift + height,
                '%d' % int(height),
                ha='center', va='bottom')

def plotRewardDist(rewardFreq, numTrials):
    """
    Make a bar graph of distinct rewards and frequencies
    Input: rewardFreq, a dictionary mapping from rewards to counts
           numTrials, the number of trials recorded
    """
    fig, ax = plt.subplots()
    p = plt.bar(list(rewardFreq.keys())[1:], 
        list(rewardFreq.values())[1:], color='g')
    ax.set_xlabel('Discrete Reward')
    ax.set_ylabel('Num Times Earned over ' + str(numTrials) + ' Trials')
    ax.set_title('Amidar Reward Distribution across ' + str(numTrials) + ' Trials')
    autolabel(p, ax)
    plt.show()

def clippedVsNonClipped(clipped_agent, nonclipped_agent):
    """
    Make a side-by-side bar graph of performance of clipped agent and nonclipped agent
    in clipped and nonclipped environments
    Input: clipped_agent: Dictionary mapping from 'Nonclipped' to list of scores and 'Clipped' to list of scores
           nonclipped_agent: Dictionary mapping from 'Nonclipped' to list of scores and 'Clipped' to list of scores
    """
    
    # clipped_agent = {'Nonclipped': [1752.0, 1949.0, 1987.0, 1312.0, 1943.0, 1850.0, 1938.0, 1636.0, 1994.0, 2083.0], 'Clipped': [181, 186, 187, 106, 185, 181, 182, 159, 184, 189]}
    # nonclipped_agent = {'Clipped': [86, 84, 86, 84, 84, 88, 88, 86, 88, 86], 'Nonclipped': [907.0, 912.0, 961.0, 955.0, 905.0, 963.0, 918.0, 915.0, 917.0, 958.0]}

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
    autolabel(rects1)
    autolabel(rects2)

    plt.show()

def parseTitle(filename):
    lstFileName = filename.split("/")
    x = lstFileName[-2].split("model-scale")[1]
    model_num = int(lstFileName[-1].split("-")[1])
    if model_num < 47000000:
        x += "*"
    x = x.replace('_', '0.')
    return x

def rewardVsScale(resultsDict):
    """
    Make a bar graph of scales and rewards
    Input: resultsDict, a dictionary mapping from scales to rewards
    """
    fig, ax = plt.subplots()
    numTrials = len(list(resultsDict.values())[0]["Nonclipped"])
    x_vals = list(map(parseTitle, list(resultsDict.keys())))
    print("X: " + str(x_vals))
    y_vals = np.mean(list(map(lambda x: x['Nonclipped'], list(resultsDict.values()))), 1)
    print("Y: " + str(y_vals))
    p = plt.bar(x_vals, 
        y_vals, color='g')
    ax.set_xlabel('Scale Factor')
    ax.set_ylabel('Average Reward')
    ax.set_title('Amidar Average Reward across ' + str(numTrials) + ' Trials using Scaled Reward')
    autolabel(p, ax)
    plt.show()

if __name__ == '__main__':
    with open('/home/kevin/Documents/RL/openai/baselines3.4/baselines/rewardScaleResultsEnjoyMany.json') as handle:
        resultsDict = json.loads(handle.read())
    rewardVsScale(resultsDict)
