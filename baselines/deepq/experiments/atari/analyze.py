import numpy as np
import matplotlib.pyplot as plt
import json
def autolabel(rects, ax):
    """
    Attach a text label above each bar displaying its height
    """
    ax_height = np.absolute(ax.get_ylim()[1] - ax.get_ylim()[0])
    shift = ax_height * 0.01
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
 
    # rewardFreq = sorted(rewardFreq, key=rewardFreq.get)
    rewardFreqTupleList = list(rewardFreq.items())
    sortedRewardFreqTupleList = sorted(rewardFreqTupleList, key = lambda x: float(x[0]))
    sortedRewardFreqTupleList = list(map(lambda x: (float(x[0]), x[1]) , sortedRewardFreqTupleList))
    print(sortedRewardFreqTupleList)
    rewardFreqLists = list(zip(*sortedRewardFreqTupleList))
    print(rewardFreqLists)
    print(rewardFreqLists[0])
    print(rewardFreqLists[1])

    p = plt.bar(rewardFreqLists[0][1:], 
        rewardFreqLists[1][1:], color='g')  
    # p = plt.bar(list(rewardFreq.keys())[1:], 
    #     list(rewardFreq.values())[1:], color='g')
    ax.set_xlabel('Discrete Reward')
    ax.set_ylabel('Num Times Earned over ' + str(numTrials) + ' Trials')
    ax.set_title('Amidar Reward Distribution across ' + str(numTrials) + ' Trials')
    autolabel(p, ax)
    plt.show()

# def plotRewardDist2(resultsDict, numTrials):
    """
    Make a bar graph of distinct rewards and frequencies
    Input: rewardFreq, a dictionary mapping from rewards to counts
           numTrials, the number of trials recorded
    """
    # rewardDistList = []
    # for model in resultsDict:
    #     rewardDistList.append(resultsDict[model]["RewardDist"])

    # fig, ax = plt.subplots()
    # N = 100
    # ind = np.arange(N)
    # width = 0.35
    # for rewardFreq in rewardDistList:
    #     rewardFreqTupleList = list(rewardFreq.items())
    #     sortedRewardFreqTupleList = sorted(rewardFreqTupleList, key = lambda x: float(x[0]))
    #     sortedRewardFreqTupleList = list(map(lambda x: (float(x[0]), x[1]) , sortedRewardFreqTupleList))
    #     print(sortedRewardFreqTupleList)
    #     rewardFreqLists = list(zip(*sortedRewardFreqTupleList))
    #     print(rewardFreqLists)
    #     print(rewardFreqLists[0])
    #     print(rewardFreqLists[1])

    #     rects = ax.bar(ind,
    #         rewardFreqLists[1][1:], color='g') 
    #     autolabel(rects, ax)
    #     ind += width

    # # add some text for labels, title and axes ticks
    #     # p = plt.bar(list(rewardFreq.keys())[1:], 
    #     #     list(rewardFreq.values())[1:], color='g')
    # ax.set_xlabel('Discrete Reward')
    # ax.set_ylabel('Num Times Earned over ' + str(numTrials) + ' Trials')
    # ax.set_title('Amidar Reward Distribution across ' + str(numTrials) + ' Trials')
    # ax.set_xticks(ind + width / 2)
    # ax.set_xticklabels(('Clip', 'Log'))
    # # ax.legend((rects1[0], rects2[0]), ('Clipped Agent', 'Nonclipped Agent'))
    # plt.show()

def compareRewardDist(resultsDict):
    """
    Summary statistics on results for a given dictionary the include reward distributions
    Input: resultsDict: Dictionary mapping from path of model to "RewardDist", a dict
            containing the reward disttribution, plus other stuff
    Output: a dictionary mapping each agent to a list of summary statistics
    """

    modelToAvgDict = {}
    totalCountDict = {}
    for model in resultsDict:
        averageRew = 0.0
        totalCount = 0.0
        rewardDist = resultsDict[model]["RewardDist"]
        for rew in rewardDist:
            freq = rewardDist[rew]
            rew = float(rew)
            averageRew += freq*rew
            totalCount += freq
        modelToAvgDict[model] = averageRew/totalCount
        totalCountDict[model] = totalCount
    return {"average": modelToAvgDict, "totalCount": totalCountDict}





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

def parseTitleScale(filename):
    return parseTitle(filename, "scale")

def parseTitleShift(filename):
    return parseTitle(filename, "shift")
def parseTitleMisc(filename):
    return parseTitle(filename, "misc")


def parseTitle(filename, transformType):
    lstFileName = filename.split("/")
    if transformType == "scale":
        x = lstFileName[-2].split("model-scale")[1]
    elif transformType == "shift":
        x = lstFileName[-2].split("model-shift")[1]
    elif transformType == "misc":
        x = lstFileName[-2].split("model-")[1]
    print(lstFileName[-1].split("-"))
    model_num = int(lstFileName[-1].split("-")[1])
    if model_num % 1000000 == 0:
        x += "*"
    x = x.replace('_', '0.')
    x = x.replace("Neg", '-')
    return x

def rewardVsScale(resultsDict):
    """
    Make a bar graph of scales and rewards
    Input: resultsDict, a dictionary mapping from scales to rewards
    """
    fig, ax = plt.subplots()
    numTrials = len(list(resultsDict.values())[0]["Nonclipped"])
    x_vals = list(map(parseTitleScale, list(resultsDict.keys())))
    print("X: " + str(x_vals))
    y_vals = np.mean(list(map(lambda x: x['Nonclipped'], list(resultsDict.values()))), 1)
    print("Y: " + str(y_vals))
    p = plt.bar(x_vals, y_vals, color='g')
    ax.set_xlabel('Scale Factor')
    ax.set_ylabel('Average Reward')
    ax.set_title('Amidar Average Reward across ' + str(numTrials) + ' Trials using Scaled Reward')
    autolabel(p, ax)
    plt.show()

def rewardVsShift(resultsDict):
    """
    Make a bar graph of shifts and rewards
    Input: resultsDict, a dictionary mapping from scales to rewards
    """
    fig, ax = plt.subplots()
    numTrials = len(list(resultsDict.values())[0]["Nonclipped"])
    x_vals = list(map(parseTitleShift, list(resultsDict.keys())))
    y_vals = np.mean(list(map(lambda x: x['Nonclipped'], list(resultsDict.values()))), 1)
    xypairs = zip(x_vals, y_vals)

    # weird horrible sorting stuff so the bars are in the right order
    xypairsSorted = list(sorted(xypairs, key=lambda x: x[0]))
    xypairsNeg = xypairsSorted[0:3]
    xypairsPos = xypairsSorted[3:]
    xypairsNeg = xypairsNeg[::-1]
    xypairsSorted = xypairsNeg + xypairsPos
    x_vals = list(list(zip(*xypairsSorted))[0])
    y_vals = list(list(zip(*xypairsSorted))[1])
    p = plt.bar(range(len(x_vals)), list(map(float,y_vals)), color = 'g')
    plt.xticks(range(len(x_vals)), x_vals)
    ax.set_xlabel('Constant Shift')
    ax.set_ylabel('Average Reward')
    ax.set_title('Amidar Average Reward across ' + str(numTrials) + ' Trials using Shifted Reward')
    autolabel(p, ax)
    plt.show()

def rewardVsMisc(resultsDict):
    """
    Make a bar graph of misc transforms and rewards
    Input: resultsDict, a dictionary mapping from scales to rewards
    """
    fig, ax = plt.subplots()
    numTrials = len(list(resultsDict.values())[0]["Nonclipped"])
    x_vals = list(map(parseTitleMisc, list(resultsDict.keys())))
    print("X: " + str(x_vals))
    y_vals = np.mean(list(map(lambda x: x['Nonclipped'], list(resultsDict.values()))), 1)
    print("Y: " + str(y_vals))
    p = plt.bar(x_vals, y_vals, color='g')
    ax.set_xlabel('Reward Transformation')
    ax.set_ylabel('Average Reward')
    ax.set_title('Amidar Average Reward across ' + str(numTrials) + ' Trials using Misc. Reward Transforms')
    autolabel(p, ax)
    plt.show()
if __name__ == '__main__':
    # with open('/home/kevin/Documents/RL/openai/baselines3.4/baselines/rewardScaleResultsEnjoyMany.json') as handle:
    #     resultsDictScale = json.loads(handle.read())
    # rewardVsScale(resultsDictScale)
    with open('/home/kevin/Documents/RL/openai/baselines3.4/baselines/rewardShiftresults.json') as handle:
        resultsDictShift = json.loads(handle.read())
    rewardVsShift(resultsDictShift)
    with open('/home/kevin/Documents/RL/openai/baselines3.4/baselines/rewardMiscResults.json') as handle:
        resultsDictMisc = json.loads(handle.read())
    plotRewardDist(resultsDictMisc["/home/kevin/Documents/RL/openai/baselines3.4/baselines/rewardexperiments/misc/model-log/model-49537763"]["RewardDist"], numTrials = 10)
    print(compareRewardDist(resultsDictMisc))
    rewardVsMisc(resultsDictMisc)
