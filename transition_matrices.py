import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import sys

import download_data as dd

def generate_transition_matrix(l, split_size):
    """Generates a transition matrix from a list of data"""
    
    # Identify all unique elements and create an empty transition matrix
    elements = list(set(l))
    elements_len = len(elements)
    transition_matrix = [[0] * elements_len for x in range(elements_len)]

    # Create a zipped pair of each state and its next, and then a transition matrix
    for (i,j) in zip(l,l[split_size:]):
        transition_matrix[i][j] += 1
    for i in transition_matrix:
        if sum(i) > 0:
            i[:] = [x/sum(i) for x in i]
    
    """
    # Print transition matrix
    for _ in transition_matrix:
        print([np.around(x, 4) for x in _])
    """
    return transition_matrix



def generate_combine_transition_matrices(symbols, start, end, choice, min_days = 1, max_days = 1):
    """Runs each part of the progrm for each symbol and combines the transition matrices"""
    
    list_tm = []
    
    # Generate transition matrix for each symbol
    for i in symbols:
        df = dd.download_stock_data(i, start, end)
        list_inner_tm = []
        for j in list(range(min_days, max_days + 1)):
            transitions = choice(df, j)
            transition_matrix = generate_transition_matrix(transitions, j)
            list_inner_tm.append(transition_matrix)
        list_tm.append(list_inner_tm)
    
    # Average transition matrix for each symbol, this still satisfies Kolmogorov axioms
    avg_list, first, c_s = [], True, 0
    for i in list_tm:
        c = 0
        print(i.index)
        for j in i:
            try:
                if first == True:
                    avg_list.append(np.array(j))
                else:
                    avg_list[c] = avg_list[c] + np.array(j)
                    c = c + 1
            except:
                print('ERROR OCCURS AT ENTRY '+str(c)+' FOR EQUITY '+symbols[c_s])
                c = c + 1
                continue
        first = False
        c_s = c_s + 1
    for k in range(0, len(avg_list)):
        avg_list[k] = avg_list[k] / len(symbols)

    return avg_list
    


def stationary_values(transition_matrix, mmult_number = 50):
    """From a transition matrix, generate stationary values"""
            
    # Create a 1st state array
    state = np.array([[0.0] * len(transition_matrix)])
    state[0][0] = 1.0

    # Create a state tracker
    stateHist = state
    dfStateHist = pd.DataFrame(state)
    distr_hist = [[0,0]]
    
    # Calculate stationary values
    for x in range(mmult_number):
        state = np.dot(state, transition_matrix)
        stateHist = np.append(stateHist,state,axis=0)
        dfDistrHist = pd.DataFrame(stateHist)
    svals = dfDistrHist[-1:].iloc[0]
        
    return svals


def plot_summaries(tms, choice):
    """Runs the requested plotting summary tool"""
    choice(tms)
    
    return None
    


################################
#### DAY CHANGE TRANSITIONS ####
################################

def day_change_transitions(df, num_days):
    """Calculates a measure of change from good/middle/bad days sto good/middle/bad."""
    
    # Controls
    min_change = 0.001
    
    # How has the stock performed in the last X days?
    df[str(num_days)+'D Change'] = df['Close'] - df.loc[:,'Close'].shift(num_days)
    df[str(num_days)+'D Change Switch'] = np.where(df[str(num_days)+'D Change'] > 0, True, False)

    # Is the change significant?
    df['Change Percent Value'] = np.absolute(df[str(num_days)+'D Change'] / df['Close'])
    df['Change Percent Value Switch'] = np.where(df['Change Percent Value'] > min_change, True, False)

    # Combine everything
    conditions = [
        (df[str(num_days)+'D Change Switch'] == True) & (df['Change Percent Value Switch'] == True),
        (df[str(num_days)+'D Change Switch'] == False) & (df['Change Percent Value Switch'] == True),
        (df['Change Percent Value Switch'] == False)]
    
    # Assign categories
    choices = [0, 1, 2]
    df['Output'] = np.select(conditions, choices, default = 2)
    
    # Isolate transitions
    transitions = list(df['Output'])[num_days:]

    # Plot it because it pretty
    #df[['Close', str(num_days)+'D Change']].plot()
    #plt.show()
    
    return transitions



def plot_summaries_day_change(tms):
    """Plots the summary of the data"""
    
    l_stationary = []
    for _ in tms:
        l_stationary.append(list(stationary_values([list(x) for x in _])))

    for j in range(0, 3):
        x = [x[j] for x in l_stationary]
        plt.plot(x)
    plt.legend(['Good', 'Bad', 'No Change'])
    plt.title('Stationary Odds for Lengths of Time of Measurements')
    plt.ylabel('Chance of Occurance')
    plt.xlabel('Days')
    plt.show()

    for i in range(0, 2):
        for j in range(0, 2):
            x = np.array([x[j][i] for x in tms])
            plt.plot(x)
    plt.legend(['Good >>> Good', 'Good >>> Bad', 'Bad >>> Good', 'Bad >>> Bad'])
    plt.title('Transition Odds for Lengths of Time of Measurements')
    plt.ylabel('Chance of Occurance')
    plt.xlabel('Days')
    plt.show()

    for i in range(0, 2):
        for j in range(0, 2):
            x = np.array([x[j][i] for x in tms])
            x = x - [x[i] for x in l_stationary]
            plt.plot(x)
    plt.legend(['Good >>> Good', 'Good >>> Bad', 'Bad >>> Good', 'Bad >>> Bad'])
    plt.title('Adjusted for Stationary Odds')
    plt.ylabel('Chance of Occurance')
    plt.xlabel('Days')
    plt.show()
    
    return None