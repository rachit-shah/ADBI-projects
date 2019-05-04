########################################################################################################################
# Author: Rachit Shah                                                                                                  #
# Unity ID: rshah25                                                                                                    #
# Assignment: Adwords                                                                                                  #
# Course: CSC 591 ADBI                                                                                                 #
# Date: 3/3/2019                                                                                                       #
########################################################################################################################

# Import Libraries
import numpy as np
import pandas as pd 
import sys
import random
import math
random.seed(0)

# Function to check whether any advertiser has enough budget for the bid. If none, return True
def all_wallets_empty(bids, advBudget):
    for x in bids:
        if bids[x] <= advBudget[x]:
            return False
    return True

# Function for implementing the greedy algorithm to calculate revenue
def greedy(queries, advBudget, queBids):
    revenue = 0
    for q in queries:
        bids = queBids[q]
        # If no advertiser has enough budget, don't display any ads, i.e., 0 revenue
        if all_wallets_empty(bids, advBudget):
            continue
        else:
            # Find the advertiser with the highest bid by iterating through all advertisers that have bidded for query q
            highestBidder = None
            highestBid = -math.inf
            for x in bids:
                # Consider the advertiser only if it has enough budget for the bid
                if advBudget[x] >= bids[x]:
                    if bids[x] > highestBid:
                        highestBid = bids[x]
                        highestBidder = x
                    # If same bid, choose the advertiser with smaller id
                    elif bids[x] == highestBid:
                        if x < highestBidder:
                            highestBidder = x
        # Add the highestbid to revenue
        revenue += highestBid
        # Subtract the amount of bid from the highest bidder's budget
        advBudget[highestBidder] -= highestBid

    return revenue

# Function to calculate the psi value for MSVV algorithm
def psiBid(remBudget, origBudget):
    # Get the ratio of spent budget
    ratioRemBudget = (origBudget - remBudget)/origBudget
    # Apply formula
    psi = 1 - math.exp(ratioRemBudget - 1)
    return psi

# Function for implementing the MSVV algorithm to calculate revenue
def msvv(queries,advBudget,remAdvBudget,queBids):
    revenue = 0
    for q in queries:
        bids = queBids[q]
        # If no advertiser has enough budget, don't display any ads, i.e., 0 revenue
        if all_wallets_empty(bids, remAdvBudget):
            continue
        else:
            # Find the advertiser with the highest value of bid * PSI function by iterating through all advertisers that have bidded for query q
            highestBidder = None
            highestBid = -math.inf
            for x in bids:
                # Consider the advertiser only if it has enough budget for the bid
                if remAdvBudget[x] >= bids[x]:
                    if highestBidder is not None:
                        psiHigh = highestBid * psiBid(remAdvBudget[highestBidder],advBudget[highestBidder])
                    else:
                        psiHigh = -math.inf
                    psiCur = bids[x] * psiBid(remAdvBudget[x],advBudget[x])

                    if psiCur > psiHigh:
                        highestBid = bids[x]
                        highestBidder = x
                    # If same bid, choose the advertiser with smaller id
                    elif psiCur == psiHigh:
                        if x < highestBidder:
                            highestBidder = x
        # Add the highestbid to revenue
        revenue += highestBid
        # Subtract the amount of bid from the highest bidder's budget
        remAdvBudget[highestBidder] -= highestBid

    return revenue

# Function for implementing the Balance algorithm to calculate revenue
def balance(queries, advBudget, queBids):
    revenue = 0
    for q in queries:
        bids = queBids[q]
        # If no advertiser has enough budget, don't display any ads, i.e., 0 revenue
        if all_wallets_empty(bids, advBudget):
            continue
        else:
            # Find advertiser with the highest budget by iterating through all advertisers that have bidded for query q
            highestBidder = list(bids.keys())[0]
            for x in bids:
                # Consider the advertiser only if it has enough budget for the bid
                if advBudget[x] >= bids[x]:
                    if advBudget[x] > advBudget[highestBidder]:
                        highestBidder = x
                    # If same budget, choose the advertiser with smaller id
                    elif advBudget[x] == advBudget[highestBidder]:
                        if x < highestBidder:
                            highestBidder = x
        # Add the highestbid to revenue
        revenue += bids[highestBidder]
        # Subtract the amount of bid from the highest bidder's budget
        advBudget[highestBidder] -= bids[highestBidder]
    return revenue

# Main Function
def main():
    # Read bidders csv into pandas dataframe
    df = pd.read_csv("bidder_dataset.csv")
    # Initialize dictionary to store budgets of all advertisers   {advID:budget}
    advBudget = {}
    # Initialize dictionary to store bids for each unique query  {query:{advID:bid}}
    queBids = {}
    # Populate dictionaries by traversing whole df
    for x in range(len(df)):
        temp = df.iloc[x]
        advId = temp[0]
        query = temp[1]
        bid = temp[2]
        budget = temp[3]
        #print(budget)
        # if the first row for advertiser, add its budget
        if not np.isnan(budget):
            #print("*",budget,advId)
            advBudget[advId] = budget
        # Initialize inner dictionary for the specific query {query:{advID:bid}}
        if query not in queBids:
            queBids[query] = {}
        # Populate inner dictionary with the bids of the advertisers
        if advId not in queBids[query]:
            queBids[query][advId] = bid

    # The optimal revenue is the sum of all advertisers budget
    optimal = sum(advBudget.values())

    # Read all queries from file
    with open('queries.txt') as file:
        queries = list(map(lambda x: x.strip(),file.readlines()))

    # Get method from args and perform the provided method 100 times and take the mean of all revenues
    method = sys.argv[1]
    avgRevenue = 0

    if method == "greedy":
        revenue = greedy(queries, advBudget.copy(), queBids)
    elif method == "msvv":
        revenue = msvv(queries, advBudget.copy(), advBudget.copy(), queBids)
    elif method == "balance":
        revenue = balance(queries, advBudget.copy(), queBids)
    else:
        print("Invalid argument")
        return

    print(round(revenue,2))

    for i in range(100):
        # Shuffle queries to produce different results for each iteration
        random.shuffle(queries)
        if method == "greedy":
            revenue = greedy(queries,advBudget.copy(),queBids)
        elif method == "msvv":
            revenue = msvv(queries,advBudget.copy(),advBudget.copy(),queBids)
        elif method == "balance":
            revenue = balance(queries,advBudget.copy(),queBids)
        avgRevenue+=revenue
    avgRevenue /= 100

    # Competitive Ratio is ALG/OPT
    compRatio = avgRevenue/optimal
    print(round(compRatio,2))
  
if __name__== "__main__":
    main()

