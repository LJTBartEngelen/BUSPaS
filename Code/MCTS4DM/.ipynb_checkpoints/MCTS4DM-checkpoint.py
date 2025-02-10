''' Code is largely based on the MCTS4DM Java implementation of Bosc et. al 2018 '''


import pandas as pd
import numpy as np
import time
import random
import heapq

import sys
import os

from refine import *
from UCB import *
sys.path.insert(0, '../')
from dataImporter import *
from helperFunctions import *
from qualityMeasure import *


class BoundedPriorityQueue:
    """
    Used to store the <q> most promising subgroups
    Ensures uniqueness
    Keeps a maximum size (throws away value with least quality)
    """

    def __init__(self, bound):
        # Initializes empty queue with maximum length of <bound>
        self.values = []
        self.bound = bound
        self.entry_count = 0

    
    def add(self, element, quality, coverage, **adds):
        # Adds <element> to the bounded priority queue if it is of sufficient quality
        new_entry = (quality, coverage, self.entry_count, element, adds)

        if (len(self.values) >= self.bound):
            heapq.heappushpop(self.values, new_entry)
        else:
            heapq.heappush(self.values, new_entry)

        self.entry_count += 1

    
    def get_values(self):
        # Returns elements in bounded priority queue in sorted order
        for (q, coverage, element, count, _) in sorted(self.values, reverse=True):
            yield (q, coverage, element, count)

    
    def get_element_sets(self):
        # Returns elements in bounded priority queue in sorted order
        element_sets = []
        for (_, _, e, _,_) in sorted(self.values, reverse=True):
            element_sets.append(set(e))
        return element_sets

    
    def show_contents(self):
        # Prints contents of the bounded priority queue (used for debugging)
        print("show_contents")
        for (q, coverage, element, entry_count, _) in self.values:
            print(q, coverage, element, entry_count)



###TODO LATER Add Diversity check

def eval_quality(sub_group, df, target, quality_measure, comparison_type='complement',**kwargs):


    if comparison_type == 'population':
        complement_df = df
    elif comparison_type == 'complement':
        complement_df = df.loc[~df.index.isin(sub_group.index)]
    else:
        complement_df = df

    phi = quality_measure(sub_group, target, complement_df, **kwargs)

    coverage = len(sub_group)/len(df)

    return phi, coverage



class Subgroup:
    
    def __init__(self, description, parent=None):
        """
        Initialize a new node in the MCTS tree.

        Args:
        - description (str): The subgroup description or state represented by this node.
        - parent (Subgroup, optional): The parent node of this node.
        """
        self.description = description
        self.parent = [parent]
        self.children = None
        self.candidates = None
        self.visits = 0
        self.value = 0.0
        self.quality = 0.0
        self.fullExpanded = False
        self.fullTerminated = False
        self.hashed_coverage = 0
        self.coverage = 0.0

        self.min_heap = []
        self.total_sum = 0.0

        self.std = None

    
    def add_child(self, child_node):
        """Adds a child node to this node."""
        self.children.append(child_node)

    
    def update(self, new_value):
        """Updates the node's statistics with a new reward value."""
        self.value = new_value
        self.visits += 1

    
    def print_info(self):
        """Prints all relevant information about the subgroup."""
        # parent_desc = self.parent.description if self.parent else "None"
        print(f"Description: {self.description}")
        # print(f"Parent: {parent_desc}")
        print(f"Number of children: {len(self.children)}") if type(self.children) == list else print(f"Number of children: {0}")
        print(f"Visits: {self.visits}")
        print(f"Value: {self.value}")
        print(f"Quality: {self.quality}")
        print(f"Full Expanded: {self.fullExpanded}")
        if (type(self.children) == list) and len(self.children)>0:
            print("Children Descriptions:")
            for child in self.children:
                print(f"  - {child.description}")
        else:
            print("Has No Children")
        

UCB_FUNCTIONS = {
    'UCB1': (ucb1, ['exploration_weight']),
    'UCB1-Tuned': (ucb1_tuned, ['exploration_weight']),
    'SP-MCTS': (sp_mcts, ['exploration_weight']),
    'UCT': (uct, ['exploration_weight']),
    'DFS-UCT': (dfs_uct, ['exploration_weight', 's', 'dfs_factor'])
}



class MCTS4DM:
    
    def __init__(self, df, target_column = 'target',
                 q = 10,
                 root_description = [],
                 n_chunks = 5,
                 allow_exclusion = False,
                 minutes = float('inf'),
                 max_nr_iterations = float('inf'),
                 ucb_type='UCB1', #or 'UCB1', 'UCB1-Tuned', 'SP-MCTS', 'UCT, 'DFS-UCT'
                 ucb_params={},
                 quality_params={},
                 matrix = None,
                 size_correction_method = no_size_corr,
                 max_desc_length=float('inf'),
                 min_coverage=0.00001,
                 roll_out_strategy='direct-freq', #or 'large-freq' or 'naive'
                 roll_out_jump_length = 1, # if >1 then roll-out-strategy = 'large-freq' 
                 reward_policy = 'max_path', #or 'random_pick', 'mean_path', 'mean_top_k'
                 reward_policy_k = 3,
                 memory_policy = 'all', #or 'top_k'
                 memory_policy_k_value = 3,
                 update_policy = 'max_update', #or 'mean_update', 'top_k_mean_update'
                 update_policy_k = 3,
                 diversity_policy = 'AMAF',
                 show_progress = False
                ):
        """
        Initialize the MCTS4DM with a root node.

        Args:
        - df (df): dataframe 
        - target_column (str): column name target containing target attributes/space (should be of same len as len(df_descriptives) )
        - root_description (str): The description of the root node
        - q (int): The number of best subgroups retrieved
        - root_description (list): description used to start from = []
        - n_chunks (int): number of bins used for numerical attributes in refinement
        - allow_exclusion (bool): allow conditions that exlude categorical values in descriptions
        - minutes (float): time limiting parameter for search
        - max_nr_iterations (int): iteration limiting parameter for search
        - ucb_type (str): UCB policy: options ->  'UCB1', 'UCB1-Tuned', 'SP-MCTS', 'UCT, 'DFS-UCT'
        - ucb_params={}
        - quality_params={}
        - matrix (i x i matrix): distance matrix used for distance based quality measure (i should be equal to len(df_descriptives) )
        - size_correction_method (str): subgroup size correction policy
        - max_desc_length (int): max length of descriptions
        - min_coverage (float): minimal coverage of a subgroup compared as quotient of population
        - roll_out_strategy (str): roll-out policy: options -> 'direct-freq' 'large-freq' or 'naive'
        - roll_out_jump_length (int): jump length in case of large-freq, roll_out_jump_length=1 == 'direct-freq'
        - reward_policy (str): reward policy 'max_path', 'random_pick', 'mean_path', 'mean_top_k'
        - reward_policy_k (int): k value in case reward policy 'mean_top_k' is chosen
        - memory_policy (str): memory policy: options -> 'all' 'top_k'
        - memory_policy_k_value (int): k value in case memory policy 'top_k' is chosen
        - update_policy (str): 'max_update', 'mean_update', 'top_k_mean_update'
        - update_policy_k (int):  k value in case update policy 'top_k_mean_update' is chosen
        - diversity_policy (str): #"AMAF"
        
        """
        
        self.n_chunks = n_chunks
        self.allow_exclusion = allow_exclusion

        self.q = q
        self.result_set = BoundedPriorityQueue(self.q)
        self.max_desc_length  = max_desc_length
        self.min_cov  = min_coverage
        
        self.minutes = minutes
        self.max_nr_iterations = max_nr_iterations
        self.nr_iterations = 0
        
        self.root = Subgroup(root_description)
        self.root.hashed_coverage = None
        
        self.df_descriptives = df.drop(columns=[target_column])
        self.N = len(self.df_descriptives)
        self.features = self.df_descriptives.columns
        self.df_target = df[target_column]

        self.len_df = len(self.df_descriptives)

        df = None

        self.roll_out_strategy = roll_out_strategy
        self.roll_out_jump_length = roll_out_jump_length
        self.reward_policy = reward_policy
        self.reward_policy_k = reward_policy_k

        self.memory_policy = memory_policy #or 'all' or 'top_k'
        self.memory_policy_k_value = memory_policy_k_value

        self.update_policy = update_policy
        self.update_policy_k = update_policy_k

        self.diversity_policy = diversity_policy #"AMAF"
        self.amaf = {}

        ###TODO LATER add parameters for adapted/different quality measure(s), also in variable definition in function
        self.matrix = matrix # only used in specific quality measure
        
        self.size_correction_method = size_correction_method

        # UCB initialization
        if ucb_type not in UCB_FUNCTIONS:
            raise ValueError(f"Unknown UCB type: {ucb_type}")
        self.ucb_type = ucb_type
        self.ucb_function, self.ucb_args = UCB_FUNCTIONS[ucb_type]
        
        self.ucb_params = ucb_params

        self.show_progress = show_progress

    
    def __hash__(self,item):
        return hash(tuple(item))
    
    
    def select(self):
        """
        SELECT
        
        Select a node to expand based on the UCB1 formula.
        
        Returns:
        - Subgroup: The selected node for expansion.
        """
        ### (Bosc 2028 says) use SP-MCTS
        
        node = self.root
        
        while node.fullExpanded == True:
            parent_node = node
            node = self._best_child(node)
            if node is None:
                parent_node.fullTerminated = True
                parent_node.fullExpanded = True

                return node
        return node


    def _best_child(self, node, exploration_weight= 2/(2**0.5) ):
        """
        Select the best child node using a UCB.

        Args:
        - node (Subgroup): The parent node.
        - exploration_weight (float): The exploration parameter.

        Returns:
        - Node: The child node with the highest UCB value.
        """
        best_value = float('-inf')
        best_node = None
        params = {arg: self.ucb_params.get(arg, exploration_weight) for arg in self.ucb_args}
        
        for child in node.children:

            if child is None or child.fullTerminated:
                continue
        
            ucb_value = self.ucb_function(node, child, **params)

            if ucb_value > best_value:
                best_value = ucb_value
                best_node = child

        if best_node is None:
            node.fullTerminated = True
        
        return best_node


    def expand(self, node):
        ### (Bosc 2018 says) Expand: We advise to use the label-gen strategy that enables to reach more quickly
        #the best patterns, but it can require more computational time.
        ### (Bosc 2028 says) use PU(=AMAF)

        ###TODO LATER Implement LO
        ###TODO LATER (after LO is implemented) Normalized exploration rate
        
        """
        EXPAND
        
        Expand the given node by adding all possible actions as children.

        Args:
        - node (Subgroup): The node to expand.
        """

        
        extended = False
        
        if node.children is None:
            node.children = []
        if node.candidates is None:
            node.candidates = []
            nr_candidates = 0

            for desc in eta(node.description, self.df_descriptives, self.features, self.n_chunks, self.allow_exclusion):
                node.candidates.append(desc)
        nr_candidates = len(node.candidates)
        
        while extended == False and len(node.candidates)>0:
            
            chosen_candidate_idx = random.randint(0, nr_candidates-1)
            
            chosen_candidate_description = node.candidates[chosen_candidate_idx]
            del node.candidates[chosen_candidate_idx]
            nr_candidates = len(node.candidates)

            subgroup = self.df_descriptives.query(as_string(chosen_candidate_description))
            
            hashed_coverage_candidate = self.__hash__( subgroup.index ) #create hash of de coverage of a description
                        
            if hashed_coverage_candidate != node.hashed_coverage and (len(subgroup)/self.N) >= self.min_cov: #we don't pick an subgroup that has the same items as the parents (gen-expand)
                
                if self.diversity_policy == "AMAF":
                
                    if self.__hash__(chosen_candidate_description) in self.amaf:
                        
                        child_node = self.amaf[self.__hash__(chosen_candidate_description)]
                        child_node.parent.append(node)
                    else:                
                        child_node = Subgroup(description=chosen_candidate_description, parent=node)
                        child_node.hashed_coverage = hashed_coverage_candidate
                        ###TODO LATER make quality measure more elegant/flixble, below
                        child_node.quality, child_node.coverage = eval_quality(subgroup, self.df_descriptives, 'target', quality_measure = cluster_based_quality_measure, quality_comparison_type = "complement", distance_matrix = self.matrix, correct_for_size = self.size_correction_method)
                        self.amaf[self.__hash__(chosen_candidate_description)] = child_node
                
                else:
                    child_node = Subgroup(description=chosen_candidate_description, parent=node)
                    child_node.hashed_coverage = hashed_coverage_candidate
                    ###TODO LATER make quality measure more elegant/flixble, below
                    child_node.quality, child_node.coverage = eval_quality(subgroup, self.df_descriptives, 'target', quality_measure = cluster_based_quality_measure, quality_comparison_type = "complement", distance_matrix = self.matrix, correct_for_size = self.size_correction_method)
                
                node.add_child(child_node)
                
                extended = True

            ###TODO NOW TEST AMAF strategy
        
        if nr_candidates == 0:
            node.fullExpanded = True
            node.fullTerminated = True
            child_node = None
        
        try:
            return child_node
        except:
            child_node = None
            return child_node

    def simulate(self, node):
        ### (Bosc 2018) RollOut: For nominal attributes, the direct-freq-roll-out is an efficient strategy.
        # However, when facing numerical attributes, we recommend to employ the largefreq-
        # roll-out since it may require a lot of time to reach the maximal frequent
        # patterns.
        
        """
        ROLL-OUT
        
        Simulate the outcome from a given node.

        Args:
        - node (Subgroup): The node from which the simulation starts.

        Returns:
        - float: The simulated reward.
        """

        s_exp = node
        desc_length = len(node.description)
        path = [s_exp]
        reward = float('-inf')
        if self.roll_out_strategy == 'direct-freq' or self.roll_out_strategy == 'large-freq':
            
            jump_length_start = random.randint(1, self.roll_out_jump_length)
            jump_length = jump_length_start
            
            while (desc_length < self.max_desc_length) and (node.fullTerminated == False):

                if jump_length == 0:
                    jump_length = jump_length_start

                if node.children is None:
                    node.children = []                
                if node.candidates is None:
                    node.candidates = []
    
                    for desc in eta(node.description, self.df_descriptives, self.features, self.n_chunks, self.allow_exclusion):
                        node.candidates.append(desc)
    
                # if len(node.candidates) > 0:    
                found = False
                while (found == False) and (len(node.candidates) > 0):
                    candidate_description = random.choice(node.candidates)
                    node.candidates.remove(candidate_description)
                
                    candidate_subgroup = Subgroup(candidate_description,node)
                    subgroup_df = self.df_descriptives.query(as_string(candidate_description))
                    
                    ###TODO LATER add flexible quality measure when is there

                    if jump_length == 1:
                        candidate_subgroup.quality, candidate_subgroup.coverage = eval_quality(subgroup_df, self.df_descriptives, 'target', quality_measure = cluster_based_quality_measure, quality_comparison_type = "complement", distance_matrix = self.matrix, correct_for_size = self.size_correction_method)
                    else:
                        candidate_subgroup.coverage = len(subgroup_df)/self.len_df

                    hashed_coverage_candidate = self.__hash__( subgroup_df.index ) #create hash of de coverage of a description
                    if hashed_coverage_candidate != node.hashed_coverage and candidate_subgroup.coverage >= self.min_cov:
                        
                        node.add_child(candidate_subgroup)
                        if jump_length == 1:
                            path.append(candidate_subgroup)
                        desc_length = len(candidate_subgroup.description)
                        
                        node = candidate_subgroup
                        found = True
                    elif (len(node.candidates) == 0):
                        found = True
                        node.fullTerminated = True
                    else:
                        found = None
                        node.fullTerminated = True
                jump_length -= 1


        elif self.roll_out_strategy == 'naive':
            print("ATTENTION! Roll out strategy: 'naive', not implemented yet")
            pass
            ###TODO LATER
        
        memory_list = [] ###TODO LATER MAYBE can probably be more efficient with some tree shaped Data Structure
        if self.reward_policy == 'random_pick': 
            index = self.r.randint(0, len(path) - 1)
            group = path[index]
            reward = group.quality
            memory_list.append(group)
        elif self.reward_policy == 'mean_path':
            reward = 0
            for group in path:
                reward += group.quality
            reward /= len(path)
            memory_list.extend(path)
        elif self.reward_policy == 'max_path':
            reward = 0
            for group in path:
                if reward < group.quality:
                    reward = group.quality
            memory_list.extend(path)
        elif self.reward_policy == 'mean_top_k':
            k_placeholder = 0 
            sum_reward = 0
            rankedPath = [(-item.quality, item) for item in path]
            heapq.heapify(ranked_path)
            while k_placeholder <= self.reward_policy_k:
                neg_quality, item = heapq.heappop(max_heap)
                sum_reward += -neg_quality
                k_placeholder += 1
            reward = sum_reward / self.reward_policy_k
            memory_list.extend(path)        
   
        if self.memory_policy == 'all':
            for group in memory_list:
                self.result_set.add( group.description, group.quality, group.coverage )
        elif self.memory_policy == 'top_k':
            top_k_subgroups = heapq.nlargest(self.memory_policy_k_value, memory_list, key=lambda x: x.quality)
            for group in top_k_subgroups:
                self.result_set.add( group.description, group.quality, group.coverage )
        
        return reward

    def backpropagate(self, node, reward):
        ###(Bosc 2018) Update: When there are potentially many local optima in the search space, we
        #recommend to set the mean-update strategy for the Update step. Indeed it enables
        #to exploit the areas that are deemed to be interesting in average. However, when
        #there are fewlocal optima among lots of uninteresting patterns, using mean-update
        #is not optimal since the mean of the rewards would converge to 0. In place, the
        #max-update should be used to ensure that an area containing a local optima is well
        #identified.
        
        
        """
        UPDATE
        
        Backpropagate the reward through the nodes to update statistics.

        Args:
        - node (Subgroup): The node where the reward is backpropagated.
        - reward (float): The reward to backpropagate.
        """
        #ucb_type='UCB1', #or UCB1, UCB1-Tuned, SP-MCTS, UCT, DFS-UCT

        def backpropagate_recursive(node, reward):
            
            if (self.ucb_type == 'SP-MCTS') or (self.ucb_type == 'UCB1-Tuned'):
            #update std of rewards by new rewards
                if node.std:
                    node.std = update_std_dev(reward, node.mean, node.std, node.n)
                    node.mean = ((node.mean * node.n) + reward) / (node.n + 1)
                    node.n = node.n + 1
                else:
                    node.std = 0
                    node.mean = reward
                    node.n = 1    
                
            if self.update_policy == 'mean_update': 
                node.value = ( node.visits * node.value + reward ) / ( node.visits + 1 )
            elif self.update_policy == 'max_update':
                if reward > node.value:
                    node.value = reward
            elif self.update_policy == 'top_k_mean_update':
                if len(node.min_heap) < self.update_policy_k:
                    # If the heap is not full, add the new value.
                    heapq.heappush(node.min_heap, reward)
                    node.total_sum += reward
                else:
                    # If the new value is larger than the smallest in the heap, replace the smallest.
                    if reward > node.min_heap[0]:
                        node.total_sum += reward - node.min_heap[0]
                        heapq.heapreplace(node.min_heap, reward)
        
                # Update the value with the mean of the current top x values
                node.value = node.total_sum / len(node.min_heap)
                
                node.update(node.value)

                if node.parent and len(node.parent) >= 1:
                    for parent in node.parent:
                        backpropagate_recursive(parent, reward)
    
        backpropagate_recursive(node, reward)


    def show_results(self):
        print(' ')
        print(' ')
        print('--------------------------------- Results ---------------------------------')
        r = 1
        for i in self.result_set.get_values():
            #(q, coverage, element, count) # = sorted(self.result_set, key=lambda x: x[0], reverse=True)
            print('#',r,'\t q= ',round(i[0],4),' cov= ',round(i[1],4))
            if len(i[3])>2:
                print('\t desc= ',str(i[3][:2])[:-1],',')
                print('\t\t\t',str(i[3][2:])[1:] )
            else:
                print('\t desc= ',i[3])
                print(' ')
            r += 1
        print('---------------------------------------------------------------------------')
        print('\t')
        pass
    
    def run(self):
        
        start = time.time()

        print('started at ', time.ctime(start))

        while (time.time() - start < self.minutes*60) and (self.nr_iterations <= self.max_nr_iterations):
            s_sel = self.select()

            if s_sel is None:
                continue #return False

            s_exp = self.expand(s_sel)

            if s_exp is None or s_exp.fullExpanded:
                continue #return False

            reward = self.simulate(s_exp)
            
            self.backpropagate(s_exp, reward)
            
            self.nr_iterations += 1
            
            if self.show_progress:
                print('finished iteration: ',self.nr_iterations)
            
            time_elapsed = time.time() - start
            seconds_left = self.minutes * 60 - time_elapsed
            
            hours_left = int(seconds_left // 3600)
            minutes_left = int((seconds_left % 3600) // 60)
            seconds_remaining = int(seconds_left % 60)
            if self.show_progress:
                if seconds_remaining >= 0 and minutes_left >= 0 and hours_left >= 0:
                    print(f"Time left: {hours_left}h {minutes_left}m {seconds_remaining}s")
        
        print('finished after: ',self.nr_iterations,' iterations')
        self.show_results()
        pass










