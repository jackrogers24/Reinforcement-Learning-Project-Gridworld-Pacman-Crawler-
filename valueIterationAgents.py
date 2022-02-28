# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()
    

    def runValueIteration(self):

        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        
        # Initialize value at each state to 0
        states = self.mdp.getStates()
        for i in states:
            self.values.update({i : 0})

        for i in range(self.iterations):

            # Create a copy of self.values to alter
            values_copy = self.values.copy()
            
            for state in values_copy:
                sums_list = []
                actions = self.mdp.getPossibleActions(state)
                for action in actions:
                    curr_sum = 0
                    for i in self.mdp.getTransitionStatesAndProbs(state, action):
                        prob = i[1]
                        next_state = i[0]
                        reward = self.mdp.getReward(state, action, next_state)
                        next_state_val = self.values[next_state]
                        curr_sum = curr_sum + prob * (reward + self.discount * next_state_val)
                    sums_list.append(curr_sum)
                
                if len(sums_list) == 0:
                    best_sum = 0
                else:
                    best_sum = max(sums_list)
                values_copy.update({state : best_sum})
            self.values = values_copy.copy()
                    



    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        q_value = 0
        states = self.mdp.getTransitionStatesAndProbs(state, action)
        for index, tuple in enumerate(states):
            next_state = tuple[0]
            prob = tuple[1]
            reward = self.mdp.getReward(state, action, next_state)
            q_value = q_value + (prob * (reward + self.discount * self.values[next_state]))
        return q_value


        

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"

        actions = self.mdp.getPossibleActions(state)
        if actions is None:
            return None
        elif "exit" in actions:
            return "exit"
        else:
            max_q = float('-inf')
            best_action = None
            for action in actions:
                if self.getQValue(state, action) > max_q:
                    max_q = self.getQValue(state, action)
                    best_action = action
            return best_action


    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"

        # Initialize value at each state to 0
        states = self.mdp.getStates()
        for i in states:
            self.values.update({i : 0})

        num_states = len(states)
        count = 0

        for i in range(self.iterations):
            sums_list = []
            actions = self.mdp.getPossibleActions(states[count])
            for action in actions:
                curr_sum = 0
                for i in self.mdp.getTransitionStatesAndProbs(states[count], action):
                    prob = i[1]
                    next_state = i[0]
                    reward = self.mdp.getReward(states[count], action, next_state)
                    next_state_val = self.values[next_state]
                    curr_sum = curr_sum + prob * (reward + self.discount * next_state_val)
                sums_list.append(curr_sum)
            
            if len(sums_list) == 0:
                best_sum = 0
            else:
                best_sum = max(sums_list)
            self.values.update({states[count] : best_sum})
            count = count + 1

            if(count == num_states):
                count = 0

                    

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"

