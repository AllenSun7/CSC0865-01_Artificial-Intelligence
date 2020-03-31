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
import copy

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
        for _ in range(self.iterations):
            #update values every iteration
            iteration_values = copy.deepcopy(self.values)
            for state in self.mdp.getStates():
                # skip if it is the terminal state
                if self.mdp.isTerminal(state):
                    continue
                # max V(s) = max_{a in actions} Q(s,a)
                q_values = self.q_values(state)
                iteration_values[state] = q_values[q_values.argMax()]
            self.values = iteration_values
        
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
        for next_state, p in self.mdp.getTransitionStatesAndProbs(state, action):
            # Q = sum(T * (R+lambda*V))
            q_value_next_state = p * (self.mdp.getReward(state, action, next_state) + self.discount*self.getValue(next_state))
            q_value += q_value_next_state
        return q_value
        # util.raiseNotDefined()


    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        # Q values with actions 
        q_values = self.q_values(state)
        # return the action with the max value
        if len(q_values) == 0:
            return None
        else :
            # return max(q_values, key=q_values.get)
            return q_values.argMax()
        # util.raiseNotDefined()

    # return all Q-values of a state as a util.Conter
    def q_values(self, state):
        """
        return a dictonary of q_values with corresponding Q-values and actions
        """
        q_values = util.Counter()
        for action in self.mdp.getPossibleActions(state):
            q_values[action] = self.getQValue(state, action)
        return q_values


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

        states = self.mdp.getStates()
        for i in range(self.iterations):
            #update value of certain state 
            #cycle state
            state = states[i % len(states)]
            # skip if it is the terminal state
            if self.mdp.isTerminal(state):
                continue
                # max V(s) = max_{a in actions} Q(s,a)
            q_values = {}
            for action in self.mdp.getPossibleActions(state):
                q_values[action] = self.getQValue(state, action)
            self.values[state] = max(q_values.values())
    

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
        # Initialize an empty priority queue
        p_queue = util.PriorityQueue()
        # Initialize an empty util.Counter to store states and their predecesoors
        p_states = util.Counter()
        for state in self.mdp.getStates():
            if not self.mdp.isTerminal(state):
                p_states[state] = set()
        # copy values for updating the Q-Value of states in p_states
        p_values = copy.deepcopy(self.values)
                
        # Compute predecessors of all states.
        for p_state in p_states.keys():
            for action in self.mdp.getPossibleActions(p_state):
                for s_state, _ in self.mdp.getTransitionStatesAndProbs(p_state, action):
                    # print("next_states")
                    # print(next_states)
                    if not self.mdp.isTerminal(s_state):
                        p_states[s_state].add(p_state)

        def _update_queue(state, theta=-1.0):
            """
            update p_queue
            only update p_queue when diff > theta
            # In ValuesIterationAgent, we define the q_values(self, state)
            # return all Q-values of a state as a util.Conter
            """
            # get highest Q-values
            q_values = self.q_values(state)
            q_value = q_values[q_values.argMax()]
            diff = abs(q_value - self.values[state])
            if diff > theta:
                p_values[state] = q_value
                p_queue.update(state, -diff)

        # update p_queue with states for first time
        for state in p_states.keys():
            _update_queue(state)
         
        # Interation and update state values
        for i in range(self.iterations):
            if p_queue.isEmpty():
                break
            else:
                state = p_queue.pop()
                # update values
                self.values[state] = p_values[state]
                for p_state in p_states[state]:
                    _update_queue(p_state, theta=self.theta)

