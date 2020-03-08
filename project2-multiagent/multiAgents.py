# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
import pysnooper
import numpy as np

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        new_food_list = newFood.asList()
        newGhostStates = successorGameState.getGhostStates()
        new_ghost_pos = successorGameState.getGhostPositions()
        num_ghosts = successorGameState.getNumAgents()-1
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        "*** YOUR CODE HERE ***"
        
        #heuristic distance to food 
        dis_food = [((newPos[0]-f[0])**2 + (newPos[1]-f[1])**2)**0.5 for f in new_food_list]
        #if food is not empty
        if dis_food:
            food_nearest_idx = dis_food.index(min(dis_food))
            pos_food_nearest = new_food_list[food_nearest_idx]
            dis_food_nearest = dis_food[food_nearest_idx]
        else: 
            dis_food_nearest = 1
        #heuristic distance to ghost
        dis_ghost = [((newPos[0]-g[0])**2 + (newPos[1]-g[1])**2)**0.5 for g in new_ghost_pos]
        ghost_nearest_idx = dis_ghost.index(min(dis_ghost))
        pos_ghost_nearest = new_ghost_pos[ghost_nearest_idx]
        dis_ghost_nearest = dis_ghost[ghost_nearest_idx]

        # print("newFood: ")
        # print(str(newFood))
        # print(new_food_list)
        # print("newPos: ")
        # print(str(newPos))
        # print("num_ghosts: ")
        # print(str(num_ghosts))       
        # print("new_ghost_pos: ")
        # print(str(new_ghost_pos))  
        # print("successorGameState.getScore():", successorGameState.getScore())     
        # print("dis_food_nearest:", dis_food_nearest)     
        # print("dis_ghost_nearest:", dis_ghost_nearest) 

        # distance to nearest food: ther closer, the better
        score_dis_food = 1 / dis_food_nearest if dis_food_nearest != 0 else 0
        # distance to ghost: ther further, the better
        score_dis_ghost = - 1 / dis_ghost_nearest if dis_ghost_nearest != 0 else 0
        # pacman successor score 
        score = successorGameState.getScore() + score_dis_food + score_dis_ghost
        return score
          
        


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """
    #@pysnooper.snoop()
    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        # if trigger score evaluate funtion
        def evalute_score(state, depth):
            return state.isWin() or state.isLose() or depth == self.depth

        def min_value(state, depth, g_idx):
            if evalute_score(state, depth):
                return self.evaluationFunction(state)

            v = 99999999999999
            for action in state.getLegalActions(g_idx):
                if g_idx == num_ghosts:
                    v = min(v, max_value(state.generateSuccessor(g_idx, action), depth+1))
                else:
                    v = min(v, min_value(state.generateSuccessor(g_idx, action), depth, g_idx+1))
            #print("min value: ", v)
            return v
        
        def max_value(state, depth):
            if evalute_score(state, depth):
                return self.evaluationFunction(state)

            v = -99999999999999
            for action in state.getLegalActions(0):
                v = max(v, min_value(state.generateSuccessor(0, action), depth, 1))
            #print("max value: ", v)
            return v

        num_ghosts = gameState.getNumAgents()-1
        actions = gameState.getLegalActions(0)
        scores = [min_value(gameState.generateSuccessor(0, action), 0, 1) for action in actions]
        index_best = scores.index(max(scores))
        return actions[index_best]
        #util.raiseNotDefined()


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def min_value(state, depth, g_idx, alpha, beta):
            # if trigger score evaluate funtion
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)

            v = 99999999999999
            for action in state.getLegalActions(g_idx):
                if g_idx == num_ghosts:
                    v = min(v, max_value(state.generateSuccessor(g_idx, action), depth+1, alpha, beta)[0])
                else:
                    v = min(v, min_value(state.generateSuccessor(g_idx, action), depth, g_idx+1, alpha, beta))
                #prune 
                if v < alpha:
                    return v
                beta = min(beta, v)
            #print("min value: ", v)
            return v
        
        def max_value(state, depth, alpha, beta):
            best_action = None
            # if trigger score evaluate funtion
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state), best_action
            v = -99999999999999
            for action in state.getLegalActions(0):
                v = max(v, min_value(state.generateSuccessor(0, action), depth, 1, alpha, beta))
                #prune 
                if v > beta:
                    return v, best_action
                #get the alpha and best action
                if v > alpha:
                    alpha = v
                    best_action = action
            #print("max value: ", v)
            return v, best_action

        alpha = -99999999999999
        beta = 99999999999999
        num_ghosts = gameState.getNumAgents()-1
        #get the action with the max score
        _, best_action = max_value(gameState, 0, alpha, beta)

        return best_action
        #util.raiseNotDefined()


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"

        def chance(state, depth, g_idx):
            # if trigger score evaluate funtion
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)
            
            #initial value, probability 
            v = 0
            actions = state.getLegalActions(g_idx)
            p = 1 / len(actions)

            for action in actions:
                if g_idx == num_ghosts:
                    v += p * max_value(state.generateSuccessor(g_idx, action), depth+1)[0]
                else:
                    v += p * chance(state.generateSuccessor(g_idx, action), depth, g_idx+1)

            #print("mean value: ", v)
            return v          

        def max_value(state, depth):
            best_action = None
            # if trigger score evaluate funtion
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state), best_action
            v = -99999999999999
            for action in state.getLegalActions(0):
                v_new = chance(state.generateSuccessor(0, action), depth, 1)
                #return the best action
                if v < v_new:
                    v = v_new
                    best_action = action
            #print("max value: ", v)
            return v, best_action

        #number of ghosts
        num_ghosts = gameState.getNumAgents()-1
        #get the action with the max score
        _, best_action = max_value(gameState, 0)

        return best_action
        #util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    get the heuristic distance of nearest food and ghost
    if in scare time, score of distance to ghost = -20, 
    I guess that ghost = -20 is best because it is the half of capsule moves(40)  

    """
    "*** YOUR CODE HERE ***"
    
    # Useful information you can extract from a GameState (pacman.py)
    current_pos = currentGameState.getPacmanPosition()
    current_food = currentGameState.getFood()
    current_food_list = current_food.asList()
    current_ghost_states = currentGameState.getGhostStates()
    current_ghost_pos = currentGameState.getGhostPositions()
    num_ghosts = currentGameState.getNumAgents()-1
    current_scared_times = [ghostState.scaredTimer for ghostState in current_ghost_states]
    
    #heuristic distance to food 
    dis_food = [((current_pos[0]-f[0])**2 + (current_pos[1]-f[1])**2)**0.5 for f in current_food_list]
    #if food is not empty
    if dis_food:
        food_nearest_idx = dis_food.index(min(dis_food))
        pos_food_nearest = current_food_list[food_nearest_idx]
        dis_food_nearest = dis_food[food_nearest_idx]
    else: 
        dis_food_nearest = 1
    #heuristic distance to ghost
    dis_ghost = [((current_pos[0]-g[0])**2 + (current_pos[1]-g[1])**2)**0.5 for g in current_ghost_pos]
    ghost_nearest_idx = dis_ghost.index(min(dis_ghost))
    pos_ghost_nearest = current_ghost_pos[ghost_nearest_idx]
    dis_ghost_nearest = dis_ghost[ghost_nearest_idx]


    for i in range(len(current_scared_times)):
        if current_scared_times[i] != 0:
            #print("scare time at ghost: {}".format(i+1))
            #print("time: {}".format(current_scared_times))
            #if in scare time, score of distance to ghost = -20 (half of capsule moves)
            score_dis_ghost = 20
        else:
            # distance to ghost: ther further, the better
            score_dis_ghost = - 1 / dis_ghost_nearest if dis_ghost_nearest != 0 else 0
            
    # distance to nearest food: ther closer, the better
    score_dis_food = 1 / dis_food_nearest if dis_food_nearest != 0 else 0
    
    # pacman overall score 
    score = currentGameState.getScore() + score_dis_food + score_dis_ghost

    return score
    
    #util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
