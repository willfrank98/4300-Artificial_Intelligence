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
from game import Directions, Actions
import random, util, Queue

from game import Agent

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
        some Directions.X for some X in the set {North, South, West, East, Stop}
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
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"

        foodList = currentGameState.getFood().asList()
        if len(foodList) == 0:
          return 1 # makes sure pacman eats all his food

        foodHeuristic = []
        for food in foodList:
            # gets the manhattan distance of each item of food
            foodHeuristic.append((manhattanDistance(newPos, food) + 1))

        ghostHeuristic = []
        for ghost in newGhostStates:
          ghostHeuristic.append(manhattanDistance(newPos, ghost.getPosition()) + 1)

        adjustedFood = 1/float(min(foodHeuristic)) * 0.95
        adjustedGhost = 1/float(min(ghostHeuristic)) * 1.0

        return adjustedFood - adjustedGhost

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
        """
        "*** YOUR CODE HERE ***"
        
        return minimax(gameState, self.depth, self.evaluationFunction)[0]

def minimax(gameState, maxDepth, evalFunc):
  return evaluate_state(gameState, 0, maxDepth, 0, evalFunc)

def evaluate_state(gameState, agentId, maxDepth, depth, evalFunc):
  # increments the depth before the final agent makes their move
  # makes sure eval function is used in the right place
  if agentId == gameState.getNumAgents() - 1:
    depth += 1
  agentId %= gameState.getNumAgents()
  if agentId == 0:
    return max_action(gameState, agentId, maxDepth, depth, evalFunc)
  else:
    return min_action(gameState, agentId, maxDepth, depth, evalFunc)

def max_action(gameState, agentId, maxDepth, depth, evalFunc):
  maxAction = None
  maxValue = -float("inf")

  actions = gameState.getLegalActions(agentId)
  if len(actions) == 0:
    # if there are no moves evaluate here
    return None, evalFunc(gameState)

  for action in actions:
    val = evaluate_state(gameState.generateSuccessor(agentId, action), agentId + 1, maxDepth, depth, evalFunc)[1]
    if val > maxValue:
      maxValue = val
      maxAction = action
  
  return maxAction, maxValue

def min_action(gameState, agentId, maxDepth, depth, evalFunc):
  minAction = None
  minValue = float("inf")

  actions = gameState.getLegalActions(agentId)
  if len(actions) == 0:
    return None, evalFunc(gameState)

  for action in actions:
    if depth < maxDepth:
      val = evaluate_state(gameState.generateSuccessor(agentId, action), agentId + 1, maxDepth, depth, evalFunc)[1]
    else:
      val = evalFunc(gameState.generateSuccessor(agentId, action))
    if val < minValue:
      minValue = val
      minAction = action
  
  return minAction, minValue

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        return alphabeta(gameState, self.depth, self.evaluationFunction)[0]

def alphabeta(gameState, maxDepth, evalFunc):
  return evaluate_state_ab(gameState, 0, maxDepth, 0, evalFunc, -float("inf"), float("inf"))

def evaluate_state_ab(gameState, agentId, maxDepth, depth, evalFunc, a, b):
  if agentId == gameState.getNumAgents() - 1:
    depth += 1
  agentId %= gameState.getNumAgents()
  if agentId == 0:
    return max_action_ab(gameState, agentId, maxDepth, depth, evalFunc, a, b)
  else:
    return min_action_ab(gameState, agentId, maxDepth, depth, evalFunc, a, b)

def max_action_ab(gameState, agentId, maxDepth, depth, evalFunc, a, b):
  maxAction = None
  maxValue = -float("inf")

  actions = gameState.getLegalActions(agentId)
  if len(actions) == 0:
    return None, evalFunc(gameState)

  for action in actions:
    val = evaluate_state_ab(gameState.generateSuccessor(agentId, action), agentId + 1, maxDepth, depth, evalFunc, a, b)[1]
    if val > maxValue:
      maxValue = val
      maxAction = action
    if maxValue > b:
      return maxAction, maxValue
    a = max(a, maxValue)
  
  return maxAction, maxValue

def min_action_ab(gameState, agentId, maxDepth, depth, evalFunc, a, b):
  minAction = None
  minValue = float("inf")

  actions = gameState.getLegalActions(agentId)
  if len(actions) == 0:
    return None, evalFunc(gameState)

  for action in actions:
    if depth < maxDepth:
      val = evaluate_state_ab(gameState.generateSuccessor(agentId, action), agentId + 1, maxDepth, depth, evalFunc, a, b)[1]
    else:
      val = evalFunc(gameState.generateSuccessor(agentId, action))
    if val < minValue:
      minValue = val
      minAction = action
    if minValue < a:
      return minAction, minValue
    b = min(b, minValue)

  
  return minAction, minValue

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
        return expectimax(gameState, self.depth, self.evaluationFunction)[0]

def expectimax(gameState, maxDepth, evalFunc):
  return evaluate_state_em(gameState, 0, maxDepth, 0, evalFunc)

def evaluate_state_em(gameState, agentId, maxDepth, depth, evalFunc):
  if agentId == gameState.getNumAgents() - 1:
    depth += 1
  agentId %= gameState.getNumAgents()
  if agentId == 0:
    return max_action_em(gameState, agentId, maxDepth, depth, evalFunc)
  else:
    return expect_action(gameState, agentId, maxDepth, depth, evalFunc)

def max_action_em(gameState, agentId, maxDepth, depth, evalFunc):
  maxAction = None
  maxValue = -float("inf")

  actions = gameState.getLegalActions(agentId)
  if len(actions) == 0:
    return None, evalFunc(gameState)

  for action in actions:
    val = evaluate_state_em(gameState.generateSuccessor(agentId, action), agentId + 1, maxDepth, depth, evalFunc)[1]
    if val > maxValue:
      maxValue = val
      maxAction = action
  
  return maxAction, maxValue

def expect_action(gameState, agentId, maxDepth, depth, evalFunc):
  total = 0.0

  actions = gameState.getLegalActions(agentId)
  if len(actions) == 0:
    return None, evalFunc(gameState)

  for action in actions:
    if depth < maxDepth:
      val = evaluate_state_em(gameState.generateSuccessor(agentId, action), agentId + 1, maxDepth, depth, evalFunc)[1]
    else:
      val = evalFunc(gameState.generateSuccessor(agentId, action))
    total += val
  
  return None, total/len(actions)

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <Combines number of food items left, maze distance to the nearest food item, 
      number of capsule items left, and the score into one ultimate heuristic>
    """

    "*** YOUR CODE HERE ***"
    foodList = currentGameState.getFood().asList()

    # gets distance to nearest ghost
    ghostHeuristic = []
    for ghost in currentGameState.getGhostStates():
      ghostHeuristic.append(manhattanDistance(currentGameState.getPacmanPosition(), ghost.getPosition()) + 1)
    ghostHeuristic = min(ghostHeuristic)
    # makes sure pacman never gets too close to a ghost
    if ghostHeuristic <= 3:
      return -2000

    # the number of pellets/capsules left
    pelletList = currentGameState.getCapsules()
    numPellets = len(pelletList)
    pelletHeuristic = numPellets * -100

    adjustedFood = -len(foodList)

    if adjustedFood == 0:
      return currentGameState.getScore() # winner!

    # finds closest food via manhattan
    target = None
    targetDis = 10000000
    for food in foodList:
      dis = manhattanDistance(currentGameState.getPacmanPosition(), food)
      if dis < targetDis:
        target = food
        targetDis = dis
    
    # maze distance of nearest food
    adjustedDist = (1/float(MazeDistance(currentGameState, target))) * 0.9

    adjustedScore = currentGameState.getScore()/100

    total = adjustedFood + adjustedDist + adjustedScore + pelletHeuristic

    return total

def MazeDistance(gameState, target):
  """Gets the maze distance between Pacman and target, using maze information from gameState"""

  currentPos = gameState.getPacmanPosition()
  for action in gameState.getLegalActions():
    x, y = Actions.directionToVector(action)
    newPos = (currentPos[0] + x, currentPos[1] + y)
    if newPos == target:
      return 1

  queue = Queue.PriorityQueue()

  dis = manhattanDistance(target, gameState.getPacmanPosition())
  queue.put((dis, (gameState, 1, "Stop")))

  while not queue.empty():
    state, depth, prevAct = queue.get()[1]
    if depth > 50:
      break
    actions = state.getLegalActions()
    if "Stop" in actions: 
      actions.remove("Stop")
    rev = Actions.reverseDirection(prevAct)
    if rev in actions:
      actions.remove(rev)
    for act in actions:
      newState = state.generatePacmanSuccessor(act)
      newPos = newState.getPacmanPosition()
      if newPos == target:
        return depth
      dis = manhattanDistance(target, newPos)
      queue.put((dis, (newState, depth + 1, act)))

  return 1000


# Abbreviation
better = betterEvaluationFunction