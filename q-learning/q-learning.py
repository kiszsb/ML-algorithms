import numpy as np
import logging
import copy
import sys
import os

from matplotlib import pyplot as plt
from wakepy import keepawake
from progress.bar import Bar

DIRECTIONS = ['LEFT', 'UP', 'RIGHT', 'DOWN']
AGENT_SYMBOL = 'X'

logging.basicConfig(filename='qlearning.log',
                    filemode='w',
                    format='%(asctime)s - %(message)s',
                    level=logging.INFO)

if not os.path.exists("plots"):
    os.makedirs("plots")

logging.info('Program start')
logging.info('FrozenLake Problem Solved with Q-Learning\n')

class Environment:
    def __init__(self, _map, position_x = 0, position_y = 0):
        self.environment = np.array(copy.deepcopy(_map))
        self.initial_environment = copy.deepcopy(self.environment)
        self.qtable = np.zeros((len(_map) * len(_map[0]), 4))
        self.agent_position_x = position_x
        self.agent_position_y = position_y
        self.initial_agent_position_x = position_x
        self.initial_agent_position_y = position_y
        self.agent_previous_x = position_x
        self.agent_previous_y = position_y
        self.current_tile = _map[position_x][position_y]
        self.initial_tile = _map[position_x][position_y]
        self.environment[position_x][position_y] = AGENT_SYMBOL

    def agent_move(self, direction):
        '''
        Moves the agent to the next state in given direction.
        If the global parameter SLIPPERY is initialised as TRUE,
        then there is a 66% chance of choosing another
        perpendicular direction instead of given one.
        It returns taken direction index, gained score and epoch end flag.
        '''
        self.agent_previous_x = self.agent_position_x
        self.agent_previous_y = self.agent_position_y
        if SLIPPERY:
            if direction == 'LEFT':
                direction = np.random.choice(['LEFT', 'UP', 'DOWN'])
            elif direction == 'UP':
                direction = np.random.choice(['UP', 'LEFT', 'RIGHT'])
            elif direction == 'RIGHT':
                direction = np.random.choice(['RIGHT', 'UP', 'DOWN'])
            elif direction == 'DOWN':
                direction = np.random.choice(['DOWN', 'LEFT', 'RIGHT'])
        if direction == 'LEFT':
            if self.agent_position_y - 1 >= 0:
                self.environment[self.agent_position_x][self.agent_position_y] = self.current_tile
                self.agent_position_y = self.agent_position_y - 1
                self.current_tile = self.environment[self.agent_position_x][self.agent_position_y]
                self.environment[self.agent_position_x][self.agent_position_y] = AGENT_SYMBOL
                if DETAILED_LOGGING:
                    logging.info(f'Agent moved {direction}')
            else:
                if DETAILED_LOGGING:
                    logging.info(f"Agent tried to move {direction}")
        elif direction == 'UP':
            if self.agent_position_x - 1 >= 0:
                self.environment[self.agent_position_x][self.agent_position_y] = self.current_tile
                self.agent_position_x = self.agent_position_x - 1
                self.current_tile = self.environment[self.agent_position_x][self.agent_position_y]
                self.environment[self.agent_position_x][self.agent_position_y] = AGENT_SYMBOL
                if DETAILED_LOGGING:
                    logging.info(f'Agent moved {direction}')
            else:
                if DETAILED_LOGGING:
                    logging.info(f"Agent tried to move {direction}")
        elif direction == 'RIGHT':
            if self.agent_position_y + 1 < len(self.environment[0]):
                self.environment[self.agent_position_x][self.agent_position_y] = self.current_tile
                self.agent_position_y = self.agent_position_y + 1
                self.current_tile = self.environment[self.agent_position_x][self.agent_position_y]
                self.environment[self.agent_position_x][self.agent_position_y] = AGENT_SYMBOL
                if DETAILED_LOGGING:
                    logging.info(f'Agent moved {direction}')
            else:
                if DETAILED_LOGGING:
                    logging.info(f"Agent tried to move {direction}")
        elif direction == 'DOWN':
            if self.agent_position_x + 1 < len(self.environment):
                self.environment[self.agent_position_x][self.agent_position_y] = self.current_tile
                self.agent_position_x = self.agent_position_x + 1
                self.current_tile = self.environment[self.agent_position_x][self.agent_position_y]
                self.environment[self.agent_position_x][self.agent_position_y] = AGENT_SYMBOL
                if DETAILED_LOGGING:
                    logging.info(f'Agent moved {direction}')
            else:
                if DETAILED_LOGGING:
                    logging.info(f"Agent tried to move {direction}")
        if self.current_tile == 'G':
            finish = True
            if DETAILED_LOGGING:
                logging.info("Agent WON")
        elif self.current_tile == 'H':
            finish = True
            if DETAILED_LOGGING:
                logging.info("Agent LOST")
        else:
            finish = False
        return DIRECTIONS.index(direction), self.agent_evaluate(), finish

    def agent_evaluate(self):
        '''
        It checks for the state symbol underneath the agent.
        It returns 1 if it's the final state resulting in a win.
        It returns 0 otherwise.
        '''
        #if self.current_tile == 'G':
        #    return 1
        #else:
        #    return 0
        if self.current_tile == 'G':
            return 1
        if self.current_tile == 'H':
            return -1
        else:
            return 0

    def qtable_update(self, action, reward, learning_rate, discount_factor):
        '''
        Updates the Q-Table used in Q-Learning algorithm.
        Uses default Q-Table update formula to further adjust it's values.
        It takes most recent action, reward value, learning rate and discount factor.
        It doesn't return anything.
        '''
        previous_state = self.agent_previous_x * len(self.environment[0]) + self.agent_previous_y
        current_state = self.agent_position_x * len(self.environment[0]) + self.agent_position_y
        self.qtable[previous_state][action] += learning_rate * (reward + discount_factor * np.max(self.qtable[current_state]) - self.qtable[previous_state][action])

    def reset(self):
        '''
        Completely resets the environment to the state of first initialisation,
        excluding Q-Table values. It uses deepcopy to create a unique copy of saved initial
        environment, preventing accidentaly updating it.
        It doesn't return anything.
        '''
        self.agent_position_x = self.initial_agent_position_x
        self.agent_position_y = self.initial_agent_position_y
        self.agent_previous_x = self.agent_position_x
        self.agent_previous_y = self.agent_position_y
        self.environment = copy.deepcopy(self.initial_environment)
        self.current_tile = self.initial_tile
        self.environment[self.agent_position_x][self.agent_position_y] = AGENT_SYMBOL
            
    def render(self, epoch = 'N/S', step = 'N/S'):
        '''
        Saves the current environment visualisation to the file.
        It doesn't return anything.
        '''
        logging.info(f'Epoch {epoch}, step {step}; Environment:\n{self.environment}')

    def qtable_show(self, epoch = 'N/S'):
        '''
        Saves the current Q-Table visualisation to the file.
        It doesn't return anything.
        '''
        logging.info(f'Epoch {epoch}; Q-Table:\n{self.qtable}')

    def train(self, epochs, max_steps, learning_rate, discount_factor):
        '''
        Training part of Q-Learning algorithm.
        It creates an empty list that collects results,
        which are equal to loss or win of the agent at the end of particular epoch.
        It uses exponential Epsilon-Greedy method to control the rate of exploration of the environment.
        It returns the rewards list.
        '''
        epsilon = 1.0
        max_epsilon = 1.0
        min_epsilon = 0.001
        decay_rate = 0.00005
        rewards = []
        for epoch in range(epochs):
            self.reset()
            finish = False
            total_rewards = 0
            for step in range(max_steps):
                exp_exp_tradeoff = np.random.uniform(0, 1)
                if exp_exp_tradeoff > epsilon:
                    current_state = self.agent_position_x * len(self.environment[0]) + self.agent_position_y
                    action = np.argmax(self.qtable[current_state])
                else:
                    action = np.random.choice(range(4))
                action, reward, finish = self.agent_move(DIRECTIONS[action])
                if DETAILED_LOGGING:
                    self.render(epoch, step)
                self.qtable_update(action, reward, learning_rate, discount_factor)
                if(reward < 0):
                    reward = 0
                total_rewards += reward
                if finish == True:
                    if DETAILED_LOGGING:
                        self.qtable_show(epoch)
                        logging.info(f"Steps for this epoch: {step}\n\n")
                    break
            epsilon = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay_rate*epoch)
            rewards.append(total_rewards)
            BAR.next()
        logging.info(f'Final training score: {(sum(rewards)/epochs) * 100}%')
        logging.info(f'Final Q-Table:\n{self.qtable}')
        return rewards

        
    def test(self, epochs, max_steps):
        '''
        Testing part of Q-Learning algorithm.
        It creates an empty list that collects results,
        which are equal to loss or win of the agent at the end of particular epoch.
        It uses the trained Q-Table to reach the goal in the environment.
        It returns the rewards list.
        '''
        rewards = []
        for _ in range(epochs):
            self.reset()
            finish = False
            total_rewards = 0
            for _ in range(max_steps):
                current_state = self.agent_position_x * len(self.environment[0]) + self.agent_position_y
                action = np.argmax(self.qtable[current_state])
                action, reward, finish = self.agent_move(DIRECTIONS[action])
                if(reward < 0):
                    reward = 0
                total_rewards += reward
                if finish == True:
                    break
            rewards.append(total_rewards)
            BAR.next()
        logging.info(f'Final testing score: {(sum(rewards)/epochs) * 100}%')
        return rewards


if __name__ == '__main__':

    number_of_args = len(sys.argv)
    if number_of_args == 1:
        SLIPPERY = False
        DETAILED_LOGGING = True
        map_type = '4x4'
        epochs = 2000
        testing_epochs = 100
        max_steps = 200
        learning_rate = 0.5
        discount_factor = 0.9
    elif number_of_args == 9:
        SLIPPERY = bool(sys.argv[1])
        DETAILED_LOGGING = bool(sys.argv[2])
        map_type = sys.argv[3]
        epochs = int(sys.argv[4])
        testing_epochs = int(sys.argv[5])
        max_steps = int(sys.argv[6])
        learning_rate = float(sys.argv[7])
        discount_factor = float(sys.argv[8])
    else:
        print("Error: Bad args, see the log file")
        logging.error("Bad arguments were provided while trying to start the program")
        logging.error("Correct args: SLIPPERY, DETAILED_LOGGING, MAP_TYPE, EPOCHS, TESTING_EPOCHS, MAX_STEPS, LEARNING_RATE, DISCOUNT_FACTOR or none")
        sys.exit()

    frozen_lake_map4x4 = [['S', 'F', 'F', 'F'],
                          ['F', 'H', 'F', 'H'],
                          ['F', 'F', 'F', 'H'],
                          ['H', 'F', 'F', 'G']]

    frozen_lake_map8x8 = [['S', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
                       ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
                       ['F', 'F', 'F', 'H', 'F', 'F', 'F', 'F'],
                       ['F', 'F', 'F', 'F', 'F', 'H', 'F', 'F'],
                       ['F', 'F', 'F', 'H', 'F', 'F', 'F', 'F'],
                       ['F', 'H', 'H', 'F', 'F', 'F', 'H', 'F'],
                       ['F', 'H', 'F', 'F', 'H', 'F', 'H', 'F'],
                       ['F', 'F', 'F', 'H', 'F', 'F', 'F', 'G']]

    if map_type == '4x4':
        chosen_map = frozen_lake_map4x4
    elif map_type == '8x8':
        chosen_map = frozen_lake_map8x8
    else:
        print("Error: Bad map_name, see the log file")
        logging.error("Bad map name was provided, accepted values: 4x4 or 8x8")
        exit(0)

    logging.info(f'Parameters: SLIPPERY = {SLIPPERY}, DETAILED_LOGGING = {DETAILED_LOGGING}')
    logging.info(f'MAP_TYPE = {map_type}, EPOCHS = {epochs}, TESTING_EPOCHS = {testing_epochs}, MAX_STEPS = {max_steps}')
    logging.info(f'LEARNING_RATE = {learning_rate}, DISCOUNT_FACTOR = {discount_factor}\n')

    lake_environment = Environment(chosen_map)

    with keepawake(keep_screen_awake=True):
        BAR = Bar('Training Q-Table', max = epochs, check_tty=False, suffix ='%(percent).1f%% - %(eta)ds')
        training_results = lake_environment.train(epochs, max_steps, learning_rate, discount_factor)
        BAR.finish()
        BAR = Bar('Testing Q-Table', max = testing_epochs, check_tty=False, suffix ='%(percent).1f%% - %(eta)ds')
        testing_results = lake_environment.test(testing_epochs, max_steps)
        BAR.finish()

    plt.figure()
    y_training_results = []
    for result_pos in range(len(training_results)):
       y_training_results.append(sum(training_results[0:result_pos]))
    plt.plot(range(epochs), y_training_results)
    plt.title('Goals Reached Graph - Training')
    plt.savefig('plots/goals_training.png')

    plt.figure()
    y_testing_results = []
    for result_pos in range(len(testing_results)):
       y_testing_results.append(sum(testing_results[0:result_pos]))
    plt.plot(range(testing_epochs), y_testing_results)
    plt.title('Goals Reached Graph - Testing')
    plt.savefig('plots/goals_testing.png')

    logging.info('Program end')
    print('\a')