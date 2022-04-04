
import sys
import gym_connect4
import numpy as np
import gym
from MCTS_c4 import run_MCTS
from train_c4 import train_connectnet
from argparse import ArgumentParser
from evaluator_c4 import evaluate_nets
from evaluator_c4 import evaluate_pos


env = gym.make('Connect4Env-v0')

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--iteration", type=int, default=0, help="Current iteration number to resume from")
    parser.add_argument("--total_iterations", type=int, default=1000, help="Total number of iterations to run")
    parser.add_argument("--MCTS_num_processes", type=int, default=1, help="Number of processes to run MCTS self-plays")
    parser.add_argument("--num_games_per_MCTS_process", type=int, default=2, help="Number of games to simulate per MCTS self-play process")
    parser.add_argument("--temperature_MCTS", type=float, default=1.1, help="Temperature for first 10 moves of each MCTS self-play")
    parser.add_argument("--num_evaluator_games", type=int, default=3, help="No of games to play to evaluate neural nets")
    parser.add_argument("--neural_net_name", type=str, default="cc4_current_net_", help="Name of neural net")
    parser.add_argument("--batch_size", type=int, default=32, help="Training batch size")
    parser.add_argument("--num_epochs", type=int, default=300, help="No of epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--gradient_acc_steps", type=int, default=1, help="Number of steps of gradient accumulation")
    parser.add_argument("--max_norm", type=float, default=1.0, help="Clipped gradient norm")
    args = parser.parse_args()
    
    #run_MCTS(args, start_idx=0, iteration=5)


    agents = ['Agent1()', 'Agent2()']
    obses = env.reset()  # dict: {0: obs_player_1, 1: obs_player_2}
    game_over = False
    
    while not game_over:
        action_dict = {}
        for agent_id, agent in enumerate(agents):
            if agent_id == 0:
                action = env.action_space.sample()
                if env.game.player == 1:
                    print("player (1), random decision:", action+1)
                
            else:
                winner = evaluate_pos(args, 0, env.game)
                action = np.argmax(winner)
                if env.game.player == 0:
                    print("player (2), MCTS decision:", action+1)
            action_dict[agent_id] = action
        
        obses, rewards, game_over, info = env.step(action_dict)
        env.render()
        #print(env.game.current_board)

        if game_over:
            print("WINNER: ", obses[0]['winner'])




        

 