
import sys
import gym_connect4
import numpy as np
import os
import gym as gym 
import time
import torch
from MCTS_c4 import run_MCTS
from train_c4 import train_connectnet
from argparse import ArgumentParser
import encoder_decoder_c4 as ed
from evaluator_c4 import evaluate_nets
from evaluator_c4 import evaluate_position
from alpha_net_c4 import ConnectNet
from datetime import datetime
env = gym.make('Connect4Env-v0')

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--iteration", type=int, default=0, help="Current iteration number to resume from")
    parser.add_argument("--total_iterations", type=int, default=100, help="Total number of iterations to run")
    parser.add_argument("--MCTS_num_processes", type=int, default=12, help="Number of processes to run MCTS self-plays")
    parser.add_argument("--num_games_per_MCTS_process", type=int, default=120, help="Number of games to simulate per MCTS self-play process")
    parser.add_argument("--temperature_MCTS", type=float, default=1.1, help="Temperature for first 10 moves of each MCTS self-play")
    parser.add_argument("--num_evaluator_games", type=int, default=40, help="No of games to play to evaluate neural nets")
    parser.add_argument("--neural_net_name", type=str, default="cc4_current_net_", help="Name of neural net")
    parser.add_argument("--batch_size", type=int, default=32, help="Training batch size")
    parser.add_argument("--num_epochs", type=int, default=200, help="No of epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--gradient_acc_steps", type=int, default=1, help="Number of steps of gradient accumulation")
    parser.add_argument("--max_norm", type=float, default=1.0, help="Clipped gradient norm")
    parser.add_argument("--print_board", type= bool, default=False, help="show board")
    args = parser.parse_args()


    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("START SELF PLAY: ", current_time)
    #run_MCTS(args, start_idx=0, iteration=5)
    #print(torch.cuda.current_device())
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("FINISHED SELF PLAY: ", current_time)
    #train_connectnet(args, iteration=5, new_optim_state=True)
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("START EVALUATION", current_time)
    evaluate_nets(args, iteration_1=3, iteration_2=5)
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("END EVALUATION", current_time)
    # #evaluate_nets(args, iteration_1=5, iteration_2=1)
    # agents = ['Agent1()', 'Agent2()']
    # obses = env.reset()  # dict: {0: obs_player_1, 1: obs_player_2}
    # game_over = False
    
    #LOAD NEURAL NETWORK
    # current_net="%s_iter%d.pth.tar" % (args.neural_net_name, 1);
    # current_net_filename = os.path.join("",\
    #                                 current_net)
    # current_cnet = ConnectNet()
    
    # current_cnet.share_memory()
    # current_cnet.eval()
        
    # checkpoint = torch.load(current_net_filename)
    # current_cnet.load_state_dict(checkpoint['state_dict'])
    
    #END LOAD NEURAL NETWORK
    
    # while not game_over:
    #     action_dict = {}
    #     for agent_id, agent in enumerate(agents):
    #         if agent_id == 0:
    #             action = env.action_space.sample()
    #             while action not in env.game.get_moves():
    #                 action = env.action_space.sample()
    #             if env.game.player == 1:
    #                 print("player (1), random decision:", action+1)
    #         else:
    #             #winner = evaluate_nets(args, 1, env.game)
    #             winner = evaluate_position(args, env.game, current_cnet)
    #             action = np.argmax(winner)
    #             if env.game.player == 0:
    #                 print("player (2), MCTS decision:", action+1)
    #         action_dict[agent_id] = action
        
    #     obses, rewards, game_over, info = env.step(action_dict)
    #     env.render()
        #print(env.game.current_board)

        # if game_over:
        #     print("WINNER: ", obses[0]['winner'])




        

 
