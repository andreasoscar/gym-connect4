
from calendar import c
import sys
from tabnanny import check
import gym_connect4
import numpy as np
import os
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OMP_NUM_THREADS"] = "1" 
import gym as gym 
import time
import multiprocessing as mp
import random
import torch
from minimax import minimax
from MCTS_c4 import run_MCTS
from train_c4 import train_connectnet
from argparse import ArgumentParser
import encoder_decoder_c4 as ed
from evaluator_c4 import evaluate_nets
from evaluator_c4 import evaluate_position
from alpha_net_c4 import ConnectNet
from alpha_net_c4_2 import ConnectNet1
from datetime import datetime
env = gym.make('Connect4Env-v0')
def foo(q):
    q.put('hello')
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--iteration", type=int, default=0, help="Current iteration number to resume from")
    parser.add_argument("--total_iterations", type=int, default=100, help="Total number of iterations to run")
    parser.add_argument("--MCTS_num_processes", type=int, default=8, help="Number of processes to run MCTS self-plays")
    parser.add_argument("--num_games_per_MCTS_process", type=int, default=5, help="Number of games to simulate per MCTS self-play process")
    parser.add_argument("--temperature_MCTS", type=float, default=1.1, help="Temperature for first 10 moves of each MCTS self-play")
    parser.add_argument("--num_evaluator_games", type=int, default=24, help="No of games to play to evaluate neural nets")
    parser.add_argument("--neural_net_name", type=str, default="cc4_current_net_", help="Name of neural net")
    parser.add_argument("--batch_size", type=int, default=32, help="Training batch size")
    parser.add_argument("--num_epochs", type=int, default=120, help="No of epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--gradient_acc_steps", type=int, default=1, help="Number of steps of gradient accumulation")
    parser.add_argument("--max_norm", type=float, default=1.0, help="Clipped gradient norm")
    parser.add_argument("--print_board", type= bool, default=False, help="show board")
    parser.add_argument("--minimax_depth", type=int, default=1, help="set minimax depth")
    args = parser.parse_args()




    def play_net_vs_net(it1,it2):
        cuda = torch.cuda.is_available()
        #LOAD NEURAL NETWORK
        
        current_net="%s_iter%d.pth.tar" % (args.neural_net_name, it1);
        current_net_filename = os.path.join("",\
                                        current_net)
        current_cnet = ConnectNet1()
        
        current_net_1 ="%s_iter%d.pth.tar" % (args.neural_net_name, it2);
        current_net_filename_1 = os.path.join("",\
                                        current_net_1)
        current_cnet_1 = ConnectNet()
        
        
        current_cnet.share_memory()
        current_cnet.eval()
        
        current_cnet_1.share_memory()
        current_cnet_1.eval()
        if not cuda:
            checkpoint = torch.load(current_net_filename, map_location=torch.device('cpu'))
        else:
            checkpoint = torch.load(current_net_filename)
        #print(checkpoint['state_dict'].keys)
        #print(checkpoint)
        current_cnet.load_state_dict(checkpoint['state_dict'])
        
        if not cuda:
            
            checkpoint_1 = torch.load(current_net_filename_1, map_location=torch.device('cpu'))
        else:
            checkpoint_1 = torch.load(current_net_filename_1)
        #print(checkpoint['state_dict'].keys)
        current_cnet_1.load_state_dict(checkpoint_1['state_dict'])
        
        
        
        #END LOAD NEURAL NETWORK
    
        agents = ['Agent1()', 'Agent2()']
        obses = env.reset()  # dict: {0: obs_player_1, 1: obs_player_2}
        game_over = False
        
        games_completed = 0
        #depth = 2
        winners = [0,0]
        p1 = 0
        p2 = 0
        while games_completed < args.num_evaluator_games:
            v = random.uniform(0,1)
            print(v)
            if v > 0.5:
                t = 0
            else:
                t = 1
            starts.append(t+1)
            if t == 0:
                print("network1 start")
                p1 += 1
            else:
                print("network2 start")
                p2 += 1
            while not game_over:
                    action_dict = {}
                    #print("CURRENT PLAYER: ", env.game.player)
                    action = env.action_space.sample()
                    #print(env.game.get_moves())
                    action = random.choice(env.game.get_moves())
                    #print(env.game.get_moves(), action)
                    
                    a = [0,1,2,3,4,5,6]
                    policy = evaluate_position(args, env.game, current_cnet)
                    policy_1 = evaluate_position(args, env.game, current_cnet_1)
                    if t == 0:
                        action_dict[0] = np.random.choice(a, p=policy)
                    else:
                        action_dict[1] = np.random.choice(a,p=policy_1)
                    #print("bot: ", policy_1+1)
                    
                    #print(policy, np.argmax(policy))
                    if t == 1:
                        action_dict[0] = np.random.choice(a,p=policy_1)
                    else:
                        action_dict[1] =np.random.choice(a,p=policy)
                    #print("player (2), MCTS decision:", np.argmax(policy)+1)
                    obses, rewards, game_over, info = env.step(action_dict)
                    env.render()
                    #print(env.game.current_board)
                    
            if game_over:
                winner = obses[0]['winner']
                winners[winner] = winners[winner] + 1
                games_completed += 1
                if games_completed == args.num_evaluator_games:
                    f = open("random_log/log.txt", "a")
                    now = datetime.now()
                    current_time = now.strftime("%H:%M:%S")
                    f.write(current_time + ", win ratio: " + str(winners[1]/args.num_evaluator_games) + " vs random after " + str(iteration*args.num_games_per_MCTS_process) + " games "+"\n")
                    f.close()
                    game_over = True
                else:
                    game_over = False
                    env.reset()
                if t == 0 and winner == 0:
                    winList.append(1)
                    print("iteration " + str(it1) + " wins")
                    wins[0]  = wins[0] + 1
                elif t == 1 and winner == 0:
                    winList.append(2)
                    print("iteration " + str(it2) + " wins")
                    wins[1]  = wins[1] + 1
                elif t == 0 and winner == 1:
                    winList.append(2)
                    print("iteration " + str(it2) + " wins")
                    wins[1]  = wins[1] + 1
                elif t == 1 and winner == 0:
                    winList.append(1)
                    print("iteration " + str(it1) + " wins")
                    wins[0]  = wins[0] + 1
                break

    
    
    def evaluate_minimax_MCTS():
        cuda = torch.cuda.is_available()
        #LOAD NEURAL NETWORK
        
        current_net="%s_iter%d.pth.tar" % (args.neural_net_name, 0);
        current_net_filename = os.path.join("",\
                                        current_net)
        current_cnet = ConnectNet()
        
        current_cnet.share_memory()
        current_cnet.eval()
        if not cuda:
            checkpoint = torch.load(current_net_filename, map_location=torch.device('cpu'))
        else:
            checkpoint = torch.load(current_net_filename)
        current_cnet.load_state_dict(checkpoint['state_dict'])
        
        #END LOAD NEURAL NETWORK
    
        agents = ['Agent1()', 'Agent2()']
        obses = env.reset()  # dict: {0: obs_player_1, 1: obs_player_2}
        game_over = False
        
        games_completed = 0
        #depth = 2
        winners = [0,0]
        
        while games_completed < 3:
            print("GAME: ", games_completed)
            t = minimax(args.minimax_depth)
            while not game_over:
                action_dict = {}
            
                action = env.action_space.sample()
                #col = t.student_move(env.game.get_current_board())
            
                action = random.choice(env.game.get_moves())
                action_dict[0] = action
                print("bot: ", action+1)
                policy = evaluate_position(args, env.game, current_cnet)
                print(policy)
                action_dict[1] = np.argmax(policy)
                print("player (2), MCTS decision:", np.argmax(policy)+1)

                obses, rewards, game_over, info = env.step(action_dict)
                env.render()
                print(env.game.current_board)

                if game_over:
                    winner = obses[0]['winner']
                    winners[winner] = winners[winner] + 1
                    games_completed += 1
                    if games_completed == 3:
                        f = open("minimax_log/log.txt", "a")
                        now = datetime.now()
                        current_time = now.strftime("%H:%M:%S")
                        f.write(current_time + ", win ratio: " + str(winners[1]/args.num_evaluator_games) + " vs depth: " + str(args.minimax_depth) + "\n")
                        f.close()
                        game_over = True
                    else:
                        game_over = False
                        env.reset()
                    
                    print("WINNER: ", obses[0]['winner'])
                    break

    #evaluate_minimax_MCTS()


    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("START SELF PLAY: ", current_time)
    #test_random(0)
    # for i in range(2):
    #    run_MCTS(args, start_idx=i*args.num_games_per_MCTS_process, iteration=18)
    #    train_connectnet(args, iteration=18, new_optim_state=True)
    #    test_random(i)
    #test_random(0)
    #print(torch.cuda.current_device())
    now = datetime.now()
    #run_MCTS(args, start_idx=0, iteration=1)
    #test_random(0)
    current_time = now.strftime("%H:%M:%S")
    print("FINISHED SELF PLAY: ", current_time)
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("START EVALUATION", current_time)
    #evaluate_nets(args, iteration_1=3, iteration_2=5)
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("END EVALUATION", current_time)
    wins = [0,0]
    starts = []
    winList = []
    p1 = 0
    p2 = 0
    for i in range(20):
        play_net_vs_net(2, 7)
        print("network indices")
        print(wins, winList, starts)
    print("network indices")
    print(wins, winList, starts)


    
    
    #mp.set_start_method('spawn')
    #q = mp.Queue()
    #p = mp.Process(target=foo, args=(q,))
    ##p.start()
    #print(q.get())
    #p.join()