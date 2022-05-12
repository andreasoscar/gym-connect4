
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
import multiprocess as mp
import random
import torch
from minimax import minimax
from MCTS_c4 import run_MCTS
from train_c4 import train_connectnet
import logging
from argparse import ArgumentParser
import encoder_decoder_c4 as ed
from evaluator_c4 import evaluate_nets
from evaluator_c4 import evaluate_position
from alpha_net_c4 import ConnectNet
from datetime import datetime
logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', \
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger(__file__)
env = gym.make('Connect4Env-v0')
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--iteration", type=int, default=1, help="Current iteration number to resume from")
    parser.add_argument("--total_iterations", type=int, default=4, help="Total number of iterations to run")
    parser.add_argument("--MCTS_num_processes", type=int, default=2, help="Number of processes to run MCTS self-plays")
    parser.add_argument("--num_games_per_MCTS_process", type=int, default=1, help="Number of games to simulate per MCTS self-play process")
    parser.add_argument("--temperature_MCTS", type=float, default=1.1, help="Temperature for first 10 moves of each MCTS self-play")
    parser.add_argument("--num_evaluator_games", type=int, default=2, help="No of games to play to evaluate neural nets")
    parser.add_argument("--neural_net_name", type=str, default="cc4_current_net_", help="Name of neural net")
    parser.add_argument("--batch_size", type=int, default=32, help="Training batch size")
    parser.add_argument("--num_epochs", type=int, default=10, help="No of epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--gradient_acc_steps", type=int, default=1, help="Number of steps of gradient accumulation")
    parser.add_argument("--max_norm", type=float, default=1.0, help="Clipped gradient norm")
    parser.add_argument("--print_board", type= bool, default=False, help="show board")
    parser.add_argument("--minimax_depth", type=int, default=1, help="set minimax depth")
    args = parser.parse_args()
    


    #wins = [0,0]
    def play_net_vs_net(args, iteration, it1,it2, cpu):
        cuda = torch.cuda.is_available()
        #LOAD NEURAL NETWORK
        wins = [0,0]  
        current_net="%s_iter%d.pth.tar" % (args.neural_net_name, it1);
        current_net_filename = os.path.join("",\
                                        current_net)
        current_cnet = ConnectNet()
        
        current_net_1 ="%s_iter%d.pth.tar" % (args.neural_net_name, it2);
        current_net_filename_1 = os.path.join("",\
                                        current_net_1)
        current_cnet_1 = ConnectNet()
        
        
        current_cnet.share_memory()
        current_cnet.eval()
        
        current_cnet_1.share_memory()
        current_cnet_1.eval()
        #if not cuda:
        #    checkpoint = torch.load(current_net_filename, map_location=torch.device('cpu'))
        #else:
        #    checkpoint = torch.load(current_net_filename)
        #print(checkpoint['state_dict'].keys)
        #print(checkpoint)
        #current_cnet.load_state_dict(checkpoint['state_dict'])
        idxx = 0
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
        wins = [0,0]
        starts = []
        winList = []
        p1 = 0
        p2 = 0
        
        while games_completed < ((6+9*args.num_evaluator_games)//32):
            v = random.uniform(0,1)
           
            logger.info("[CPU: %d]: Game %d" % (cpu, idxx))
            print(v)
            if v > 0.5:
                t = 0
            else:
                t = 1
            starts.append(t+1)
            if t == 0:
                #print("network1 start")
                p1 += 1
            else:
                #print("network2 start")
                p2 += 1
            while not game_over:
                    action_dict = {}
                    #print("CURRENT PLAYER: ", env.game.player)
                    action = env.action_space.sample()
                    #print(env.game.get_moves())
                    action = random.choice(env.game.get_moves())
                    #print(env.game.get_moves(), action)
                    
                    a = [0,1,2,3,4,5,6]
                    #policy = evaluate_position(args, env.game, current_cnet)
                    policy_1 = evaluate_position(args, env.game, current_cnet_1)
                    #print(policy_1)
                    if t == 0:
                        action_dict[0] = action
                        action_dict[1] = np.random.choice(a,p=policy_1)
                        #action_dict[0] = np.random.choice(a, p=policy)
                    #print("bot: ", policy_1+1)
                    
                    #print(policy, np.argmax(policy))
                    if t == 1:
                        action_dict[0] = np.random.choice(a,p=policy_1)
                        action_dict[1] = action
                    #print("player (2), MCTS decision:", np.argmax(policy)+1)
                    obses, rewards, game_over, info = env.step(action_dict)
                    env.render()
                    #print(env.game.current_board)
                    
            if game_over:
                winner = obses[0]['winner']
                winners[winner] = winners[winner] + 1
                games_completed += 1
                idxx += 1
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
                elif t == 1 and winner == 1:
                    winList.append(1)
                    print("iteration " + str(it1) + " wins")
                    wins[0]  = wins[0] + 1
                
                if games_completed == ((6+9*args.num_evaluator_games)//32):
                    game_over = True
                    f = open("./logger/result_it9.txt", "a")
                    f.write("random " + str(wins[0]) + ", network " + str(wins[1]) + "\n")
                    f.close()
         
           
                else:
                    game_over = False
                    env.reset()
               

    def start_net_vs_net_cores(args, iteration, it1, it2):
        #queue = mp.SimpleQueue()
        num_processes = 1
        if args.MCTS_num_processes > 1:
            logger.info("Preparing model for multi-process evaluation...")
            mp.set_start_method("spawn",force=True)
           
            processes = []
            if args.MCTS_num_processes > mp.cpu_count():
                num_processes = mp.cpu_count()
                logger.info("Required number of processes exceed number of CPUs! Setting MCTS_num_processes to %d" % num_processes)
            else:
                num_processes = args.MCTS_num_processes
        
            logger.info("Spawning %d processes..." % num_processes)
            with torch.no_grad():
                for i in range(num_processes):
                    p = mp.Process(target=play_net_vs_net, args=(args, iteration, it1, it2, i))
                    p.start()
                    processes.append(p)
                for p in processes:
                    p.join()
            logger.info("Finished multi-process evaluation!")


    def log_wins(iteration, games):
        wins = [0,0]
        f = open("./logger/result_it9.txt", "r")
        lines = f.readlines()
        for line in lines:       
            first,last = line.replace(' ', '').split(',')
            v1,v2 = int(first[6:]), int(last[7:])
            wins[0] += v1
            wins[1] += v2
        now = datetime.now()
        f.close()
        f = open("./random_log/log1.txt", "a")            
        current_time = now.strftime("%H:%M:%S")
        current_date = datetime.today()
        d1 = current_date.strftime("%d/%m/%Y")
        f.write(d1 + " " + current_time + ", winrate:" + str(wins[1]/(6+30*args.num_evaluator_games)) + " after " + str(games) + " played with iteration: " + str(iteration) + " " + "\n")
        f.close()
        open("./logger/result_it9.txt", "w").close()

    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("START SELF PLAY: ", current_time)
    #test_random(0)
    
    p1 = 0
    p2 = 0
    starts = []
    winList = []

    logger.info("Starting iteration pipeline...")
    for i in range(args.iteration, args.total_iterations):
        #if i == 0:
        #    run_MCTS(args, start_idx=0, iteration=i)
        #    train_connectnet(args, iteration=i, new_optim_state=True)
        if i >= 1:
            net_to_play="%s_iter%d.pth.tar" % (args.neural_net_name, i+1)
            net = ConnectNet()
            net.share_memory()
            net.eval()
            torch.save({'state_dict': net.state_dict()}, os.path.join("",net_to_play))
            logger.info("Initialized model without training.")
            run_MCTS(args, start_idx=0, iteration=i)
            train_connectnet(args, iteration=i+1, new_optim_state=True)
            winner = evaluate_nets(args, i, i+1)
            counts = 0
            while (winner != (i+1)):
                logger.info("Trained net didn't perform better, generating more MCTS games for retraining...")
                run_MCTS(args, start_idx=(counts + 1)*args.num_games_per_MCTS_process, iteration=i)
                counts += 1
                train_connectnet(args, iteration=i+1, new_optim_state=True)
                winner = evaluate_nets(args, i, i+1)
            start_net_vs_net_cores(args, counts, i+1, i+1)
            log_wins(i+1,counts)


    #    print(wins)
    #start_net_vs_net_cores(args,0,0,9)
    #log_wins(9,(0+1)*args.num_games_per_MCTS_process*args.MCTS_num_processes)
    #train_connectnet(args, iteration=7, new_optim_state=True)
    #start_net_vs_net_cores(args, 0, 0, 7)
    #log_wins(7, 1*args.num_games_per_MCTS_process*args.MCTS_num_processes+7224)
    #start_net_vs_net_cores(args,0,0,7)
    #log_wins(7,1*args.num_games_per_MCTS_process*args.MCTS_num_processes+6552)
    #start_net_vs_net_cores(args, 0,0,7)
    #log_wins(8, 0)     
    #log_wins(8,wins, 0*args.num_games_per_MCTS_process*args.MCTS_num_processes)
    #print(torch.cuda.current_device())
    #now = datetime.now()
    #run_MCTS(args, start_idx=0, iteration=1)
    #test_random(0)
    #current_time = now.strftime("%H:%M:%S")
    #print("FINISHED SELF PLAY: ", current_time)
    #now = datetime.now()
    #current_time = now.strftime("%H:%M:%S")
    #print("START EVALUATION", current_time)
    #evaluate_nets(args, iteration_1=3, iteration_2=5)
    #now = datetime.now()
    #current_time = now.strftime("%H:%M:%S")
    #print("END EVALUATION", current_time)
    #start_net_vs_net_cores(0,2,8)
    #wins = [0,0]
    #starts = []
    #winList = []
    #p1 = 0
    #p2 = 0
    #for i in range(20):
    #    play_net_vs_net(2, 7)
    #    print("network indices")
    #    print(wins, winList, starts)
    #print("network indices")
    #print(wins, winList, starts)
    #play_net_vs_net(1,2,7)
    #print(wins, winList, starts)

    
    
    #mp.set_start_method('spawn')
    #q = mp.Queue()
    #p = mp.Process(target=foo, args=(q,))
    ##p.start()
    #print(q.get())
    #p.join()