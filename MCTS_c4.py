#!/usr/bin/env python
import pickle
import os
import collections
import numpy as np
import math
from gym_connect4.envs.connect4_env import Connect4 as c_board
import copy
import torch
import torch.multiprocessing as mp
import encoder_decoder_c4 as ed
from alpha_net_c4 import ConnectNet
import datetime
import logging
from tqdm import tqdm
np.seterr(divide='ignore', invalid='ignore')
logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', \
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger(__file__)

def save_as_pickle(filename, data):
    completeName = os.path.join("./datasets/",\
                                filename)
    with open(completeName, 'wb') as output:
        pickle.dump(data, output)

def load_pickle(filename):
    completeName = os.path.join("./datasets/",\
                                filename)
    with open(completeName, 'rb') as pkl_file:
        data = pickle.load(pkl_file)
    return data

class UCTNode():
    def __init__(self, game, move, parent=None):
        self.game = game # state s
        self.move = move # action index
        self.is_expanded = False
        self.parent = parent  
        self.children = {}
        self.child_priors = np.zeros([7], dtype=np.float32)
        self.child_total_value = np.zeros([7], dtype=np.float32)
        self.child_number_visits = np.zeros([7], dtype=np.float32)
        self.action_idxes = []
        
    @property
    def number_visits(self):
        return self.parent.child_number_visits[self.move]

    @number_visits.setter
    def number_visits(self, value):
        self.parent.child_number_visits[self.move] = value
    
    @property
    def total_value(self):
        return self.parent.child_total_value[self.move]
    
    @total_value.setter
    def total_value(self, value):
        self.parent.child_total_value[self.move] = value
    
    def child_Q(self):
        return self.child_total_value / (1 + self.child_number_visits)
    
    def child_U(self):
        return math.sqrt(self.number_visits) * (
            abs(self.child_priors) / (1 + self.child_number_visits))
    
    def best_child(self):
        if self.action_idxes != []:
            bestmove = self.child_Q() + self.child_U()
            bestmove = self.action_idxes[np.argmax(bestmove[self.action_idxes])]
        else:
            bestmove = np.argmax(self.child_Q() + self.child_U())
        return bestmove
    
    def select_leaf(self):
        current = self
        while current.is_expanded:
          best_move = current.best_child()
          current = current.maybe_add_child(best_move)
        return current
    
    def add_dirichlet_noise(self,action_idxs,child_priors):
        valid_child_priors = child_priors[action_idxs] # select only legal moves entries in child_priors array
        valid_child_priors = 0.75*valid_child_priors + 0.25*np.random.dirichlet(np.zeros([len(valid_child_priors)], \
                                                                                          dtype=np.float32)+192)
        child_priors[action_idxs] = valid_child_priors
        return child_priors
    
    def expand(self, child_priors):
        self.is_expanded = True
        action_idxs = self.game.get_moves(); c_p = child_priors
        if action_idxs == []:
            self.is_expanded = False
        self.action_idxes = action_idxs
        c_p[[i for i in range(len(child_priors)) if i not in action_idxs]] = 0.000000000 # mask all illegal actions
        if self.parent.parent == None: # add dirichlet noise to child_priors in root node
            c_p = self.add_dirichlet_noise(action_idxs,c_p)
        self.child_priors = c_p
    
    def decode_n_move_pieces(self,board,move):
        board.move(move)
        return board
            
    def maybe_add_child(self, move):
        if move not in self.children:
            copy_board = copy.deepcopy(self.game) # make copy of board
            copy_board = self.decode_n_move_pieces(copy_board,move)
            self.children[move] = UCTNode(
              copy_board, move, parent=self)
        return self.children[move]
    
    def backup(self, value_estimate: float):
        current = self
        while current.parent is not None:
            current.number_visits += 1
            if current.game.player == 1: # same as current.parent.game.player = 0PE
                current.total_value += (1*value_estimate) # value estimate +1 = O wins
            elif current.game.player == 0: # same as current.parent.game.player = 1
                current.total_value += (-1*value_estimate)
            current = current.parent
        
class DummyNode(object):
    def __init__(self):
        self.parent = None
        self.child_total_value = collections.defaultdict(float)
        self.child_number_visits = collections.defaultdict(float)

def UCT_search(game_state, num_reads,net,temp, max_depth = 20):
    root = UCTNode(game_state, move=None, parent=DummyNode())
    cuda = torch.cuda.is_available()
    for i in range(num_reads):
        leaf = root.select_leaf()
        encoded_s = ed.encode_board(leaf.game); encoded_s = encoded_s.transpose(2,0,1)
<<<<<<< HEAD
        if not cuda:
            encoded_s = torch.from_numpy(encoded_s).float()
            
        if cuda:
            encoded_s = torch.from_numpy(encoded_s).cuda().float()
=======
        encoded_s = torch.from_numpy(encoded_s).cuda().float()
        if cuda:
>>>>>>> b9b338686fa4a3cba7ff0c68b1db2deabb55fb58
            net = net.cuda()
            
        #   print(encoded_s.to("cuda:0"))
        #   net = encoded_s.to("cuda:0")
        #   child_priors, value_estimate = net(encoded_s)    
        child_priors, value_estimate = net(encoded_s)
        
        child_priors = child_priors.detach().cpu().numpy().reshape(-1); value_estimate = value_estimate.item()
        if (leaf.game.is_winner(0) or leaf.game.is_winner(1)) == True or leaf.game.get_moves() == []: # if somebody won or draw
            leaf.backup(value_estimate); continue
        leaf.expand(child_priors) # need to make sure valid moves
        leaf.backup(value_estimate)
    return root

def do_decode_n_move_pieces(board,move):
    board.move(move)
    return board

def get_policy(root, temp=1):
    #policy = np.zeros([7], dtype=np.float32)
    #for idx in np.where(root.child_number_visits!=0)[0]:
    #    policy[idx] = ((root.child_number_visits[idx])**(1/temp))/sum(root.child_number_visits**(1/temp))
    return ((root.child_number_visits)**(1/temp))/sum(root.child_number_visits**(1/temp))

def MCTS_self_play(connectnet, num_games, start_idx, cpu, args, iteration):
    logger.info("[CPU: %d]: Starting MCTS self-play..." % cpu)
    
    if not os.path.isdir("./datasets/iter_%d" % iteration):
        if not os.path.isdir("datasets"):
            os.mkdir("datasets")
        os.mkdir("datasets/iter_%d" % iteration)
        
    for idxx in tqdm(range(start_idx, num_games + start_idx)):
        logger.info("[CPU: %d]: Game %d" % (cpu, idxx))
        current_board = c_board()
        checkmate = False
        dataset = [] # to get state, policy, value for neural network training
        states = []
        value = 0
        move_count = 0
        while checkmate == False and current_board.get_action_mask() != []:
            if move_count < 11:
                t = args.temperature_MCTS
            else:
                t = 0.1
            states.append(copy.deepcopy(current_board.current_board))
            board_state = copy.deepcopy(ed.encode_board(current_board))
            if torch.cuda.is_available():
                 connectnet.to("cuda:0")
            root = UCT_search(current_board,777,connectnet,t)
            policy = get_policy(root, t); 
            if args.print_board:
                print("[CPU: %d]: Game %d POLICY:\n " % (cpu, idxx), policy)
            current_board = do_decode_n_move_pieces(current_board,\
                                                    np.random.choice(np.array([0,1,2,3,4,5,6]), \
                                                                     p = policy)) # decode move and move piece(s)
            dataset.append([board_state,policy])
            if args.print_board:
                print("[Iteration: %d CPU: %d]: Game %d CURRENT BOARD:\n" % (iteration, cpu, idxx), current_board.current_board,current_board.player); print(" ")
            if (current_board.is_winner(0) or current_board.is_winner(1)) == True: # if somebody won
                if current_board.is_winner(0): # black wins
                    value = -1
                elif current_board.is_winner(1): # white wins
                    value = 1
                checkmate = True
            move_count += 1
        dataset_p = []
        for idx,data in enumerate(dataset):
            s,p = data
            if idx == 0:
                dataset_p.append([s,p,0])
            else:
                dataset_p.append([s,p,value])
        del dataset
        save_as_pickle("iter_%d/" % iteration +\
                       "dataset_iter%d_cpu%i_%i_%s" % (iteration, cpu, idxx, datetime.datetime.today().strftime("%Y-%m-%d")), dataset_p)
   
def run_MCTS(args, start_idx=0, iteration=0):
    net_to_play="%s_iter%d.pth.tar" % (args.neural_net_name, iteration)
    net = ConnectNet()
    cuda = torch.cuda.is_available()
    if cuda:
        net = net.cuda()
    
    if args.MCTS_num_processes > 1:
        logger.info("Preparing model for multi-process MCTS...")
        mp.set_start_method("spawn",force=True)
        net.share_memory()
        net.eval()
    
        current_net_filename = os.path.join("",\
                                        net_to_play)
        if os.path.isfile(current_net_filename):
            checkpoint = torch.load(current_net_filename)
            net.load_state_dict(checkpoint['state_dict'])
            logger.info("Loaded %s model." % current_net_filename)
        else:
            torch.save({'state_dict': net.state_dict()}, os.path.join("",\
                        net_to_play))
            logger.info("Initialized model.")
        
        processes = []
        if args.MCTS_num_processes > mp.cpu_count():
            num_processes = mp.cpu_count()
            logger.info("Required number of processes exceed number of CPUs! Setting MCTS_num_processes to %d" % num_processes)
        else:
            num_processes = args.MCTS_num_processes
        
        logger.info("Spawning %d processes..." % num_processes)
        with torch.no_grad():
            for i in range(num_processes):
                p = mp.Process(target=MCTS_self_play, args=(net, args.num_games_per_MCTS_process, start_idx, i, args, iteration))
                p.start()
                processes.append(p)
            for p in processes:
                p.join()
        logger.info("Finished multi-process MCTS!")
    
    elif args.MCTS_num_processes == 1:
        logger.info("Preparing model for MCTS...")
        net.eval()
        
        current_net_filename = os.path.join("",\
                                        net_to_play)
        if os.path.isfile(current_net_filename):
            checkpoint = torch.load(current_net_filename)
            net.load_state_dict(checkpoint['state_dict'])
            logger.info("Loaded %s model." % current_net_filename)
        else:
            torch.save({'state_dict': net.state_dict()}, os.path.join("",\
                        net_to_play))
            logger.info("Initialized model.")
        
        with torch.no_grad():
            MCTS_self_play(net, args.num_games_per_MCTS_process, start_idx, 0, args, iteration)
        logger.info("Finished MCTS!")

