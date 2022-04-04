import gym
import sys
import gym_connect4
from MCTS_c4 import run_MCTS
from train_c4 import train_connectnet
from argparse import ArgumentParser
from evaluator_c4 import evaluate_nets


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
    winner = evaluate_nets(args, 5, 0)
    print(winner)

    # agents = ['Agent1()', 'Agent2()']
    # obses = env.reset()  # dict: {0: obs_player_1, 1: obs_player_2}
    # game_over = False
    # while not game_over:
    #     action_dict = {}
    #     for agent_id, agent in enumerate(agents):
    #         #action = env.action_space.sample()
    #         print("ran mcts")
    #         run_MCTS(args, start_idx=0, iteration=5)
    #         #train_connectnet(args, iteration=5, new_optim_state=True)
    #         #action_dict[agent_id] = action
    #     obses, rewards, game_over, info = env.step(action_dict)
    #     env.render()
    #     if game_over:
    #         print("WINNER: ", obses[0]['winner'])




        

 