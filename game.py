import gym
import gym_connect4


env = gym.make('Connect4Env-v0')

agents = ['Agent1()', 'Agent2()']
obses = env.reset()  # dict: {0: obs_player_1, 1: obs_player_2}
game_over = False
while not game_over:
    action_dict = {}
    for agent_id, agent in enumerate(agents):
        action = env.action_space.sample()
        action_dict[agent_id] = action
    obses, rewards, game_over, info = env.step(action_dict)
    env.render()
    if game_over:
        print("WINNER: ", obses[0]['winner'])
        


        

 