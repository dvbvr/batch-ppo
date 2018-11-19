import numpy as np
import retro
import gym

class ScreenFrame:
    def __init__(self, min=0, max=255, h=104, w=255, c=3):
        self.min = min
        self.max = max
        self.h = h
        self.w = w
        self.c = c


class MarioKartWrapper(gym.Wrapper):
    def __init__(self, env):
        super(MarioKartWrapper, self).__init__(env)
        buttons = ["B", "Y", "SELECT", "START", "UP", "DOWN", "LEFT", "RIGHT", "A", "X", "L", "R"]
        actions = [[], ['B'], ['LEFT'], ['RIGHT'], ['B', 'LEFT'], ['B', 'RIGHT'], ['A'], ['X','B','LEFT'], ['X','B','RIGHT'], ['R'], ['A', 'B'],['R','B']]
        self._actions = []
        for action in actions:
            arr = np.array([False] * 12)
            for button in action:
                arr[buttons.index(button)] = True
            self._actions.append(arr)
        self.action_space = gym.spaces.Discrete(len(self._actions))

        self.obs_shape = (104, 255, 3)
        self.frame = ScreenFrame()
        self.observation_space = gym.spaces.Box(
            low=self.frame.min, high=self.frame.max, shape=[self.frame.h, self.frame.w, self.frame.c], dtype=np.float32)  #(104, 256, 3)
        # self.observation_space = gym.spaces.Box(
        #     low=self.frame.min, high=self.frame.max, [self.frame.h, self.frame.w, self.frame.c])  #(104, 256, 3)
        self.viewer = None
        self.states_dict = {
            'cc':   ['50', '100'],
            'cup':  ['1', '2', '3'],
            'track':['1', '2', '3', '4', '5']
        }

    def action(self, a):
        return self._actions[a].copy()

    @staticmethod
    def create(cc=50, cup=1, race=1, players=1, record=False):
        if record:
            return retro.make(game='SuperMarioKart-Snes', players=players, record='.')
        else:
            return retro.make(game='SuperMarioKart-Snes', players=players)

    def set_state(self, cc=None, cup=None, track=None):
        statename = 'SuperMarioKart.SP.'
        statename += cc if cc else np.random.choice(self.states_dict['cc'])
        statename += 'cc.Cup'
        statename += cup if cup else np.random.choice(self.states_dict['cup'])
        statename += '.Race'
        statename += track if track else np.random.choice(self.states_dict['track'])
        self.env.load_state(statename)
        return self.crop(self.env.reset())

    def step(self, action, repeat=15):
        keys = self.get_keys(action)
        reward = 0
        for _ in range(repeat):
            obs, r, done, _ = self.env.step(keys)
            r = r if type(r) != list else r[0]
            reward += r
            if done:
                break
        return self.crop(obs), reward, done, _

    def get_keys(self, action):
        if hasattr(action, '__len__'):
            if self.env.players == 1 or len(action) == 1:
                return self._actions[action[0]]
            elif self.env.players == 2:
                return self._actions[action[0]] + self._actions[action[1]]
        else:
            return self._actions[action]
        
    def crop(self, obs, player=0):
        if player == 0:
            return obs[3:107, :, :]
        elif player == 1:
            return obs[115:219, :, :]

    def reset(self, **kwargs):
        return self.crop(self.env.reset(**kwargs))

    def render(self, player=-1):
        if player == -1:
            return self.env.render()
        else:
            if not self.viewer:
                from gym.envs.classic_control.rendering import SimpleImageViewer
                self.viewer = [SimpleImageViewer(), SimpleImageViewer()]
            return self.viewer[player].imshow(self.crop(self.env.img, player))


if __name__ == '__main__':
    env = MarioKartWrapper(MarioKartWrapper.create())
    observation = env.reset()
    for _ in range(1000):
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        print(reward)