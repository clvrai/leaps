"""
Compact karel environment for speeding up Supervised learning
"""
import time
import numpy as np
from multiprocessing import Pool

import sys
sys.path.insert(0, './')
sys.path.insert(0, './karel_env')
from karel_env import generator

MAX_NUM_MARKER = 10

state_table = {
    0: 'Karel facing North',
    1: 'Karel facing East',
    2: 'Karel facing South',
    3: 'Karel facing West',
    4: 'Wall',
    5: '0 marker',
    6: '1 marker',
    7: '2 markers',
    8: '3 markers',
    9: '4 markers',
    10: '5 markers',
    11: '6 markers',
    12: '7 markers',
    13: '8 markers',
    14: '9 markers',
    15: '10 markers'
}
action_table = {
    0: 'Move',
    1: 'Turn left',
    2: 'Turn right',
    3: 'Pick up a marker',
    4: 'Put a marker'
}

# get location (x, y) and facing {north, east, south, west}
def get_location(s):
    x, y, z = np.where(s[:, :, :4] > 0)
    return np.array([x[0], y[0], z[0]])

# get the neighbor {front, left, right} location
def get_neighbor(s, loc, face):
    if face == 0:
        neighbor_loc = loc[:2] + {
            0: [-1, 0],
            1: [0, 1],
            2: [1, 0],
            3: [0, -1]
        }[loc[2]]
    elif face == 1:
        neighbor_loc = loc[:2] + {
            0: [0, -1],
            1: [-1, 0],
            2: [0, 1],
            3: [1, 0]
        }[loc[2]]
    elif face == 2:
        neighbor_loc = loc[:2] + {
            0: [0, 1],
            1: [1, 0],
            2: [0, -1],
            3: [-1, 0]
        }[loc[2]]
    return neighbor_loc

###################################
###    Perception Primitives    ###
###################################
# return if the neighbor {front, left, right} of Karel is clear
def neighbor_is_clear(s, loc, face):
    h, w = s.shape[0], s.shape[1]
    neighbor_loc = get_neighbor(s, loc, face)
    if neighbor_loc[0] >= h or neighbor_loc[0] < 0 \
            or neighbor_loc[1] >= w or neighbor_loc[1] < 0:
        return False
    return not s[neighbor_loc[0], neighbor_loc[1], 4], neighbor_loc

def front_is_clear(s, loc):
    return neighbor_is_clear(s, loc, 0)

def left_is_clear(s, loc):
    return neighbor_is_clear(s, loc, 1)

def right_is_clear(s, loc):
    return neighbor_is_clear(s, loc, 2)

def marker_present(s, loc):
    return np.sum(s[loc[0], loc[1], 6:]) > 0

def no_marker_present(s, loc):
    return np.sum(s[loc[0], loc[1], 6:]) == 0

def get_perception_vector(s, loc):
    vec = [front_is_clear(s, loc)[0], left_is_clear(s, loc)[0],
           right_is_clear(s, loc)[0], marker_present(s, loc),
           no_marker_present(s, loc)]
    return np.array(vec)

###################################
###       State Transition      ###
###################################
# given a state and a action, return the next state
def state_transition(s, a, make_error):
    made_error = False
    a_idx = a
    loc = get_location(s)

    if a_idx == 0:
        # move
        fic, front_loc = front_is_clear(s, loc)
        if fic:
            loc_vec = s[loc[0], loc[1], :4]
            s[front_loc[0], front_loc[1], :4] = loc_vec
            s[loc[0], loc[1], :4] = np.zeros(4) > 0
        else:
            if make_error:
                raise RuntimeError("Failed to move.")
            loc_vec = np.zeros(4) > 0
            loc_vec[(loc[2] + 2) % 4] = True  # Turn 180
            s[loc[0], loc[1], :4] = loc_vec
    elif a_idx == 1 or a_idx == 2:
        # turn left or right
        loc_vec = np.zeros(4) > 0
        loc_vec[(a_idx * 2 - 3 + loc[2]) % 4] = True
        s[loc[0], loc[1], :4] = loc_vec
    elif a_idx == 3 or a_idx == 4:
        # pick up or put a marker
        num_marker = s[loc[0], loc[1], 5:].argmax()
        # just clip the num of markers for now
        # new_num_marker = np.clip(a_idx*2-7 + num_marker, 0, MAX_NUM_MARKER-1)
        new_num_marker = a_idx*2-7 + num_marker
        if new_num_marker < 0:
            if make_error:
                raise RuntimeError("No marker to pick up.")
            else:
                new_num_marker = num_marker
            made_error = True
        elif new_num_marker > MAX_NUM_MARKER-1:
            if make_error:
                raise RuntimeError("Cannot put more marker.")
            else:
                new_num_marker = num_marker
            made_error = True
        marker_vec = np.zeros(MAX_NUM_MARKER+1) > 0
        marker_vec[new_num_marker] = True
        s[loc[0], loc[1], 5:] = marker_vec
    else:
        if not a > 4:
            raise RuntimeError("Invalid action")
    return s

class Karel_world_supervised(object):
    def __init__(self, s=None, make_error=True):
        if s is not None:
            self.set_new_state(s)
        self.make_error = make_error
        self.num_actions = len(action_table)

    def set_new_state(self, s):
        self.s = s.copy().astype(np.bool)
        self.h = self.s.shape[-3]
        self.w = self.s.shape[-2]

    ###################################
    ###    Collect Demonstrations   ###
    ###################################

    def print_state(self, state=None):
        agent_direction = {0: 'N', 1: 'E', 2: 'S', 3: 'W'}
        state = self.s if state is None else state
        state_2d = np.chararray(state.shape[:2])
        state_2d[:] = '.'
        state_2d[state[:,:,4]] = 'x'
        state_2d[state[:,:,6]] = 'M'
        x, y, z = np.where(state[:, :, :4] > 0)
        state_2d[x[0], y[0]] = agent_direction[z[0]]

        state_2d = state_2d.decode()
        for i in range(state_2d.shape[0]):
            print("".join(state_2d[i]))

    def step(self, a):
        assert self.s.shape[0] == a.shape[0]
        # self.print_state(self.s[0])
        xyz = [state_transition(self.s[i], a[i], self.make_error) if a[i] < 5 else self.s[i] for i in range(self.s.shape[0])]
        list(xyz)
        #self.print_state(self.s[0])
        return self.s

    def render(self, mode='rgb_array'):
        return self.s

    def reset(self, init_states):
        self.set_new_state(init_states)
        assert len(init_states.shape) == 4
        assert init_states.shape[-1] == 16
        assert init_states.shape[-3] == self.h
        assert init_states.shape[-2] == self.w


if __name__ == "__main__":
    world = Karel_world_batched(s=None, make_error=False)
    w, h = 8, 8

    s_gen = generator.KarelStateGenerator(seed=1)
    s, _, _, _, _ = s_gen.generate_single_state_mountain_climber(h,w)

    s = np.tile(np.expand_dims(s,0), (2560,1,1,1))
    world.reset(s)
    a = 0 * np.ones(2560, dtype=np.int16)
    t = time.time()
    for i in range(116):
        print("batch: ",i)
        start = time.time()
        for j in range(1*1*100):
            s = world.step(a.reshape(2560))
        print("Time for one batch: ", time.time()-start)
    print("total time: {}", time.time()-t)


