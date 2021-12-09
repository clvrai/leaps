import os
import numpy as np
import scipy
from scipy import spatial


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


class Karel_world(object):

    def __init__(self, s=None, make_error=True, env_task="program", task_definition='program' ,reward_diff=False, final_reward_scale=True):
        if s is not None:
            self.set_new_state(s)
        self.make_error = make_error
        self.env_task = env_task
        self.task_definition = task_definition
        self.rescale_reward = True
        self.final_reward_scale = final_reward_scale
        self.reward_diff = reward_diff
        self.num_actions = len(action_table)

    def set_new_state(self, s, metadata=None):
        self.perception_count = 0
        self.s = s.astype(np.bool)
        self.s_h = [self.s.copy()]
        self.a_h = []
        self.h = self.s.shape[0]
        self.w = self.s.shape[1]
        p_v = self.get_perception_vector()
        self.p_v_h = [p_v.copy()]

        if self.task_definition != "program":
            self.r_h = []
            self.d_h = []
            self.program_reward = 0.0
            self.prev_pos_reward = 0.0
            self.done = False
            self.metadata = metadata
            self.total_markers = np.sum(s[:,:,6:])

    ###################################
    ###    Collect Demonstrations   ###
    ###################################

    def clear_history(self):
        self.perception_count = 0
        self.s_h = [self.s.copy()]
        self.a_h = []
        self.p_v_h = []

        if self.task_definition != "program":
            self.r_h = []
            self.d_h = []
            self.program_reward = 0.0
            self.prev_pos_reward = 0.0
            self.done = False
            self.total_markers = np.sum(self.s_h[-1][:,:,6:])

    def add_to_history(self, a_idx, agent_pos, made_error=False):
        self.s_h.append(self.s.copy())
        self.a_h.append(a_idx)
        p_v = self.get_perception_vector()
        self.p_v_h.append(p_v.copy())

        if self.task_definition != "program":
            reward, done = self._get_state_reward(agent_pos, made_error)
            self.done = self.done or done
            self.r_h.append(reward)
            self.d_h.append(done)
            self.program_reward += reward

        if self.task_definition != 'program' and not made_error:
            if a_idx == 3: self.total_markers -= 1
            if a_idx == 4: self.total_markers += 1

    def _get_cleanHouse_task_reward(self, agent_pos):
        done = False
        reward = 0
        w = self.w
        h = self.h
        state = self.s_h[-1]

        for mpos in self.metadata['marker_positions']:
            if state[mpos[0], mpos[1], 5] and not state[mpos[0], mpos[1], 6]:
                reward += 1

        reward = reward / len(self.metadata['marker_positions'])
        done = reward == 1

        reward = reward if self.env_task == 'cleanHouse' else float(done)
        self.done = self.done or done
        return reward, done

    def _get_harvester_task_reward(self, agent_pos):
        done = False
        reward = 0
        w = self.w
        h = self.h
        state = self.s_h[-1]

        max_markers = (w-2)*(h-2)
        reward = ( max_markers - self.total_markers ) / max_markers
        done = reward == 1

        reward = reward if self.env_task == 'harvester' else float(done)
        self.done = self.done or done
        return reward, done

    def _get_randomMaze_task_reward(self, agent_pos):
        done = False
        reward = 0
        w = self.w
        h = self.h
        state = self.s_h[-1]

        # initial marker position
        init_state = self.s_h[0]
        x, y = np.where(init_state[:, :, 6] > 0)
        if len(x) != 1: assert 0, '{} markers found!'.format(len(x))
        marker_pos = np.asarray([x[0], y[0]])
        reward = -1 * spatial.distance.cityblock(agent_pos[:2], marker_pos)

        done = reward == 0
        reward = float(done)
        self.done = self.done or done
        return reward, done

    def _get_fourCorners_task_reward(self, agent_pos):
        done = False
        reward = 0
        w = self.w
        h = self.h
        state = self.s_h[-1]

        #give 0.25 reward for putting marker in each corner
        if state[1, 1, 6]:
            reward += 0.25
        if state[h-2, 1, 6]:
            reward += 0.25
        if state[h-2, w-2, 6]:
            reward += 0.25
        if state[1, w-2, 6]:
            reward += 0.25

        #give zero reward if agent places marker anywhere else
        correct_markers = int(reward*4)
        incorrect_markers = self.total_markers - correct_markers
        if incorrect_markers > 0:
            reward = 0
        done = reward == 1

        reward = reward if self.env_task == 'fourCorners' else float(done)
        if self.env_task == 'fourCorners_sparse':
            reward = reward if done and not self.done else 0
        self.done = self.done or done
        return reward, done

    def _get_stairClimber_task_reward(self, agent_pos):
        done = False
        reward = 0
        state = self.s_h[-1]

        # initial marker position
        init_state = self.s_h[0]
        x, y = np.where(init_state[:, :, 6] > 0)
        if len(x) != 1: assert 0, '{} markers found!'.format(len(x))
        marker_pos = np.asarray([x[0], y[0]])
        reward = -1 * spatial.distance.cityblock(agent_pos[:2], marker_pos)

        # NOTE: need to do this to avoid high negative reward for first action
        if len(self.s_h) == 2:
            x, y, z = np.where(self.s_h[0][:, :, :4] > 0)
            prev_pos = np.asarray([x[0], y[0], z[0]])
            self.prev_pos_reward = -1 * spatial.distance.cityblock(prev_pos[:2], marker_pos)

        if not self.reward_diff:
            # since reward is based on manhattan distance, rescale it to range between 0 to 1
            if self.rescale_reward:
                from_min, from_max, to_min, to_max = -(sum(self.s.shape[:2])), 0, -1, 0
                reward = ((reward - from_min) * (to_max - to_min) / (from_max - from_min)) + to_min
            if tuple(agent_pos[:2]) not in self.metadata['agent_valid_positions']:
                reward = -1.0
            done = reward == 0
        else:
            abs_reward = reward
            reward = self.prev_pos_reward-1.0 if tuple(agent_pos[:2]) not in self.metadata['agent_valid_positions'] else reward
            reward = reward - self.prev_pos_reward
            self.prev_pos_reward = abs_reward
            done = abs_reward == 0

        reward = reward if self.env_task == 'stairClimber' else float(done)
        if self.env_task == 'stairClimber_sparse':
            reward = reward if done and not self.done else 0
        self.done = self.done or done
        return reward, done

    def _get_topOff_task_reward(self, agent_pos):
        assert self.reward_diff
        done = False
        reward = 0
        w = self.w
        h = self.h
        state = self.s_h[-1]

        for c in range(1, agent_pos[1]+1):
            if (h-2, c) in self.metadata['not_expected_marker_positions']:
                if state[h-2, c, 7]:
                    reward += 1
                else:
                    break
            else:
                assert (h-2, c) in self.metadata['expected_marker_positions']
                if state[h-2, c, 5]:
                    reward += 1
                else:
                    break

        if (self.w - 2 == agent_pos[1] and self.h - 2 == agent_pos[0]) and reward == w-2:
            reward += 1

        reward = reward / (w-1)

        done = sum([state[pos[0], pos[1], 7] for pos in self.metadata['not_expected_marker_positions']]) == len(
            self.metadata['not_expected_marker_positions']) and (self.w - 2 == agent_pos[1] and self.h - 2 == agent_pos[0]) and reward==1.0

        reward = reward if self.env_task == 'topOff' else float(done)
        if self.env_task == 'topOff_sparse':
            reward = reward if done and not self.done else 0
        self.done = self.done or done
        return reward, done

    def _get_state_reward(self, agent_pos, made_error=False):
        if self.env_task == 'cleanHouse' or self.env_task == 'cleanHouse_sparse':
            reward, done = self._get_cleanHouse_task_reward(agent_pos)
        elif self.env_task == 'harvester' or self.env_task == 'harvester_sparse':
            reward, done = self._get_harvester_task_reward(agent_pos)
        elif self.env_task == 'fourCorners' or self.env_task == 'fourCorners_sparse':
            reward, done = self._get_fourCorners_task_reward(agent_pos)
        elif self.env_task == 'randomMaze' or self.env_task == 'randomMaze_sparse':
            reward, done = self._get_randomMaze_task_reward(agent_pos)
        elif self.env_task == 'stairClimber' or self.env_task == 'stairClimber_sparse':
            reward, done = self._get_stairClimber_task_reward(agent_pos)
        elif self.env_task == 'topOff' or self.env_task == 'topOff_sparse':
            reward, done = self._get_topOff_task_reward(agent_pos)
        else:
            raise NotImplementedError('{} task not yet supported'.format(self.env_task))

        return reward, done

    def print_state(self, state=None):
        agent_direction = {0: 'N', 1: 'E', 2: 'S', 3: 'W'}
        state = self.s_h[-1] if state is None else state
        state_2d = np.chararray(state.shape[:2])
        state_2d[:] = '.'
        state_2d[state[:,:,4]] = 'x'
        state_2d[state[:,:,6]] = 'M'
        x, y, z = np.where(state[:, :, :4] > 0)
        state_2d[x[0], y[0]] = agent_direction[z[0]]

        state_2d = state_2d.decode()
        for i in range(state_2d.shape[0]):
            print("".join(state_2d[i]))

    def render(self, mode='rgb_array'):
        return self.s_h[-1]

    # get location (x, y) and facing {north, east, south, west}
    def get_location(self):
        x, y, z = np.where(self.s[:, :, :4] > 0)
        return np.asarray([x[0], y[0], z[0]])

    # get the neighbor {front, left, right} location
    def get_neighbor(self, face):
        loc = self.get_location()
        if face == 'front':
            neighbor_loc = loc[:2] + {
                0: [-1, 0],
                1: [0, 1],
                2: [1, 0],
                3: [0, -1]
            }[loc[2]]
        elif face == 'left':
            neighbor_loc = loc[:2] + {
                0: [0, -1],
                1: [-1, 0],
                2: [0, 1],
                3: [1, 0]
            }[loc[2]]
        elif face == 'right':
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
    def neighbor_is_clear(self, face):
        self.perception_count += 1
        neighbor_loc = self.get_neighbor(face)
        if neighbor_loc[0] >= self.h or neighbor_loc[0] < 0 \
                or neighbor_loc[1] >= self.w or neighbor_loc[1] < 0:
            return False
        return not self.s[neighbor_loc[0], neighbor_loc[1], 4]

    def front_is_clear(self):
        return self.neighbor_is_clear('front')

    def left_is_clear(self):
        return self.neighbor_is_clear('left')

    def right_is_clear(self):
        return self.neighbor_is_clear('right')

    # return if there is a marker presented
    def marker_present(self):
        self.perception_count += 1
        loc = self.get_location()
        return np.sum(self.s[loc[0], loc[1], 6:]) > 0

    def no_marker_present(self):
        self.perception_count += 1
        loc = self.get_location()
        return np.sum(self.s[loc[0], loc[1], 6:]) == 0

    def get_perception_list(self):
        vec = ['frontIsClear', 'leftIsClear',
               'rightIsClear', 'markersPresent',
               'noMarkersPresent']
        return vec

    def get_perception_vector(self):
        vec = [self.front_is_clear(), self.left_is_clear(),
               self.right_is_clear(), self.marker_present(),
               self.no_marker_present()]
        return np.array(vec)

    ###################################
    ###       State Transition      ###
    ###################################
    # given a state and a action, return the next state
    def state_transition(self, a):
        made_error = False
        a_idx = np.argmax(a)
        loc = self.get_location()


        if a_idx == 0:
            # move
            if self.front_is_clear():
                front_loc = self.get_neighbor('front')
                loc_vec = self.s[loc[0], loc[1], :4]
                self.s[front_loc[0], front_loc[1], :4] = loc_vec
                self.s[loc[0], loc[1], :4] = np.zeros(4) > 0
                next_loc = front_loc
            else:
                if self.make_error:
                    raise RuntimeError("Failed to move.")
                loc_vec = np.zeros(4) > 0
                loc_vec[(loc[2] + 2) % 4] = True  # Turn 180
                self.s[loc[0], loc[1], :4] = loc_vec
                next_loc = loc
            self.add_to_history(a_idx, next_loc)
        elif a_idx == 1 or a_idx == 2:
            # turn left or right
            loc_vec = np.zeros(4) > 0
            loc_vec[(a_idx * 2 - 3 + loc[2]) % 4] = True
            self.s[loc[0], loc[1], :4] = loc_vec
            self.add_to_history(a_idx, loc)

        elif a_idx == 3 or a_idx == 4:
            # pick up or put a marker
            num_marker = np.argmax(self.s[loc[0], loc[1], 5:])
            # just clip the num of markers for now
            # new_num_marker = np.clip(a_idx*2-7 + num_marker, 0, MAX_NUM_MARKER-1)
            new_num_marker = a_idx*2-7 + num_marker
            if new_num_marker < 0:
                if self.make_error:
                    raise RuntimeError("No marker to pick up.")
                else:
                    new_num_marker = num_marker
                made_error = True
            elif new_num_marker > MAX_NUM_MARKER-1:
                if self.make_error:
                    raise RuntimeError("Cannot put more marker.")
                else:
                    new_num_marker = num_marker
                made_error = True
            marker_vec = np.zeros(MAX_NUM_MARKER+1) > 0
            marker_vec[new_num_marker] = True
            self.s[loc[0], loc[1], 5:] = marker_vec
            self.add_to_history(a_idx, loc, made_error)
        else:
            raise RuntimeError("Invalid action")
        return

    # given a karel env state, return a visulized image
    def state2image(self, s=None, grid_size=100, root_dir='./'):
        h = s.shape[0]
        w = s.shape[1]
        img = np.ones((h*grid_size, w*grid_size, 1))
        import pickle
        from PIL import Image
        import os.path as osp
        f = pickle.load(open(osp.join(root_dir, 'karel_env/asset/texture.pkl'), 'rb'))
        wall_img = f['wall'].astype('uint8')
        marker_img = f['marker'].astype('uint8')
        agent_0_img = f['agent_0'].astype('uint8')
        agent_1_img = f['agent_1'].astype('uint8')
        agent_2_img = f['agent_2'].astype('uint8')
        agent_3_img = f['agent_3'].astype('uint8')
        blank_img = f['blank'].astype('uint8')
        #blanks
        for y in range(h):
            for x in range(w):
                img[y*grid_size:(y+1)*grid_size, x*grid_size:(x+1)*grid_size] = blank_img
        # wall
        y, x = np.where(s[:, :, 4])
        for i in range(len(x)):
            img[y[i]*grid_size:(y[i]+1)*grid_size, x[i]*grid_size:(x[i]+1)*grid_size] = wall_img
        # marker
        y, x = np.where(np.sum(s[:, :, 6:], axis=-1))
        for i in range(len(x)):
            img[y[i]*grid_size:(y[i]+1)*grid_size, x[i]*grid_size:(x[i]+1)*grid_size] = marker_img
        # karel
        y, x = np.where(np.sum(s[:, :, :4], axis=-1))
        if len(y) == 1:
            y = y[0]
            x = x[0]
            idx = np.argmax(s[y, x])
            marker_present = np.sum(s[y, x, 6:]) > 0
            if marker_present:
                extra_marker_img = Image.fromarray(f['marker'].squeeze()).copy()
                if idx == 0:
                    extra_marker_img.paste(Image.fromarray(f['agent_0'].squeeze()))
                    extra_marker_img = f['marker'].squeeze() + f['agent_0'].squeeze()
                    extra_marker_img = np.minimum(f['marker'].squeeze() , f['agent_0'].squeeze())
                    img[y*grid_size:(y+1)*grid_size, x*grid_size:(x+1)*grid_size] = np.expand_dims(np.array(extra_marker_img), axis=-1)
                elif idx == 1:
                    extra_marker_img.paste(Image.fromarray(f['agent_1'].squeeze()))
                    extra_marker_img = f['marker'].squeeze() + f['agent_1'].squeeze()
                    extra_marker_img = np.minimum(f['marker'].squeeze() , f['agent_1'].squeeze())
                    img[y*grid_size:(y+1)*grid_size, x*grid_size:(x+1)*grid_size] = np.expand_dims(np.array(extra_marker_img), axis=-1)
                elif idx == 2:
                    extra_marker_img.paste(Image.fromarray(f['agent_2'].squeeze()))
                    extra_marker_img = f['marker'].squeeze() + f['agent_2'].squeeze()
                    extra_marker_img = np.minimum(f['marker'].squeeze() , f['agent_2'].squeeze())
                    img[y*grid_size:(y+1)*grid_size, x*grid_size:(x+1)*grid_size] = np.expand_dims(np.array(extra_marker_img), axis=-1)
                elif idx == 3:
                    extra_marker_img.paste(Image.fromarray(f['agent_3'].squeeze()))
                    extra_marker_img = f['marker'].squeeze() + f['agent_3'].squeeze()
                    extra_marker_img = np.minimum(f['marker'].squeeze() , f['agent_3'].squeeze())
                    img[y*grid_size:(y+1)*grid_size, x*grid_size:(x+1)*grid_size] = np.expand_dims(np.array(extra_marker_img), axis=-1)
            else:
                if idx == 0:
                    img[y*grid_size:(y+1)*grid_size, x*grid_size:(x+1)*grid_size] = f['agent_0']
                elif idx == 1:
                    img[y*grid_size:(y+1)*grid_size, x*grid_size:(x+1)*grid_size] = f['agent_1']
                elif idx == 2:
                    img[y*grid_size:(y+1)*grid_size, x*grid_size:(x+1)*grid_size] = f['agent_2']
                elif idx == 3:
                    img[y*grid_size:(y+1)*grid_size, x*grid_size:(x+1)*grid_size] = f['agent_3']
        elif len(y) > 1:
            raise ValueError
        return img
