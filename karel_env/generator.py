from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import h5py
import os
import argparse
import progressbar
import random
import pickle

import numpy as np

from dsl import get_DSL
from dsl.dsl_parse_and_trace import parse_and_trace
from util import log

import karel


class KarelStateGenerator(object):
    def __init__(self, seed=None):
        self.rng = np.random.RandomState(seed)

    def print_state(self, state=None):
        agent_direction = {0: 'N', 1: 'E', 2: 'S', 3: 'W'}
        state_2d = np.chararray(state.shape[:2])
        state_2d[:] = '.'
        state_2d[state[:,:,4]] = 'x'
        state_2d[state[:,:,6]] = 'M'
        x, y, z = np.where(state[:, :, :4] > 0)
        state_2d[x[0], y[0]] = agent_direction[z[0]]

        state_2d = state_2d.decode()
        for i in range(state_2d.shape[0]):
            print("".join(state_2d[i]))

    # generate an initial env
    def generate_single_state(self, h=8, w=8, wall_prob=0.1, env_task_metadata={}):
        s = np.zeros([h, w, 16]) > 0
        # Wall
        s[:, :, 4] = self.rng.rand(h, w) > 1 - wall_prob
        s[0, :, 4] = True
        s[h-1, :, 4] = True
        s[:, 0, 4] = True
        s[:, w-1, 4] = True
        # Karel initial location
        valid_loc = False
        while(not valid_loc):
            y = self.rng.randint(0, h)
            x = self.rng.randint(0, w)
            if not s[y, x, 4]:
                valid_loc = True
                s[y, x, self.rng.randint(0, 4)] = True
        # Marker: num of max marker == 1 for now
        s[:, :, 6] = (self.rng.rand(h, w) > 0.9) * (s[:, :, 4] == False) > 0
        s[:, :, 5] = 1 - (np.sum(s[:, :, 6:], axis=-1) > 0) > 0
        assert np.sum(s[:, :, 5:]) == h*w, np.sum(s[:, :, :5])
        marker_weight = np.reshape(np.array(range(11)), (1, 1, 11))
        return s, y, x, np.sum(s[:, :, 4]), np.sum(marker_weight*s[:, :, 5:])

    # generate an initial env for cleanHouse problem
    def generate_single_state_clean_house(self, h=12, w=12, wall_prob=0.1, env_task_metadata={}, is_top_off=False):
        """
        initial state generator for cleanHouse problem
        Valid program for cleanHouse problem:
        DEF run m( WHILE c( frontIsClear c) w( IF c( markersPresent c) i( pickMarker i) IFELSE c( leftIsClear c) i( turnLeft i) ELSE e( move e) w) m)

        :param h:
        :param w:
        :param wall_prob:
        :return:
        """

        world_map = [
            ['-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-'],
            ['-',   0,   0, '-', '-', '-', '-', '-', '-', '-', '-', '-', '-',   0, '-',   0,   0,   0,   0,   0,   0, '-'],
            ['-',   0,   0, '-',   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, '-', '-', '-', '-', '-',   0, '-', '-'],
            ['-', '-',   0, '-',   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, '-', '-'],
            ['-', '-',   0,   0,   0,   0,   0,   0,   0,   0, '-',   0,   0,   0,   0,   0,   0,   0,   0,   0, '-', '-'],
            ['-', '-', '-',   0, '-',   0, '-', '-', '-',   0, '-',   0,   0, '-', '-', '-',   0, '-',   0, '-', '-', '-'],
            ['-',   0,   0,   0, '-',   0,   0,   0, '-',   0,   0,   0,   0, '-',   0,   0,   0, '-',   0,   0, '-', '-'],
            ['-',   0,   0,   0, '-',   0,   0,   0, '-',   0,   0,   0,   0, '-',   0,   0,   0, '-',   0,   0,   0, '-'],
            ['-',   0,   0,   0, '-',   0,   0,   0, '-',   0,   0,   0,   0, '-',   0,   0,   0, '-', '-',   0,   0, '-'],
            ['-',   0,   0,   0, '-',   0,   0,   0, '-', '-',   0,   0, '-', '-',   0,   0,   0, '-', '-',   0, '-', '-'],
            ['-',   0,   0,   0, '-',   0,   0,   0, '-', '-',   0,   0, '-', '-',   0,   0,   0, '-',   0,   0, '-', '-'],
            ['-',   0,   0,   0, '-',   0,   0,   0, '-', '-',   0,   0, '-', '-',   0,   0,   0, '-',   0,   0, '-', '-'],
            ['-',   0,   0,   0, '-',   0,   0,   0, '-', '-',   0,   0, '-', '-',   0,   0,   0, '-',   0,   0, '-', '-'],
            ['-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-'],
        ]

        assert h == 14 and w == 22, 'karel maze environment should be 13 x 13, found {} x {}'.format(h, w)
        s = np.zeros([h, w, 16]) > 0
        # Wall
        s[0, :, 4] = True
        s[h - 1, :, 4] = True
        s[:, 0, 4] = True
        s[:, w - 1, 4] = True

        # Karel initial location
        agent_pos = (1, 13)
        hardcoded_invalid_marker_locations = [(1, 13), (2, 12), (3, 10), (4, 11), (5, 11), (6, 10)]
        s[agent_pos[0], agent_pos[1], 2] = True


        for y1 in range(h):
            for x1 in range(w):
                s[y1, x1, 4] = world_map[y1][x1] == '-'
                s[y1, x1, 5] = True if world_map[y1][x1] != 'M' else False
                s[y1, x1, 6] = True if world_map[y1][x1] == 'M' else False

        expected_marker_positions = set()
        for y1 in range(h):
            for x1 in range(13):
                if s[y1, x1, 4]:
                    if y1 - 1 > 0 and not s[y1 -1, x1, 4]: expected_marker_positions.add((y1 - 1,x1))
                    if y1 + 1 < h - 1 and not s[y1 +1, x1, 4]: expected_marker_positions.add((y1 + 1,x1))
                    if x1 - 1 > 0 and not s[y1, x1 - 1, 4]: expected_marker_positions.add((y1,x1 - 1))
                    if x1 + 1 < 13 - 1 and not s[y1, x1 + 1, 4]: expected_marker_positions.add((y1,x1 + 1))

        # put 2 markers near start point for end condition
        s[agent_pos[0]+1, agent_pos[1]-1, 5] = False
        s[agent_pos[0]+1, agent_pos[1]-1, 7] = True

        # place 10 Markers
        expected_marker_positions = list(expected_marker_positions)
        random.shuffle(expected_marker_positions)
        assert len(expected_marker_positions) >= 10
        marker_positions = []
        for i, mpos in enumerate(expected_marker_positions):
            if mpos in hardcoded_invalid_marker_locations:
                continue
            s[mpos[0], mpos[1], 5] = False
            s[mpos[0], mpos[1], 6] = True
            marker_positions.append(mpos)
            if len(marker_positions) == 10:
                break

        assert np.sum(s[:, :, 8:]) == 0
        metadata = {'agent_valid_positions': None, 'expected_marker_positions': expected_marker_positions, 'marker_positions': marker_positions}

        return s, agent_pos[0], agent_pos[1], np.sum(s[:, :, 4]), metadata


    # generate an initial env for fourCorners problem
    def generate_single_state_harvester(self, h=8, w=8, wall_prob=0.1, env_task_metadata={}, is_top_off=False):
        """
        initial state generator for harvester problem
        Valid program for harvester problem:
        DEF run m( WHILE c( markersPresent c) w( WHILE c( markersPresent c) w( pickMarker move w) turnRight move turnLeft WHILE c( markersPresent c) w( pickMarker move w) turnLeft move turnRight w) m)

        :param h:
        :param w:
        :param wall_prob:
        :return:
        """
        mode = env_task_metadata.get("mode", "train")
        marker_prob = env_task_metadata.get("train_marker_prob", 1.0) if mode == 'train' else env_task_metadata.get("test_marker_prob", 1.0)


        s = np.zeros([h, w, 16]) > 0
        # Wall
        s[0, :, 4] = True
        s[h-1, :, 4] = True
        s[:, 0, 4] = True
        s[:, w-1, 4] = True

        # initial karel position: karel facing east at the last row in environment
        agent_pos = (h-2, 1)
        s[agent_pos[0], agent_pos[1], 1] = True

        # put 1 marker at every location in grid
        if marker_prob == 1.0:
            s[1:h-1, 1:w-1, 6] = True
        else:
            valid_marker_pos = np.array([(r,c) for r in range(1,h-1) for c in range(1,w-1)])
            marker_pos = valid_marker_pos[np.random.choice(len(valid_marker_pos), size=int(marker_prob*len(valid_marker_pos)), replace=False)]
            for pos in marker_pos:
                s[pos[0], pos[1], 6] = True

        metadata = {}

        return s, agent_pos[0], agent_pos[1], np.sum(s[:, :, 4]), metadata

    # generate an initial env for randomMaze problem
    def generate_single_state_random_maze(self, h=8, w=8, wall_prob=0.1, env_task_metadata={}, is_top_off=False):
        """
        initial state generator for random maze problem
        Valid program for random maze problem:
        DEF run m( WHILE c( noMarkersPresent c) w( IFELSE c( rightIsClear c) i( turnRight i) ELSE e( WHILE c( not c( frontIsClear c) c) w( turnLeft w) e) move w) m)

        :param h:
        :param w:
        :param wall_prob:
        :return:
        """

        def get_neighbors(cur_pos, h, w):
            neighbor_list = []
            #neighbor top
            if cur_pos[0] - 2 > 0: neighbor_list.append([cur_pos[0] - 2, cur_pos[1]])
            # neighbor bottom
            if cur_pos[0] + 2 < h - 1: neighbor_list.append([cur_pos[0] + 2, cur_pos[1]])
            # neighbor left
            if cur_pos[1] - 2 > 0: neighbor_list.append([cur_pos[0], cur_pos[1] - 2])
            # neighbor right
            if cur_pos[1] + 2 < w - 1: neighbor_list.append([cur_pos[0], cur_pos[1] + 2])
            return neighbor_list

        s = np.zeros([h, w, 16]) > 0
        # convert every location to wall
        s[:, :, 4] = True
        #start from bottom left corner
        init_pos = [h - 2, 1]
        visited = np.zeros([h,w])
        stack = []

        # convert initial location to empty location from wall
        # iterative implementation of random maze generator at https://en.wikipedia.org/wiki/Maze_generation_algorithm
        s[init_pos[0], init_pos[1], 4] = False
        visited[init_pos[0], init_pos[1]] = True
        stack.append(init_pos)

        while len(stack) > 0:
            cur_pos = stack.pop()
            neighbor_list = get_neighbors(cur_pos, h, w)
            random.shuffle(neighbor_list)
            for neighbor in neighbor_list:
                if not visited[neighbor[0], neighbor[1]]:
                    stack.append(cur_pos)
                    s[(cur_pos[0]+neighbor[0])//2, (cur_pos[1]+neighbor[1])//2, 4] = False
                    s[neighbor[0], neighbor[1], 4] = False
                    visited[neighbor[0], neighbor[1]] = True
                    stack.append(neighbor)

        # init karel agent position
        agent_pos = init_pos
        s[agent_pos[0], agent_pos[1], 1] = True

        # Marker location
        valid_loc = False
        while (not valid_loc):
           ym = self.rng.randint(0, h)
           xm = self.rng.randint(0, w)
           if not s[ym, xm, 4]:
               valid_loc = True
               s[ym, xm, 6] = True
               assert not s[ym, xm, 4]

        assert not s[ym, xm, 4]

        # put 0 markers everywhere but 1 location
        s[:, :, 5] = 1 - (np.sum(s[:, :, 6:], axis=-1) > 0) > 0
        assert np.sum(s[:, :, 6]) == 1
        assert np.sum(s[:, :, 7:]) == 0
        metadata = {'agent_valid_positions': None}

        return s, agent_pos[0], agent_pos[1], np.sum(s[:, :, 4]), metadata

    # generate an initial env for fourCorners problem
    def generate_single_state_four_corners(self, h=8, w=8, wall_prob=0.1, env_task_metadata={}, is_top_off=False):
        """
        initial state generator for four corners problem
        Valid program for four corners problem:
        DEF run m( WHILE c( noMarkersPresent c) w( WHILE c( frontIsClear c) w( move w) IF c( noMarkersPresent c) i( putMarker turnLeft move i) w) m)

        :param h:
        :param w:
        :param wall_prob:
        :return:
        """
        s = np.zeros([h, w, 16]) > 0
        # Wall
        s[0, :, 4] = True
        s[h-1, :, 4] = True
        s[:, 0, 4] = True
        s[:, w-1, 4] = True

        # initial karel position: karel facing east at the last row in environment
        agent_pos = (h-2, 2)
        s[agent_pos[0], agent_pos[1], 1] = True

        metadata = {}

        return s, agent_pos[0], agent_pos[1], np.sum(s[:, :, 4]), metadata

    # generate an initial env
    def generate_single_state_chain_smoker(self, h=8, w=8, wall_prob=0.1, env_task_metadata={}, is_top_off=False):
        """
        initial state generator for chain smoker and top off problem both
        Valid program for chain smoker problem:
        DEF run m( WHILE c( frontIsClear c) w( IF c( noMarkersPresent c) i( putMarker i) move w) m)
        Valid program for top off problem:
        DEF run m( WHILE c( frontIsClear c) w( IF c( MarkersPresent c) i( putMarker i) move w) m)

        :param h:
        :param w:
        :param wall_prob:
        :return:
        """
        s = np.zeros([h, w, 16]) > 0
        # Wall
        s[0, :, 4] = True
        s[h-1, :, 4] = True
        s[:, 0, 4] = True
        s[:, w-1, 4] = True

        # initial karel position: karel facing east at the last row in environment
        agent_pos = (h-2, 1)
        s[agent_pos[0], agent_pos[1], 1] = True

        # randomly put markers at row h-2
        s[h-2, 1:w-1, 6] = self.rng.rand(w-2) > 1 - wall_prob
        # NOTE: need marker in last position as the condition is to check till I reach end
        s[h-2, w-2, 6] = True if not is_top_off else False
        s[:, :, 5] = 1 - (np.sum(s[:, :, 6:], axis=-1) > 0) > 0
        assert np.sum(s[:,:,5:]) == w*h

        # randomly generate wall at h-3 row
        mode = env_task_metadata.get('mode', 'train')
        hash_info_path = env_task_metadata.get('hash_info', None)
        if is_top_off and hash_info_path is not None:
            train_configs = env_task_metadata.get('train_configs', 1.0)
            test_configs = env_task_metadata.get('test_configs', 1.0)
            hash_info = pickle.load(open(hash_info_path,"rb"))
            assert hash_info['w'] == w and hash_info['h'] == h
            hashtable = hash_info['table']
            split_idx = int(len(hashtable)*train_configs) if mode == 'train' else int(len(hashtable)*test_configs)
            hashtable = hashtable[:split_idx] if mode == 'train' else hashtable[-split_idx:]
            key = s[h-2, 1:w-2, 6].tostring()
            if key not in hashtable:
                return self.generate_single_state_chain_smoker(h, w, wall_prob, env_task_metadata, is_top_off)

        # generate valid agent positions
        valid_agent_pos = [(h-2, c) for c in range(1, w-1)]
        agent_valid_positions = list(set(valid_agent_pos))
        # generate valid marker positions
        expected_marker_positions = [(h-2, c) for c in range(1, w-1) if not s[h-2, c, 6]]
        not_expected_marker_positions = [(h-2, c) for c in range(1, w-1) if s[h-2, c, 6]]
        metadata = {'agent_valid_positions':agent_valid_positions,
                    'expected_marker_positions':expected_marker_positions,
                    'not_expected_marker_positions': not_expected_marker_positions}

        return s, agent_pos[0], agent_pos[1], np.sum(s[:, :, 4]), metadata

    # generate an initial env
    def generate_single_state_stair_climber(self, h=8, w=8, wall_prob=0.1, env_task_metadata={}):
        s = np.zeros([h, w, 16]) > 0
        # Wall
        s[0, :, 4] = True
        s[h-1, :, 4] = True
        s[:, 0, 4] = True
        s[:, w-1, 4] = True

        world_map = [
            ['-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-'],
            ['-',   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, '-'],
            ['-',   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, '-', '-'],
            ['-',   0,   0,   0,   0,   0,   0,   0,   0,   0, '-', '-', '-'],
            ['-',   0,   0,   0,   0,   0,   0,   0,   0, '-', '-',   0, '-'],
            ['-',   0,   0,   0,   0,   0,   0,   0, '-', '-',   0,   0, '-'],
            ['-',   0,   0,   0,   0,   0,   0, '-', '-',   0,   0,   0, '-'],
            ['-',   0,   0,   0,   0,   0, '-', '-',   0,   0,   0,   0, '-'],
            ['-',   0,   0,   0,   0, '-', '-',   0,   0,   0,   0,   0, '-'],
            ['-',   0,   0,   0, '-', '-',   0,   0,   0,   0,   0,   0, '-'],
            ['-',   0,   0, '-', '-',   0,   0,   0,   0,   0,   0,   0, '-'],
            ['-',   0, '-', '-',   0,   0,   0,   0,   0,   0,   0,   0, '-'],
            ['-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-'],
        ]

        c = 2
        r = h - 1
        valid_agent_pos = []
        valid_init_pos = []
        while r > 0 and c < w:
            s[r, c, 4] = True
            s[r - 1, c, 4] = True
            if r - 1 > 0 and c - 1 > 0:
                valid_agent_pos.append((r - 1, c - 1))
                valid_init_pos.append((r - 1, c - 1))
                assert not s[r - 1, c - 1, 4] , "there shouldn't be a wall at {}, {}".format(r - 1, c - 1)
            if r - 2 > 0 and c - 1 > 0:
                valid_agent_pos.append((r - 2, c - 1))
                assert not s[r - 2, c - 1, 4], "there shouldn't be a wall at {}, {}".format(r - 2, c - 1)
            if r - 2 > 0 and c > 0:
                valid_agent_pos.append((r - 2, c))
                valid_init_pos.append((r - 2, c))
                assert not s[r - 2, c, 4], "there shouldn't be a wall at {}, {}".format(r - 2, c)
            c += 1
            r -= 1

        agent_valid_positions = list(set(valid_agent_pos))
        valid_init_pos = sorted(list(set(valid_init_pos)), key=lambda x: x[1])

        # Karel initial location
        l1, l2 = 0, 0
        while l1 == l2:
            l1, l2 = self.rng.randint(0, len(valid_init_pos)), self.rng.randint(0, len(valid_init_pos))
        agent_idx, marker_idx = min(l1, l2), max(l1, l2)
        agent_pos, marker_pos = valid_init_pos[agent_idx], valid_init_pos[marker_idx]
        assert (not s[agent_pos[0], agent_pos[1], 4]) and not (s[marker_pos[0], marker_pos[1], 4])
        s[agent_pos[0], agent_pos[1], 1] = True

        # Marker: num of max marker == 1 for now
        s[:, :, 5] = True
        s[marker_pos[0], marker_pos[1], 5] = False
        s[marker_pos[0], marker_pos[1], 6] = True

        assert np.sum(s[:, :, 6]) == 1
        assert np.sum(s[:, :, 7:]) == 0
        metadata = {'agent_valid_positions': agent_valid_positions}
        return s, agent_pos[0], agent_pos[1], np.sum(s[:, :, 4]), metadata


def _branch_execution_ratio(record_dict):
    if len(record_dict) == 0:
        return None

    total_branches = 2 * len(record_dict)
    executed_branches = 0
    for key, value in record_dict.items():
        branch_dict = value[0][1]
        executed_branches += int(branch_dict[True]) + int(branch_dict[False])
    return executed_branches / total_branches

def generator(config):
    dir_name = config.dir_name
    h = config.height
    w = config.width
    c = len(karel.state_table)
    wall_prob = config.wall_prob
    num_train = config.num_train
    num_test = config.num_test
    num_val = config.num_val
    num_total = num_train + num_test + num_val

    # output files
    f = h5py.File(os.path.join(dir_name, 'data.hdf5'), 'w')
    id_file = open(os.path.join(dir_name, 'id.txt'), 'w')

    # progress bar
    bar = progressbar.ProgressBar(maxval=100,
                                  widgets=[progressbar.Bar('=', '[', ']'), ' ',
                                           progressbar.Percentage()])
    bar.start()

    dsl = get_DSL(dsl_type='prob', seed=config.seed, environment='karel')
    s_gen = KarelStateGenerator(seed=config.seed)
    karel_world = karel.Karel_world()

    count = 0
    failed_exec_count = 0
    max_demo_length_in_dataset = -1
    max_program_length_in_dataset = -1
    min_demo_length_in_dataset = float('inf')
    min_program_length_in_dataset = float('inf')
    seen_programs = set()
    while(1):
        # generate a single program
        random_code = dsl.random_code(max_depth=config.max_program_stmt_depth,
                                      max_nesting_depth=config.max_program_nesting_depth)
        # skip seen programs
        if random_code in seen_programs:
            continue
        program_seq = np.array(dsl.code2intseq(random_code), dtype=np.int8)
        if program_seq.shape[0] > config.max_program_length:
            continue

        # parse program to enable execution tracing
        if config.cover_all_branches_in_demos:
            exe, s_exe, record_dict = parse_and_trace(random_code, environment='karel')
            if len(record_dict) == 0 and ('WHILE' in random_code or 'IF' in random_code):
                assert 0, 'only non-conditional programs will have empty dict'
            prev_exec_ratio = exec_ratio = _branch_execution_ratio(record_dict)
            assert prev_exec_ratio == 0.0 or prev_exec_ratio is None
            if not s_exe:
                raise RuntimeError('If we reach here, then we should be able to parse the program')

        s_h_list = []
        a_h_list = []
        num_demo = 0
        num_trial = 0
        num_err_trial = 0
        while num_demo < config.num_demo_per_program and \
                num_trial < config.max_demo_generation_trial:
            try:
                s, _, _, _, _ = s_gen.generate_single_state(h, w, wall_prob)
                karel_world.set_new_state(s)
                if not config.cover_all_branches_in_demos:
                    s_h = dsl.run(karel_world, random_code)
                else:
                    karel_world.clear_history()
                    karel_world, n, s_run = exe(karel_world, 0, record_dict, exe)
                    if not s_run:
                        raise RuntimeError("Program execution timeout.")
                    s_h = karel_world.s_h
            except RuntimeError:
                num_err_trial += 1
                pass
            else:

                if not config.cover_all_branches_in_demos:
                    if len(karel_world.s_h) <= config.max_demo_length and \
                            len(karel_world.s_h) >= config.min_demo_length:
                        s_h_list.append(np.stack(karel_world.s_h, axis=0))
                        a_h_list.append(np.array(karel_world.a_h))
                        num_demo += 1
                else:
                    exec_ratio = _branch_execution_ratio(record_dict)

                    if len(karel_world.s_h) <= config.max_demo_length and \
                            (len(karel_world.s_h) >= config.min_demo_length or (exec_ratio is not None and (exec_ratio > prev_exec_ratio or (exec_ratio == 1.0 and np.random.uniform() < 0.5)))) and \
                            (exec_ratio is None or exec_ratio > prev_exec_ratio or exec_ratio >= 1.0):
                        s_h_list.append(np.stack(karel_world.s_h, axis=0))
                        a_h_list.append(np.array(karel_world.a_h))
                        prev_exec_ratio = exec_ratio
                        num_demo += 1

            num_trial += 1

        if num_demo < config.num_demo_per_program:
            if config.cover_all_branches_in_demos and exec_ratio is not None and exec_ratio <= 1.0:
                failed_exec_count += 1
                print("exec_coverage_failure: {}/{} exec_cov:{} Only generated {}/{} demos with {}/{} env error trials for program: {}".format(
                    failed_exec_count, count, exec_ratio, num_demo, config.num_demo_per_program, num_err_trial, num_trial, random_code))
            continue

        len_s_h = np.array([s_h.shape[0] for s_h in s_h_list], dtype=np.int16)
        if np.max(len_s_h) < config.min_max_demo_length_for_program:
            continue

        demos_s_h = np.zeros([num_demo, np.max(len_s_h), h, w, c], dtype=bool)
        for i, s_h in enumerate(s_h_list):
            demos_s_h[i, :s_h.shape[0]] = s_h

        len_a_h = np.array([a_h.shape[0] for a_h in a_h_list], dtype=np.int16)

        demos_a_h = np.zeros([num_demo, np.max(len_a_h)], dtype=np.int8)
        for i, a_h in enumerate(a_h_list):
            demos_a_h[i, :a_h.shape[0]] = a_h

        max_demo_length_in_dataset = max(max_demo_length_in_dataset, np.max(len_s_h))
        max_program_length_in_dataset = max(max_program_length_in_dataset, program_seq.shape[0])
        min_demo_length_in_dataset = min(min_demo_length_in_dataset, np.min(len_s_h))
        min_program_length_in_dataset = min(min_program_length_in_dataset, program_seq.shape[0])

        # save the state
        id = 'no_{}_prog_len_{}_max_s_h_len_{}'.format(
            count, program_seq.shape[0], np.max(len_s_h))
        id_file.write(id+'\n')
        grp = f.create_group(id)
        grp['program'] = program_seq
        grp['s_h_len'] = len_s_h
        grp['a_h_len'] = len_a_h
        grp['s_h'] = demos_s_h
        grp['a_h'] = demos_a_h
        seen_programs.add(random_code)
        # progress bar
        count += 1
        if count % (num_total / 100) == 0:
            bar.update(count / (num_total / 100))
        if count >= num_total:
            grp = f.create_group('data_info')
            grp['max_demo_length'] = max_demo_length_in_dataset
            grp['min_demo_length'] = min_demo_length_in_dataset
            grp['dsl_type'] = 'prob'
            grp['max_program_length'] = max_program_length_in_dataset
            grp['min_program_length'] = min_program_length_in_dataset
            grp['num_program_tokens'] = len(dsl.int2token)
            grp['num_demo_per_program'] = config.num_demo_per_program
            grp['num_action_tokens'] = len(dsl.action_functions)
            grp['num_train'] = config.num_train
            grp['num_test'] = config.num_test
            grp['num_val'] = config.num_val
            bar.finish()
            f.close()
            id_file.close()
            log.info('Dataset generated under {} with {}'
                     ' samples ({} for training and {} for testing '
                     'and {} for val'.format(dir_name, num_total,
                                             num_train, num_test, num_val))
            return


def check_path(path):
    if not os.path.exists(path):
        os.mkdir(path)


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dir_name', type=str, default='karel_dataset')
    parser.add_argument('--height', type=int, default=8,
                        help='height of square grid world')
    parser.add_argument('--width', type=int, default=8,
                        help='width of square grid world')
    parser.add_argument('--num_train', type=int, default=25000, help='num train')
    parser.add_argument('--num_test', type=int, default=5000, help='num test')
    parser.add_argument('--num_val', type=int, default=5000, help='num val')
    parser.add_argument('--wall_prob', type=float, default=0.1,
                        help='probability of wall generation')
    parser.add_argument('--seed', type=int, default=123, help='seed')
    parser.add_argument('--max_program_length', type=int, default=50)
    parser.add_argument('--max_program_stmt_depth', type=int, default=6)
    parser.add_argument('--max_program_nesting_depth', type=int, default=4)
    parser.add_argument('--min_max_demo_length_for_program', type=int, default=2)
    parser.add_argument('--min_demo_length', type=int, default=4,
                        help='min demo length')
    parser.add_argument('--max_demo_length', type=int, default=20,
                        help='max demo length')
    parser.add_argument('--num_demo_per_program', type=int, default=10,
                        help='number of seen demonstrations')
    parser.add_argument('--max_demo_generation_trial', type=int, default=100)
    parser.add_argument('--cover_all_branches_in_demos', action='store_true', help='cover all conditional branches while generating demonstrations')
    args = parser.parse_args()
    args.dir_name = os.path.join('datasets/', args.dir_name)
    check_path('datasets')
    check_path(args.dir_name)

    generator(args)

if __name__ == '__main__':
    main()
