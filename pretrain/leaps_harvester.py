
config = {
    'device': 'cuda:0',                             # device to run on
    'save_interval': 1000,                          # Save weights at every ith interval (None, int)
    'log_interval': 10,                             # Logging interval for debugging (None, int)
    'record_file': 'records.pkl',                   # File to save global records dictionary
    'algorithm': 'CEM',                      # current training algorithm: 'supervised', 'RL', 'supervisedRL', 'debug', 'output_dataset_split'
    'mode': 'train',                                # 'train', 'eval'

    'do_supervised': True,                          # do supervised training in supervisedRL algorithm if set True
    'do_RL': True,                                  # do RL training in supervisedRL algorithm if set True

    # config for logging
    'logging': {
        'log_file': 'run.log',                      # log file name
        'fmt': '%(asctime)s: %(message)s',          # logging format
        'level': 'DEBUG',                           # logger level
        'wandb': False,                             # enable wandb logging
    },

    # config to load and save networks
    'net': {
        'saved_params_path': None,                  # path to load saved weights in a loaded network
        'saved_sup_params_path': None,              # path to load saved weights from supervised training
        'rnn_type': 'GRU',                          # recurrent unit type
        'decoder': {
            'use_teacher_enforcing': True,          # teacher enforcing while SL training
            'freeze_params': False                  # freeze decoder params if set True
        },
        'condition':{
            'freeze_params': False,
            'use_teacher_enforcing': True,
            'observations': 'environment',          # condition policy input from ['environment', 'dataset', 'initial_state']
        },
        'controller':{
            'add_noise': False,                     # add nosie to meta-controller weights like StyleGAN
            'input_coef': 0.01,                     # if using constant vector as input to controller, use this as multiplier
            'use_decoder_dist': True,               # if True, RL on decoder distribution, otherwise RL on meta-controller distribution
            'use_previous_programs': False,         # if True, use previous program as input to meta-controller
            'program_reduction': 'identity',        # 'identity': no-reduction, 'mean': mean of all previous program as input to meta-controller
        },
        'tanh_after_mu_sigma': False,               # apply tanh after distribution (mean and std of VAE) layers
        'tanh_after_sample': False,                 # apply tanh after sampling from VAE distribution
    },

    # config for data loader
    'data_loader': {
        'num_workers': 0,                           # Number of parallel CPU workers
        'pin_memory': False,                        # Copy tensors into CUDA pinned memory before returning them
#        'collate_fn': lambda x: x,                 # collate_fn to get minibatch as list instead of tensor
        'drop_last': True,
    },

    # Random seed for numpy, torch and cuda (None, int)
    'seed': 123,

    'optimizer': {
        'name': 'adam',
        'params': {
            'lr': 5e-4,
        },
        'scheduler': {
            'step_size': 10,                        # Period of learning rate decay
            'gamma': .95,                           # Multiplicative factor of learning rate decay
        }
    },

    # config to control training
    'train': {
        'data': {                                   # Dictionary to control dataset characteristics
            'to_tensor': True,
            'use_pickled': True
        },
        'batch_size': 256,
        'shuffle': True,
        'max_epoch': 150,
    },
    # config to control validation
    'valid': {
        'data': {                                   # Dictionary to control dataset characteristics
            'to_tensor': True,
            'use_pickled': True
        },
        'batch_size': 256,
        'shuffle': True,
        'debug_samples': [3, 37, 54],               # sample ids to generate plots for (None, int, list)
    },
    # config to control testing
    'test': {
        'data': {                                   # Dictionary to control dataset characteristics
            'to_tensor': True,
            'use_pickled': True
        },
        'batch_size': 256,
        'shuffle': True,
    },
    # config to control evaluation
    'eval': {
        'usage': 'test',                            # what dataset to use {train, valid, test}
    },
    'loss': {
        'latent_loss_coef': 1.0,                    # coefficient of latent loss (beta) in VAE during SL training
        'condition_loss_coef': 1.0,                 # coefficient of condition policy loss during SL training
    },
    'dsl': {
        'use_simplified_dsl': False,                # reducing valid tokens from 50 to 31
        'max_program_len': 45,                      # maximum program length
        'grammar': 'handwritten',                   # grammar type: [None, 'handwritten']
    },
    'rl':{
        'num_processes': 64,                        # how many training CPU processes to use (default: 32)
        'num_steps': 8,                             # 'number of forward steps (default: 32)'
        'num_env_steps': 10e6,                      # 'number of environment steps to train (default: 10e6)'
        'gamma': 0.99,                              # discount factor for rewards (default: 0.99)
        'use_gae': True,                            # 'use generalized advantage estimation'
        'gae_lambda': 0.95,                         # 'gae lambda parameter (default: 0.95)'
        'use_proper_time_limits': False,            # 'compute returns taking into account time limits'
        'use_all_programs': False,                  # 'False sets all mask value to 1 (ignores done_ variable value in trainer.py)'
        'future_rewards': False,                    # True: Maximizing expected future reward, False: Maximizing current reward
        'value_method': 'mean',                     # mean: mean of token values, program_embedding: value of eop token
        'envs': {
            'executable': {
                'name': 'karel',
                'task_definition': 'custom_reward',       # choices=['program', 'custom_reward']
                'task_file': 'tasks/test1.txt',  # choose from these tokens to write a space separated VALID program
                # for ground_truth task: ['DEF', 'run', 'm(', 'm)', 'move', 'turnRight',
                # 'turnLeft', 'pickMarker', 'putMarker', 'r(', 'r)', 'R=0', 'R=1', 'R=2',
                # 'R=3', 'R=4', 'R=5', 'R=6', 'R=7', 'R=8', 'R=9', 'R=10', 'R=11', 'R=12',
                # 'R=13', 'R=14', 'R=15', 'R=16', 'R=17', 'R=18', 'R=19', 'REPEAT', 'c(',
                # 'c)', 'i(', 'i)', 'e(', 'e)', 'IF', 'IFELSE', 'ELSE', 'frontIsClear',
                # 'leftIsClear', 'rightIsClear', 'markersPresent', 'noMarkersPresent',
                # 'not', 'w(', 'w)', 'WHILE']
                'max_demo_length': 100,             # maximum demonstration length
                'min_demo_length': 1,               # minimum demonstration length
                'num_demo_per_program': 10,         # 'number of seen demonstrations'
                'dense_execution_reward': False,    # encode reward along with state and action if task defined by custom reward
            },
            'program': {
                'mdp_type': 'ProgramEnv1',          # choices=['ProgramEnv1']
                'intrinsic_reward': False,          # NGU paper based intrinsic reward
                'intrinsic_beta': 0.0,              # reward = env_reward + intrinsic_beta * intrinsic_reward
            }
        },
        'policy':{
          'execution_guided': False,                # 'enable execution guided program synthesis'
          'two_head': False,                        # 'predict end-of-program token separate than program tokens'
          'recurrent_policy': True,                 # 'use a recurrent policy'
        },
        'algo':{
            'name': 'reinforce',
            'value_loss_coef':0.5,                  # 'value loss coefficient (default: 0.5)'
            'entropy_coef':0.1,                     # 'entropy term coefficient (default: 0.01)'
            'final_entropy_coef': 0.01,             # 'final entropy term coefficient (default: None)'
            'use_exp_ent_decay': False,             # 'use a exponential decay schedule on the entropy coef'
            'use_recurrent_generator': False,       # 'use episodic memory replay'
            'max_grad_norm': 0.5,                   # 'max norm of gradients (default: 0.5)'
            'lr': 5e-4,                             # 'learning rate (default: 5e-4)'
            'use_linear_lr_decay': True,            # 'use a linear schedule on the learning rate'
            'ppo':{
                'clip_param':0.1,                   # 'ppo clip parameter (default: 0.1)'
                'ppo_epoch':2,                      # 'number of ppo epochs (default: 4)'
                'num_mini_batch':2,                 # 'number of batches for ppo (default: 4)'
                'eps': 1e-5,                        # 'RMSprop optimizer epsilon (default: 1e-5)'
            },
            'a2c':{
                'eps': 1e-5,                        # 'RMSprop optimizer epsilon (default: 1e-5)'
                'alpha': 0.99,                      # 'RMSprop optimizer apha (default: 0.99)'
            },
            'acktr':{
            },
            'reinforce': {
                'clip_param': 0.1,                  # 'ppo clip parameter (default: 0.1)'
                'reinforce_epoch': 1,               # 'number of ppo epochs (default: 4)'
                'num_mini_batch': 2,                # 'number of batches for ppo (default: 4)'
                'eps': 1e-5,                        # 'RMSprop optimizer epsilon (default: 1e-5)'
            },
        },
        'loss':{
                'decoder_rl_loss_coef': 1.0,            # coefficient of policy loss during RL training
                'condition_rl_loss_coef': 0.0,          # coefficient of condition network loss during RL training
                'latent_rl_loss_coef': 0.0,             # coefficient of latent loss (beta) in VAE during RL training
                'use_mean_only_for_latent_loss': False, # applying latent loss only to mean while searching over latent space
            }
    },
    'CEM':{
        'init_type': 'normal',
        'reduction': 'weighted_mean',
        'population_size': 32,
        'elitism_rate': 0.1,
        'max_number_of_epochs': 1000,
        'sigma': 0.5,
        'final_sigma': 0.1,
        'use_exp_sig_decay': True,
        'exponential_reward': False,
        'average_score_for_solving': 1.1,
        'detailed_dump': False,

    },
    # FIXME: This is only for backwards compatibility to old parser, should be removed soon
    'policy': 'TokenOutputPolicy',                  # output one token at a time (Ignore for intention space)
    'env_name': 'karel',
    'gamma': 0.99,                                  # discount factor for rewards (default: 0.99)
    'recurrent_policy': True,                       # If True, use RNN in policy network
    'num_lstm_cell_units': 256,                      # RNN latent space size
    'two_head': False,                              # do we want two headed policy? Not for LEAPS
    'mdp_type': 'ProgramEnv1',                      # ProgramEnv1: only allows syntactically valid program execution
    'env_task': 'harvester',                          # ['program', 'stairClimber', 'placeSetter', 'topOff', 'chainSmoker', 'shelfStocker', 'maze']
    'reward_diff': True,                            # If True, differnce between rewards of two consecutive states will be considered at each env step, otherwise current environment reward will be considered
    'prefix': 'default',                            # output directory prefix
    'max_program_len': 45,                          # maximum program length  (repeated)
    'mapping_file': None,                           # mapping_karel2prl.txt if using simplified DSL (Ignore of intention space)
    'debug': False,                                 # use this to debug RL code (provides a lot of debug data in form of dict)
    'height': 8,                                    # height of karel environment
    'width': 8,                                     # width of karel environment
    #'wall_prob': 0.1,                               # p(wall/one cell in karel gird)
    'wall_prob': 0.25,                               # p(wall/one cell in karel gird)
    'num_demo_per_program': 10,                     # 'number of seen demonstrations' (repeated)
    'max_demo_length': 100,                         # maximum demonstration length (repeated)
    'min_demo_length': 1,                           # minimum demonstration length (repeated)
    'action_type': 'program',                       # Ignore for intention space
    'obv_type': 'program',                          # Ignore for intention space
    'reward_type': 'sparse',       # sparse, extra_sparse, dense_subsequence_match, dense_frame_set_match, dense_last_state_match
    'reward_validity': False,                       # reward for syntactically valid programs (Ignore for intention space)
    'max_episode_steps': 100,                       # maximum steps in one episode before environment reset
    'AE': False,                                    # using plain AutoEncoder instead of VAE
    'experiment': 'intention_space',                # intention_space or EGPS
    'grammar':'handwritten',                        # grammar type: [None, 'handwritten']
    'use_trainable_tensor': False,                  # If True, use trainable tensor instead of meta-controller
    'cover_all_branches_in_demos': True,            # If True, make sure to cover all branches in randomly generated program in ExecEnv1
    'final_reward_scale': False
}
