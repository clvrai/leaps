
karel_obv_token_api_dict = {
    'frontIsClear': 'front_is_clear',
    'leftIsClear': 'left_is_clear',
    'rightIsClear': 'right_is_clear',
    'markersPresent': 'marker_present',
    'noMarkersPresent': 'no_marker_present'
}

karel_action_token_list = ['move', 'turnLeft', 'turnRight', 'pickMarker', 'putMarker']

cartpole_obv_token_api_dict = {
    'PolePosLeft': 'pole_pos_left',
    'PolePosCenter': 'pole_pos_center',
    'PolePosRight': 'pole_pos_right',
    'CartPosLeft': 'cart_pos_left',
    'CartPosCenter': 'cart_pos_center',
    'CartPosRight': 'cart_pos_right',
    'PoleVelFast': 'pole_vel_fast',
    'PoleVelMed': 'pole_vel_med',
    'PoleVelSlow': 'pole_vel_slow',
    'CartVelFast': 'cart_vel_fast',
    'CartVelMed': 'cart_vel_med',
    'CartVelSlow': 'cart_vel_slow',
}

cartpole_action_token_list = ['moveLeft', 'moveRight']

# NOTE: keep the environment names same as config.env_name in config/arguments.py
envs = ['karel', 'CartPoleDiscrete-v0']
obv_token_api_dict = {'karel': karel_obv_token_api_dict, 'CartPoleDiscrete-v0': cartpole_obv_token_api_dict}
action_token_list = {'karel': karel_action_token_list, 'CartPoleDiscrete-v0': cartpole_action_token_list}


class DSLData(object):

    def __init__(self, environment='karel'):
        if environment == 'karel':
            self.tokens = [
                'DEF', 'RUN', 'M_LBRACE', 'M_RBRACE',
                'MOVE', 'TURN_RIGHT', 'TURN_LEFT',
                'PICK_MARKER', 'PUT_MARKER',
                'R_LBRACE', 'R_RBRACE',
                'INT',  # 'NEWLINE', 'SEMI',
                'REPEAT',
                'C_LBRACE', 'C_RBRACE',
                'I_LBRACE', 'I_RBRACE', 'E_LBRACE', 'E_RBRACE',
                'IF', 'IFELSE', 'ELSE',
                'FRONT_IS_CLEAR', 'LEFT_IS_CLEAR', 'RIGHT_IS_CLEAR',
                'MARKERS_PRESENT', 'NO_MARKERS_PRESENT',
                'NOT',
                'W_LBRACE', 'W_RBRACE',
                'WHILE',
            ]

            self.t_FRONT_IS_CLEAR = 'frontIsClear'
            self.t_LEFT_IS_CLEAR = 'leftIsClear'
            self.t_RIGHT_IS_CLEAR = 'rightIsClear'
            self.t_MARKERS_PRESENT = 'markersPresent'
            self.t_NO_MARKERS_PRESENT = 'noMarkersPresent'

            self.conditional_functions = [
                self.t_FRONT_IS_CLEAR, self.t_LEFT_IS_CLEAR, self.t_RIGHT_IS_CLEAR,
                self.t_MARKERS_PRESENT, self.t_NO_MARKERS_PRESENT,
            ]

            self.conditional_functions_dict = {
                self.t_FRONT_IS_CLEAR: lambda x: x.front_is_clear, self.t_LEFT_IS_CLEAR: lambda x: x.left_is_clear,
                self.t_RIGHT_IS_CLEAR: lambda x: x.right_is_clear, self.t_MARKERS_PRESENT: lambda x: x.marker_present,
                self.t_NO_MARKERS_PRESENT: lambda x: x.no_marker_present,
            }

            self.t_MOVE = 'move'
            self.t_TURN_RIGHT = 'turnRight'
            self.t_TURN_LEFT = 'turnLeft'
            self.t_PICK_MARKER = 'pickMarker'
            self.t_PUT_MARKER = 'putMarker'

            self.action_functions = [
                self.t_MOVE,
                self.t_TURN_LEFT, self.t_TURN_RIGHT,
                self.t_PICK_MARKER, self.t_PUT_MARKER,
            ]

            # This dictionary is used for YACC parser in dsl_base.py for grammar
            # It seems that yacc parser uses these docstrings to learn the perception and action
            # tokens. Check karel_env/dsl/third-party/yacc.py:3139
            self.dynamic_docstring = {
                'cond_without_not': '''cond_without_not : FRONT_IS_CLEAR
                | LEFT_IS_CLEAR
                | RIGHT_IS_CLEAR
                | MARKERS_PRESENT
                | NO_MARKERS_PRESENT
                ''',
                'action': '''action : MOVE
                    | TURN_RIGHT
                    | TURN_LEFT
                    | PICK_MARKER
                    | PUT_MARKER
                    '''
            }
        else:
            self.tokens = [
                'DEF', 'RUN', 'M_LBRACE', 'M_RBRACE',
                'RIGHT', 'LEFT',
                'R_LBRACE', 'R_RBRACE',
                'INT',  # 'NEWLINE', 'SEMI',
                'REPEAT',
                'C_LBRACE', 'C_RBRACE',
                'I_LBRACE', 'I_RBRACE', 'E_LBRACE', 'E_RBRACE',
                'IF', 'IFELSE', 'ELSE',
                'POLE_LEFT', 'POLE_CENTER', 'POLE_RIGHT',
                'CART_LEFT', 'CART_CENTER', 'CART_RIGHT',
                'POLE_FAST',  'POLE_MED', 'POLE_SLOW',
                'CART_FAST', 'CART_MED', 'CART_SLOW',
                'NOT',
                'W_LBRACE', 'W_RBRACE',
                'WHILE',
            ]

            self.t_POLE_LEFT = 'PolePosLeft'
            self.t_POLE_CENTER = 'PolePosCenter'
            self.t_POLE_RIGHT = 'PolePosRight'
            self.t_CART_LEFT = 'CartPosLeft'
            self.t_CART_CENTER = 'CartPosCenter'
            self.t_CART_RIGHT = 'CartPosRight'
            self.t_POLE_FAST = 'PoleVelFast'
            self.t_POLE_MED = 'PoleVelMed'
            self.t_POLE_SLOW = 'PoleVelSlow'
            self.t_CART_FAST = 'CartVelFast'
            self.t_CART_MED = 'CartVelMed'
            self.t_CART_SLOW = 'CartVelSlow'

            self.conditional_functions = [
                self.t_POLE_LEFT, self.t_POLE_CENTER, self.t_POLE_RIGHT,
                self.t_CART_LEFT, self.t_CART_CENTER, self.t_CART_RIGHT,
                self.t_POLE_FAST, self.t_POLE_MED, self.t_POLE_SLOW,
                self.t_CART_FAST, self.t_CART_MED, self.t_CART_SLOW,

            ]

            self.conditional_functions_dict = {
                self.t_POLE_LEFT: lambda x: x.pole_pos_left,
                self.t_POLE_CENTER: lambda x: x.pole_pos_center,
                self.t_POLE_RIGHT: lambda x: x.pole_pos_right,
                self.t_CART_LEFT: lambda x: x.cart_pos_left,
                self.t_CART_CENTER: lambda x: x.cart_pos_center,
                self.t_CART_RIGHT: lambda x: x.cart_pos_right,
                self.t_POLE_FAST: lambda x: x.pole_vel_fast,
                self.t_POLE_MED: lambda x: x.pole_vel_med,
                self.t_POLE_SLOW: lambda x: x.pole_vel_slow,
                self.t_CART_FAST: lambda x: x.cart_vel_fast,
                self.t_CART_MED: lambda x: x.cart_vel_med,
                self.t_CART_SLOW: lambda x: x.cart_vel_slow,
            }

            self.t_LEFT = 'moveLeft'
            self.t_RIGHT = 'moveRight'

            self.action_functions = [
                self.t_LEFT, self.t_RIGHT,
            ]

            # This dictionary is used for YACC parser in dsl_base.py for grammar
            # It seems that yacc parser uses these docstrings to learn the perception and action
            # tokens. Check karel_env/dsl/third-party/yacc.py:3139
            self.dynamic_docstring = {
                'cond_without_not': '''cond_without_not : POLE_LEFT
                | POLE_CENTER
                | POLE_RIGHT
                | CART_LEFT
                | CART_CENTER
                | CART_RIGHT POLE_FAST
                | POLE_MED
                | POLE_SLOW
                | CART_FAST
                | CART_MED
                | CART_SLOW
                ''',
                'action': '''action : RIGHT
                | LEFT
                '''
            }

