"""misc utils for training neural networks"""


class HyperParameterScheduler(object):
    def __init__(self, initial_val, num_updates, final_val=None, func='linear', gamma=0.999):
        """ Initialize HyperParameter Scheduler class

        :param initial_val: initial value of the hyper-parameter
        :param num_updates: total number of updates for hyper-parameter
        :param final_val: final value of the hyper-parameter, if None then decay rate is fixed (0.999 for exponential 0
        for linear)
        :param func: decay type ['exponential', linear']
        :param gamma: fixed decay rate for exponential decay (if final value is given then this gamma is ignored)
        """
        self.initial_val = initial_val
        self.total_num_epoch = num_updates
        self.final_val = final_val
        self.cur_hp = self.initial_val
        self.cur_step = 0

        if final_val is not None:
            assert final_val >= 0, 'final value should be positive'

        if func == "linear":
            self.hp_lambda = self.linear_scheduler
        elif func == "exponential":
            self.hp_lambda = self.exponential_scheduler
            if initial_val == final_val:
                self.gamma = 1
            else:
                self.gamma = pow(final_val / initial_val, 1 / self.total_num_epoch) if final_val is not None else gamma
        else:
            raise NotImplementedError('scheduler not implemented')

    def linear_scheduler(self, epoch):
        if self.final_val is not None:
            return (self.final_val - self.initial_val)*(epoch/self.total_num_epoch) + self.initial_val
        else:
            return self.initial_val - (self.initial_val * (epoch / self.total_num_epoch))

    def exponential_scheduler(self, epoch):
        return self.initial_val * (self.gamma ** epoch)

    def step(self, epoch=None):
        assert self.cur_step <= self.total_num_epoch, "scheduler step shouldn't be larger than total steps"
        if epoch is None:
            epoch = self.cur_step
        self.cur_hp = self.hp_lambda(epoch)
        self.cur_step += 1
        return self.cur_hp

    @property
    def get_value(self):
        return self.cur_hp