class LearningRateDecay(object):
    def __init__(self, v, nvalues, lr_decay_method):
        self.n = 0.
        self.v = v
        self.nvalues = nvalues

        def constant(p):
            return 1

        def linear(p):
            return 1 - p

        lr_decay_methods = {
            'linear': linear,
            'constant': constant
        }

        self.decay = lr_decay_methods[lr_decay_method]

    def value(self):
        current_value = self.v * self.decay(self.n / self.nvalues)
        self.n += 1.
        return current_value

    def get_value_for_steps(self, steps):
        return self.v * self.decay(steps / self.nvalues)

class LinearSchedule(object):
    def __init__(self, schedule_timesteps, final_p, initial_p=1.0):
        """Linear interpolation between initial_p and final_p over
        schedule_timesteps. After this many timesteps pass final_p is
        returned.
        Parameters
        ----------
        schedule_timesteps: int
            Number of timesteps for which to linearly anneal initial_p
            to final_p
        initial_p: float
            initial output value
        final_p: float
            final output value
        """
        self.schedule_timesteps = schedule_timesteps
        self.final_p = final_p
        self.initial_p = initial_p

    def value(self, t):
        """See Schedule.value"""
        fraction = min(float(t) / self.schedule_timesteps, 1.0)
        return self.initial_p + fraction * (self.final_p - self.initial_p)