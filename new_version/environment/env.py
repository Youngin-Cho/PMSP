import os
import numpy as np

from new_version.environment.simulation import *


class PMSP:
    def __init__(self, num_job=200, num_m=5, episode=1, test_sample=None, reward_weight=None,
                 ddt=None, pt_var=None, is_train=True):
        self.num_job = num_job  # scheduling 대상 job의 수
        self.num_m = num_m  # parallel machine 수
        self.episode = episode
        self.test_sample = test_sample
        self.reward_weight = reward_weight if reward_weight is not None else [0.5, 0.5]
        self.ddt = ddt
        self.pt_var = pt_var
        self.is_train = is_train

        self.state_size = 5
        self.action_size = 1

        self.time = 0
        self.previous_time_step = 0

        self.time_list = list()
        self.tardiness_list = list()
        self.setup_list = list()

        self.done = False
        self.time = 0
        self.reward_setup = 0
        self.reward_tard = 0

        self.sim_env, self.model, self.source, self.sink, self.routing, self.monitor = self._modeling()

    def step(self, action):
        done = False
        self.previous_time_step = self.sim_env.now

        job_name = "Job {0}".format(action)
        self.routing.decision.succeed(job_name)
        self.routing.indicator = False

        while True:
            if self.routing.indicator:
                if self.sim_env.now != self.time:
                    self.time = self.sim_env.now
                break

            if self.sink.completed == self.num_job:
                done = True
                self.sim_env.run()
                break

            self.sim_env.step()

        reward = self._calculate_reward()
        next_state, mask = self._get_state()

        return next_state, reward, done, mask

    def _modeling(self):
        env = simpy.Environment()

        monitor = Monitor()
        monitor.reset()

        iat = 15 / self.num_m

        model = dict()
        job_list = list()
        for j in range(self.num_job):
            job_name = "Job {0}".format(j)
            processing_time = np.random.uniform(10, 20) if self.test_sample is None else self.test_sample['processing_time'][j]
            feature = random.randint(0, 5) if self.test_sample is None else self.test_sample['feature'][j]
            job_list.append(Job(name=job_name, id=j, processing_time=processing_time, feature=feature))

        routing = Routing(env, model, monitor, end_num=self.num_job)
        routing.reset()

        sink = Sink(env, monitor)
        sink.reset()

        source = Source(env, job_list, iat, self.ddt, routing, monitor)

        for m in range(self.num_m):
            machine_name = "Machine {0}".format(m)
            model[machine_name] = Process(env, machine_name, m, routing, sink, monitor, pt_var=self.pt_var)
            model[machine_name].reset()

        return env, model, source, sink, routing, monitor

    def reset(self):
        if self.is_train:
            self.pt_var = np.random.uniform(low=0.1, high=0.5)
            self.ddt = np.random.uniform(low=0.8, high=1.2)

        self.sim_env, self.model, self.source, self.sink, self.routing, self.monitor = self._modeling()
        self.done = False
        self.reward_setup = 0
        self.reward_tard = 0
        self.monitor.reset()

        while True:
            # Check whether there is any decision time step
            if self.routing.indicator:
                break

            self.sim_env.step()

        initial_state, mask = self._get_state()

        return initial_state, mask

    def _get_state(self):
        state = np.zeros((self.num_job, self.state_size))
        mask = np.zeros(self.num_job)

        target_line = self.model[self.routing.machine]
        input_queue = copy.deepcopy(list(self.routing.queue.items))
        for job in input_queue:
            state[job.id, 0] = self.sim_env.now - job.arrival_time
            state[job.id, 1] = job.processing_time
            state[job.id, 2] = (job.due_date - self.sim_env.now) / job.processing_time
            state[job.id, 3] = job.feature
            state[job.id, 4] = abs(job.feature - target_line.setup)
            mask[job.id] = 1.0

        state = (state - np.mean(state, axis=0)) / (np.std(state, axis=0) + 0.00001)

        return state, mask

    def _calculate_reward(self):
        reward_1 = - self.routing.setup / 5
        self.reward_setup -= self.routing.setup / 5

        reward_2 = 0.0
        if len(self.sink.tardiness) > 0:
            for tardiness in self.sink.tardiness:
                reward_2 += np.exp(-tardiness) - 1

        # reward_2 = np.exp(-self.sink.tardiness) - 1
        self.reward_tard += reward_2

        reward = reward_1 * self.reward_weight[1] + reward_2 * self.reward_weight[0]
        self.routing.setup = 0
        self.sink.tardiness = list()
        return reward

    def get_logs(self, path=None):
        log = self.monitor.get_logs(path)
        return log

