import simpy, random, copy
import pandas as pd
import numpy as np


class Job:
    def __init__(self, name=None, processing_time=0, feature=0):
        self.name = name
        self.processing_time = processing_time
        self.feature = feature

        self.due_date = 0
        self.completion_time = 0


class Source:
    def __init__(self, env, j_list, lmbda, ddt, routing, monitor):
        self.env = env
        self.name = "Source"
        self.monitor = monitor
        self.j_list = j_list  # job list
        self.lmbda = lmbda  # IAT 평균
        self.ddt = ddt  # due date tightness
        self.routing = routing  # routing class
        self.monitor = monitor

        self.created = 0
        env.process(self.run())

    def run(self):
        while len(self.j_list):
            created_job = self.j_list.pop(0)
            created_job.due_date = self.env.now + self.ddt * created_job.processing_time
            self.monitor.record(time=self.env.now, job=created_job.name, event="Created", class_name=self.name,
                                memo=created_job.due_date)

            self.routing.queue.put(created_job)
            self.monitor.record(time=self.env.now, job=created_job.name, event="Move to Routing", class_name=self.name,
                                queue=len(self.routing.queue.items))

            self.created += 1

            if self.routing.is_queue_event:
                self.routing.queue_event.succeed()
            self.routing.queue_list = copy.deepcopy([job for job in self.routing.queue.items])

            iat = np.random.exponential(self.lmbda)
            yield self.env.timeout(iat)


class Routing:
    def __init__(self, env, model, monitor, end_num, weight=None):
        self.env = env
        self.name = "Routing"
        self.model = model
        self.monitor = monitor
        self.end_num = end_num
        self.weight = weight  # ATCS, COVERT parameter

        self.indicator = False
        self.decision = None

        self.job = None
        self.machine = None

        self.queue = simpy.FilterStore(env)
        self.waiting_event = simpy.Store(env)  # 각 Machine에서 발생시킨 routing 요청
        self.queue_list = list()  # 현재 routing.queue에 있는 job list

        self.setup = 0
        self.created = 0
        self.queue_event = self.env.event()  # Machine에서 요청 시 job이 없을 때, routing class를 holding 하는 이벤트
        self.is_queue_event = False  # self.queue.event가 활성화 됐는지
        env.process(self.run())

    def run(self):
        while self.created < self.end_num:
            if len(self.queue.items) == 0:
                self.is_queue_event = True
                yield self.queue_event
                self.is_queue_event = False
                self.queue_event = self.env.event()

            decision = yield self.waiting_event.get()
            self.decision = decision[0]
            self.machine = decision[1]  # routing을 요청한 machine 이름
            self.indicator = True   # agent로부터 action을 받아와야한다는 표시

            routing_rule = yield self.decision  # action 받아옴
            self.routing_rule = routing_rule
            self.decision = None
            self.monitor.record(time=self.env.now, event="Routing Start", class_name=self.name,
                                queue=len(self.queue.items), memo=routing_rule)
            self.setup = 0

            if routing_rule == "SSPT":
                next_job = yield self.env.process(self._sspt())
            elif routing_rule == "ATCS":
                next_job = yield self.env.process(self._atcs())
            elif routing_rule == "MDD":
                next_job = yield self.env.process(self._mdd())
            elif routing_rule == "COVERT":
                next_job = yield self.env.process(self._covert())
            else:
                next_job = None

            self.created += 1
            self.setup = abs(next_job.feature - self.model[self.machine].setup)
            self.monitor.record(time=self.env.now, job=next_job.name, event="Move to Machine", class_name=self.name,
                                queue=len(self.queue.items), memo=self.setup)
            self.model[self.machine].queue.put(next_job)

    def _sspt(self):
        min_sspt = 1e10
        min_job_list = list()

        job_list = copy.deepcopy(self.queue_list)
        calling_machine = self.model[self.machine]

        for job in job_list:
            setup_time = abs(job.feature - calling_machine.setup)
            expected_pt = setup_time + job.processing_time
            if expected_pt < min_sspt:
                min_job_list = [job.name]
                min_sspt = expected_pt
            elif expected_pt == min_sspt:
                min_job_list.append(job.name)

        min_job_name = np.random.choice(min_job_list)
        for i in range(len(self.queue_list)):
            if self.queue_list[i].name == min_job_name:
                del self.queue_list[i]
                break
        next_job = yield self.queue.get(lambda x: x.name == min_job_name)

        return next_job

    def _atcs(self):
        max_atcs = -1
        max_job_list = list()

        job_list = copy.deepcopy(self.queue_list)
        calling_machine = self.model[self.machine]
        idx_list = list()
        k1 = self.weight["ATCS"][0]  # 6.7(job = 100)
        k2 = self.weight["ATCS"][1]  # 1.3(job = 100)

        p_avg = np.mean([job.processing_time for job in job_list]) if len(job_list) > 0 else 0
        s_avg = np.mean([abs(job.feature - calling_machine.setup) for job in job_list]) if len(job_list) > 0 else 0

        for job in job_list:
            first_term = np.exp(-max(job.due_date - job.processing_time - self.env.now, 0) / (k1 * p_avg)) if p_avg > 0 else 0
            second_term = np.exp(-abs(job.feature - calling_machine.setup) / (k2 * s_avg)) if s_avg > 0 else 0
            atcs = (1 / job.processing_time) * first_term * second_term
            if atcs > max_atcs:
                max_job_list = [job.name]
                max_atcs = atcs
            elif atcs == max_atcs:
                max_job_list.append(job.name)

        max_job_name = np.random.choice(max_job_list)
        for i in range(len(self.queue_list)):
            if self.queue_list[i].name == max_job_name:
                del self.queue_list[i]
                break
        next_job = yield self.queue.get(lambda x: x.name == max_job_name)

        return next_job

    def _mdd(self):
        min_mdd = 1e10
        min_job_list = list()

        job_list = copy.deepcopy(self.queue_list)

        for job in job_list:
            mdd = max(job.due_date, self.env.now + job.processing_time)
            if mdd < min_mdd:
                min_job_list = [job.name]
                min_mdd = mdd
            elif mdd == min_mdd:
                min_job_list.append(job.name)

        min_job_name = np.random.choice(min_job_list)
        for i in range(len(self.queue_list)):
            if self.queue_list[i].name == min_job_name:
                del self.queue_list[i]
                break
        next_job = yield self.queue.get(lambda x: x.name == min_job_name)

        return next_job

    def _covert(self):
        max_covert = -1
        max_job_list = list()

        job_list = copy.deepcopy(self.queue_list)
        k = self.weight["COVERT"]

        for job in job_list:
            second_term = max(0, 1 - (
                        max(0, job.due_date - job.processing_time - self.env.now) / (k * job.processing_time)))
            covert = (1 / job.processing_time) * second_term

            if covert > max_covert:
                max_job_list = [job.name]
                max_covert = covert
            elif covert == max_covert:
                max_job_list.append(job.name)

        max_job_name = np.random.choice(max_job_list)
        for i in range(len(self.queue_list)):
            if self.queue_list[i].name == max_job_name:
                del self.queue_list[i]
                break
        next_job = yield self.queue.get(lambda x: x.name == max_job_name)

        return next_job

    def reset(self):
        self.indicator = False
        self.decision = None

        self.job = None
        self.machine = None
        self.routing_rule = None

        self.queue.items = list()
        self.waiting_event.items = list()
        self.queue_list = list()

        self.setup = 0
        self.created = 0
        self.queue_event = self.env.event()
        self.is_queue_event = False


class Process:
    def __init__(self, env, name, routing, sink, monitor, pt_var):
        self.env = env
        self.name = name
        self.routing = routing
        self.sink = sink
        self.monitor = monitor
        self.pt_var = pt_var
        self.setup = random.randint(0, 5)

        self.queue = simpy.Store(env)
        self.job = None
        self.idle = True
        self.start_time = 0.0
        self.expected_finish_time = 0.0

        env.process(self.run())

    def run(self):
        while True:
            self.routing.waiting_event.put([self.env.event(), self.name])
            self.monitor.record(time=self.env.now, event="Routing Request", class_name=self.name,
                                queue=len(self.routing.queue.items))

            job = yield self.queue.get()
            self.job = job
            self.monitor.record(time=self.env.now, event="Get the Job", job=job.name, class_name=self.name)
            self.idle = False
            self.start_time = self.env.now
            setup_time = abs(self.setup - job.feature)
            self.expected_finish_time = self.env.now + setup_time + job.processing_time

            if setup_time > 0:
                self.monitor.record(time=self.env.now, job=job.name, event="Setup", class_name=self.name,
                                    memo=setup_time)
                self.monitor.setup += setup_time
                yield self.env.timeout(setup_time)
                self.setup = job.feature

            pt_var = np.random.uniform(low=1 - self.pt_var, high=1 + self.pt_var)
            processing_time = pt_var * job.processing_time
            self.monitor.record(time=self.env.now, event="Work Start", job=job.name, class_name=self.name)
            yield self.env.timeout(processing_time)
            self.monitor.record(time=self.env.now, event="Work Finish", job=job.name, class_name=self.name)

            self.sink.sink(job)
            self.job = None
            self.idle = True

            if len(self.routing.queue.items) == 0 and self.routing.created == self.routing.end_num:
                break

    def reset(self):
        self.setup = random.randint(0, 5)
        self.queue.items = list()
        self.job = None
        self.idle = True
        self.start_time = 0.0
        self.expected_finish_time = 0.0


class Sink:
    def __init__(self, env, monitor):
        self.env = env
        self.monitor = monitor

        self.completed = 0
        self.makespan = 0.0
        self.tardiness = list()

    def sink(self, job=None):
        self.completed += 1
        tardiness = max(self.env.now - job.due_date, 0)
        self.monitor.tardiness += tardiness
        self.tardiness.append(tardiness)
        self.makespan = self.env.now
        self.monitor.record(time=self.env.now, job=job.name, event="Completed", class_name="Sink", memo=tardiness)

    def reset(self):
        self.completed = 0
        self.makespan = 0.0
        self.tardiness = list()


class Monitor:
    def __init__(self):
        self.time = list()
        self.event = list()
        self.job = list()
        self.class_name = list()
        self.queue = list()
        self.memo = list()

        self.tardiness = 0
        self.setup = 0

    def record(self, time=None, event=None, job=None, class_name=None, queue=None, memo=None):
        self.time.append(round(time, 2))
        self.event.append(event)
        self.job.append(job)
        self.class_name.append(class_name)
        self.queue.append(queue)
        self.memo.append(memo)

    def reset(self):
        self.time = list()
        self.event = list()
        self.job = list()
        self.class_name = list()
        self.queue = list()
        self.memo = list()

        self.tardiness = 0
        self.setup = 0

    def get_logs(self, file_path=None):
        event_tracer = pd.DataFrame(columns=["Time", "Event", "Job", "Class", "Queue", "Memo"])
        event_tracer["Time"] = self.time
        event_tracer["Event"] = self.event
        event_tracer["Job"] = self.job
        event_tracer["Class"] = self.class_name
        event_tracer["Queue"] = self.queue
        event_tracer["Memo"] = self.memo

        if file_path is not None:
            event_tracer.to_csv(file_path, encoding='utf-8-sig', index=False)

        return event_tracer

