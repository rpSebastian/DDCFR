import time

import numpy as np
import ray

from ddcfr.es.worker import GroupVecWorker, VecWorker, Worker


class DiverContainer(Worker):
    @ray.remote
    def run(task):
        a = task["a"]
        b = task["b"]
        result = {
            "a": a,
            "b": b,
        }
        time.sleep(a)
        out = a / b
        result["out"] = out
        return result


class Diver(Worker):
    def run(self, task):
        result = {}
        a = task["a"]
        b = task["b"]
        result["a"] = a
        result["b"] = b
        out = a / b
        result["out"] = out
        return result


def test_run():
    diver = Diver(1)
    result = diver.execute_task(dict(a=1, b=0))
    assert result["state"] == "fail"

    result = diver.execute_task(dict(a=1, b=5))
    assert result["out"] == 1 / 5


def atest_parallel_run():
    ray.init()
    vec_worker = VecWorker(3, Diver)
    for i in range(10):
        a = np.random.randint(low=0, high=100)
        b = np.random.randint(low=0, high=100)
        vec_worker.add_task(dict(a=a, b=b))
    for i in range(20):
        time.sleep(1)
        result = vec_worker.get_result()
    ray.shutdown()


def atest_parallel_run_sync():
    ray.init()
    vec_worker = VecWorker(2, Diver)
    tasks = []
    for i in range(10):
        a = np.random.randint(low=0, high=100)
        b = np.random.randint(low=0, high=100)
        tasks.append(dict(a=a, b=b))
    results = vec_worker.execute_tasks(tasks)
    for result in results:
        print(result["a"], result["b"], result["out"])
    ray.shutdown()


def atest_group_run():
    ray.init()
    group_vec_worker = GroupVecWorker(10, DiverContainer)
    group_vec_worker.add_tasks([dict(a=3, b=4), dict(a=3, b=7), dict(a=1, b=1)])
    group_vec_worker.add_tasks([dict(a=3, b=4), dict(a=5, b=0)])
    group_vec_worker.add_tasks([dict(a=1, b=0), dict(a=3, b=4), dict(a=3, b=0)])
    group_vec_worker.add_tasks(
        [
            dict(a=1, b=4),
            dict(a=3, b=0),
            dict(a=2, b=4),
        ]
    )

    for i in range(20):
        time.sleep(1)
        print(group_vec_worker.info())
        while True:
            result = group_vec_worker.get_result()
            if result is not None:
                print(result)
            else:
                break
    ray.shutdown()
