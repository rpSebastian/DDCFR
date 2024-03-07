import pytest
import torch as th

from ddcfr.cfr.cfr_env import make_cfr_env
from ddcfr.es.cfr_eval import CFRModelEvaluator, CFRModelVecEvaluator
from ddcfr.es.cfr_model import CFRModel
from ddcfr.game import KuhnPoker


def test_cfr_model_evaluator():
    evaluator = CFRModelEvaluator(index=0)
    env = make_cfr_env("KuhnPoker")
    cfr_model = CFRModel(env.observation_space, env.action_space, th.device("cpu"))
    task = dict(game_config=KuhnPoker(5), model=cfr_model)
    result = evaluator.run(task)
    assert result["status"] == "succ"


@pytest.mark.skip(reason="skip test using ray")
def test_cfr_model_vec_evaluator():
    evaluator = CFRModelVecEvaluator(num_evaluators=10)
    env = make_cfr_env("KuhnPoker")
    models = [
        CFRModel(env.observation_space, env.action_space, th.device("cpu"))
        for _ in range(2)
    ]
    game_configs = [
        KuhnPoker(5),
        KuhnPoker(5),
    ]
    models = evaluator.eval_models(models, game_configs)
    for model in models:
        print(model.score)


@pytest.mark.skip(reason="skip test using ray")
def test_cfr_model_vec_evaluator_aysnc():
    evaluator = CFRModelVecEvaluator(num_evaluators=2)
    env = make_cfr_env("KuhnPoker")
    model = CFRModel(env.observation_space, env.action_space, th.device("cpu"))
    game_configs = [
        KuhnPoker(5),
        KuhnPoker(5),
    ]
    evaluator.eval_model_async("model_5", model, game_configs)
    while True:
        result = evaluator.get_eval_result()
        if result is not None:
            break
    print(result)
