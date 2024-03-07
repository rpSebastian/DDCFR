from ddcfr.cfr.cfr_env import make_cfr_env
from ddcfr.es.cfr_model import CFRModel
from ddcfr.es.worker import VecWorker, Worker
from ddcfr.game.game_config import GameConfig


class CFRModelEvaluator(Worker):
    def evaluate_cfr_model(
        self,
        game_config: GameConfig,
        model: CFRModel,
    ):
        env = make_cfr_env(game_config, game_config.iterations)
        s = env.reset()
        while True:
            a = model(s)
            ns, r, d, i = env.step(a)
            s = ns
            if d:
                break
        conv = i["conv"]
        result = dict(status="succ", conv=conv)
        return result

    def run(self, task):
        game_config = task["game_config"]
        model = task["model"]
        result = self.evaluate_cfr_model(game_config, model)
        result["game_config"] = game_config
        if "model_id" in task:
            result["model_id"] = task["model_id"]
        return result


class CFRModelVecEvaluator(VecWorker):
    def __init__(self, num_evaluators, verbose=False):
        super().__init__(num_evaluators, CFRModelEvaluator, verbose=verbose)

    def eval_models(self, models, game_configs):
        tasks = []
        model_info = {}
        for model in models:
            model_id = id(model)
            model_info[model_id] = {"total_score": 0, "total_weight": 0, "convs": {}}
            for game_config in game_configs:
                task = {
                    "model": model,
                    "game_config": game_config,
                    "model_id": model_id,
                }
                tasks.append(task)
        if self.verbose:
            print("start executing {} tasks".format(len(tasks)))

        results = self.execute_tasks(tasks)
        if self.verbose:
            print("finish executing {} tasks".format(len(results)))

        for result in results:
            game_config = result["game_config"]
            conv = result["conv"]
            score = game_config.conv_to_score(conv, game_config.iterations)
            model_id = result["model_id"]
            info = model_info[model_id]
            info["total_score"] += score * game_config.weight
            info["total_weight"] += game_config.weight
            info["convs"][game_config.name] = conv

        for model in models:
            model_id = id(model)
            info = model_info[model_id]
            score = info["total_score"] / info["total_weight"]
            model.score = score
            model.convs = info["convs"]
        return models

    def eval_model_async(self, model_id, model, game_configs):
        for game_config in game_configs:
            task = dict(model_id=model_id, model=model, game_config=game_config)
            self.add_task(task)

    def get_eval_result(self):
        return self.get_result()
