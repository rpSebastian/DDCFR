# from PokerRL._.CrayonWrapper import CrayonWrapper
import argparse
import time
from pathlib import Path

from ddcfr.utils.logger import Logger
from PokerRL.cfr import DDCFR
from PokerRL.cfr.cfr_model import A2CPolicy, CFRModel
from PokerRL.game import bet_sets
from PokerRL.game.games import (
    BigLeduc,
    DiscretizedNLHoldemSubGame3,
    DiscretizedNLHoldemSubGame4,
)
from PokerRL.rl.base_cls.workers.ChiefBase import ChiefBase

parser = argparse.ArgumentParser(description="Run CFR in games")
parser.add_argument("--run_id", type=int, help="run_id")
parser.add_argument("--model_id", type=int, help="model_id")
parser.add_argument("--model_name", type=str, help="model_name", default=None)
parser.add_argument("--model_version", type=str, help="model_version", default="es")

parser.add_argument("--iters", type=int, help="iterations")
parser.add_argument("--game", type=str, help="game names")
parser.add_argument("--save", action="store_true", default=False, help="game names")
parser.add_argument("--save_dir", type=str, default="results", help="save dir")

args = parser.parse_args()

n_iterations = args.iters

game_name = args.game
model_name = args.model_name

game_dict = {
    "Subgame3": DiscretizedNLHoldemSubGame3,
    "Subgame4": DiscretizedNLHoldemSubGame4,
    "BigLeduc": BigLeduc,
}

if args.model_name is None:
    run_path = (
        Path(__file__).absolute().parents[2]
        / "logs"
        / args.model_version
        / str(args.run_id)
    )
    model_path = run_path / "model" / "cfr_model_{}.pkl".format(args.model_id)
    algo_name = "DDCFR_{}_{}_{}".format(args.model_version, args.run_id, args.model_id)
else:
    model_path = (
        Path(__file__).absolute().parents[2] / "models" / "{}.pkl".format(model_name)
    )
    algo_name = "DDCFR_{}_{}".format(args.model_version, model_name)

name = algo_name
if args.model_version == "es":
    model = CFRModel()
else:
    model = A2CPolicy()
model.load(model_path)
folder = Path(__file__).absolute().parents[2] / args.save_dir / algo_name / game_name
folder.mkdir(parents=True, exist_ok=True)
if args.save:
    logger = Logger(writer_strings=["csv"], folder=folder)
else:
    logger = Logger(writer_strings=["stdout"])


# Passing None for t_prof will is enough for ChiefBase. We only use it to log; This CFR impl is not distributed.
chief = ChiefBase(t_prof=None)
cfr = DDCFR(
    total_iterations=args.iters,
    model=model,
    name=name,
    game_cls=game_dict[game_name],
    agent_bet_set=bet_sets.B_3,
    other_agent_bet_set=bet_sets.B_2,
    chief_handle=chief,
    logger=logger,
    game_name=game_name,
)
c = []
print(time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime()), "start run: ", name)

for step in range(1, n_iterations + 1):
    cfr.iteration()
    conv = cfr.expl / 1000
    print(
        time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime()),
        "Iteration: {:>4d} exp: {:>12.5f}".format(step, conv),
    )
    if step == 1:
        logger.record(f"{game_name}/conv", conv)
        logger.record(f"{game_name}/iters", 0)
        logger.dump(step=0)
    logger.record(f"{game_name}/conv", conv)
    logger.record(f"{game_name}/iters", step)
    logger.dump(step=step)
