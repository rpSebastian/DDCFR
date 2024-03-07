# from PokerRL._.CrayonWrapper import CrayonWrapper
import argparse
import time
from pathlib import Path

from ddcfr.utils.logger import Logger
from PokerRL.cfr import DCFR, CFRPlus, LinearCFR, VanillaCFR
from PokerRL.game import bet_sets
from PokerRL.game.games import (
    BigLeduc,
    DiscretizedNLHoldemSubGame3,
    DiscretizedNLHoldemSubGame4,
)
from PokerRL.rl.base_cls.workers.ChiefBase import ChiefBase

parser = argparse.ArgumentParser(description="Run CFR in games")
parser.add_argument("--iters", type=int, help="iterations")
parser.add_argument("--algo", type=str, help="algo names")
parser.add_argument("--game", type=str, help="game names")

parser.add_argument("--save", action="store_true", default=False, help="game names")

args = parser.parse_args()

n_iterations = args.iters
algo_name = args.algo
game_name = args.game
name = "{}_{}".format(algo_name, game_name)

algo_dict = {
    "CFR": VanillaCFR,
    "CFRPlus": CFRPlus,
    "LinearCFR": LinearCFR,
    "DCFR": DCFR,
}
game_dict = {
    "Subgame3": DiscretizedNLHoldemSubGame3,
    "Subgame4": DiscretizedNLHoldemSubGame4,
    "BigLeduc": BigLeduc,
}

# Passing None for t_prof will is enough for ChiefBase. We only use it to log; This CFR impl is not distributed.
chief = ChiefBase(t_prof=None)
cfr = algo_dict[algo_name](
    name=name,
    game_cls=game_dict[game_name],
    agent_bet_set=bet_sets.B_3,
    other_agent_bet_set=bet_sets.B_2,
    chief_handle=chief,
)
c = []
print(time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime()), "start run: ", name)

steps = []
convs = []

folder = Path(__file__).absolute().parents[2] / "results" / algo_name / game_name
folder.mkdir(parents=True, exist_ok=True)
if args.save:
    logger = Logger(writer_strings=["csv"], folder=folder)
else:
    logger = Logger(writer_strings=[])
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
