from pathlib import Path

import numpy as np
import pandas as pd
import pyspiel


class GameConfig:
    def __init__(
        self,
        iterations=1000,
        weight=1,
        game_name=None,
        transform=False,
    ):
        self.game_name = game_name
        self.params = {}
        self.transform = transform
        self.iterations = iterations
        self.weight = weight
        self.name = self.__class__.__name__
        self.load_baseline_conv()

    def load_baseline_conv(self):
        self.cfr_df = self.load_algo_conv("CFR")
        self.dcfr_df = self.load_algo_conv("DCFR")

    def load_algo_conv(self, algo_name):
        csv_file = (
            Path(__file__).absolute().parents[2]
            / "results"
            / algo_name
            / self.name
            / "data.csv"
        )
        df = None
        if csv_file.exists():
            df = pd.read_csv(csv_file, index_col="step")
        return df

    def conv_to_score(self, conv, iters):
        if self.cfr_df is None:
            raise ValueError(f"No baseline scores for {self.name}")
        conv_name = f"{self.name}/conv"
        cfr_conv = self.cfr_df.loc[iters, conv_name]
        dcfr_conv = self.dcfr_df.loc[iters, conv_name]
        log_conv = np.log(conv) / np.log(10)
        cfr_log_conv = np.log(cfr_conv) / np.log(10)
        dcfr_log_conv = np.log(dcfr_conv) / np.log(10)
        score = (cfr_log_conv - log_conv) / (cfr_log_conv - dcfr_log_conv)
        return score

    def load_game(self):
        params = {}
        for p, v in self.params.items():
            if p == "filename":
                v = str(Path(__file__).absolute().parents[2] / v)
            params[p] = v
        game = pyspiel.load_game(self.game_name, params)
        if self.transform:
            game = pyspiel.convert_to_turn_based(game)
        self.num_nodes = 0
        self.calc_nodes(game.new_initial_state())
        return game

    def __repr__(self):
        return "{}({})".format(self.name, self.iterations)

    def calc_nodes(self, h):
        self.num_nodes += 1
        if h.is_terminal():
            return
        for a in h.legal_actions():
            self.calc_nodes(h.child(a))


class SmallMatrix(GameConfig):
    def __init__(self, iterations, weight=1):
        super().__init__(
            game_name="nfg_game",
            iterations=iterations,
            weight=weight,
            transform=True,
        )
        self.params["filename"] = "nfg/1.nfg"


class KuhnPoker(GameConfig):
    def __init__(self, iterations=1000):
        super().__init__(
            iterations=iterations,
            weight=3,
            game_name="kuhn_poker",
        )


class LeducPoker(GameConfig):
    def __init__(self, iterations=1000):
        super().__init__(
            iterations=iterations,
            game_name="leduc_poker",
        )


class GoofSpiel3(GameConfig):
    def __init__(self, iterations=1000):
        super().__init__(
            iterations=iterations,
            game_name="goofspiel",
            transform=True,
        )
        self.params = {"num_cards": 3, "imp_info": True, "points_order": "descending"}


class GoofSpiel4(GameConfig):
    def __init__(self, iterations=1000):
        super().__init__(
            iterations=iterations,
            game_name="goofspiel",
            transform=True,
        )
        self.params = {"num_cards": 4, "imp_info": True, "points_order": "descending"}


class LiarsDice3(GameConfig):
    def __init__(self, iterations=1000):
        super().__init__(
            iterations=iterations,
            game_name="liars_dice",
        )
        self.params = {"numdice": 1, "dice_sides": 3}


class LiarsDice4(GameConfig):
    def __init__(self, iterations=1000):
        super().__init__(
            iterations=iterations,
            game_name="liars_dice",
        )
        self.params = {"numdice": 1, "dice_sides": 4}


class BattleShip2(GameConfig):
    def __init__(self, iterations=1000):
        super().__init__(
            iterations=iterations,
            game_name="battleship",
        )
        self.params = {
            "board_width": 2,
            "board_height": 2,
            "ship_sizes": "[2]",
            "ship_values": "[2]",
            "num_shots": 3,
            "allow_repeated_shots": False,
        }


class BattleShip3(GameConfig):
    def __init__(self, iterations=1000):
        super().__init__(
            iterations=iterations,
            game_name="battleship",
        )
        self.params = {
            "board_width": 2,
            "board_height": 3,
            "ship_sizes": "[2]",
            "ship_values": "[2]",
            "num_shots": 3,
            "allow_repeated_shots": False,
        }


def get_test_configs():
    test_configs = [
        SmallMatrix(1000),
        KuhnPoker(1000),
        GoofSpiel3(1000),
        LiarsDice3(1000),
        BattleShip2(1000),
        BattleShip3(1000),
        GoofSpiel4(1000),
        LiarsDice4(1000),
        LeducPoker(1000),
    ]
    return test_configs


def get_simple_configs():
    train_configs = [KuhnPoker(10), SmallMatrix(10)]
    return train_configs


def get_train_configs(num_train_games=4):
    train_configs = [
        KuhnPoker(1000),
        GoofSpiel3(1000),
        LiarsDice3(1000),
        SmallMatrix(1000),
    ]
    train_configs = train_configs[:num_train_games]
    return train_configs
