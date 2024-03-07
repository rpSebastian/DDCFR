# from PokerRL._.CrayonWrapper import CrayonWrapper

from PokerRL.cfr import VanillaCFR
from PokerRL.game import bet_sets
from PokerRL.game.games import (
    BigLeduc,
    DiscretizedNLHoldemSubGame1,
    DiscretizedNLHoldemSubGame2,
    DiscretizedNLHoldemSubGame3,
    DiscretizedNLHoldemSubGame4,
    Flop5Holdem,
    StandardLeduc,
)
from PokerRL.rl.base_cls.workers.ChiefBase import ChiefBase

game_dict = {
    "flop5": Flop5Holdem,
    "Subgame1": DiscretizedNLHoldemSubGame1,
    "Subgame2": DiscretizedNLHoldemSubGame2,
    "Subgame3": DiscretizedNLHoldemSubGame3,
    "Subgame4": DiscretizedNLHoldemSubGame4,
    "BigLeduc": BigLeduc,
    "Leduc": StandardLeduc,
}


def get_cards(tree, node):
    board = 0
    for board_card in node.env_state["board_2d"]:
        if board_card[0] != -127:
            board += 1
    hand = 0
    for hand_card in node.env_state["seats"][0]["hand"]:
        if hand_card[0] != -127:
            hand += 1
    total = tree._env_bldr.rules.N_CARDS_IN_DECK
    deck = total - board - hand * 2
    return board, hand, deck


def hidden(deck, hand):
    if hand == 1:
        return deck + hand
    if hand == 2:
        return (deck + 2) * (deck + 1)


def dfs(tree, node, info, n_hands, last_node, length):
    board, hand, deck = get_cards(tree, node)
    info["history_size"] += hidden(deck + hand, hand) * hidden(deck, hand)

    if node.is_terminal:
        info["max_length"] = max(length, info["max_length"])
        info["terminal_state_size"] += hidden(deck + hand, hand) * hidden(deck, hand)
        return
    else:
        if node.p_id_acting_next != "Ch":
            info["infostate_size"] += hidden(deck + hand, hand)

    for child in node.children:
        dfs(tree, child, info, n_hands, node, length + 1)


def output(game_infos):
    game_names = [
        ("Leduc", "Leduc Poker"),
        ("BigLeduc", "Big Leduc Poker"),
        ("Subgame3", "HUNL Subgame-3"),
        ("Subgame4", "HUNL Subgame-4"),
    ]
    attributes = [
        ("history_size", "#Histories"),
        ("infostate_size", "#Information sets"),
        ("terminal_state_size", "#Terminal histories"),
        ("max_length", "Depth"),
        ("max_infostate_count", "Maximum size of information set"),
    ]
    for original_game_name, game_name in game_names:
        game_info = game_infos[original_game_name]
        print(game_name, game_info)

    print("{:<20}& ".format("Game"), end=" ")
    for idx, attribute in enumerate(attributes):
        sign = "&" if idx < len(attributes) - 1 else "\\\\"
        print("{:<12} {}".format(attribute[1], sign), end=" ")
    print("\\midrule")

    for original_game_name, game_name in game_names:
        print("{:<20}& ".format(game_name), end=" ")
        game_info = game_infos[original_game_name]
        for idx, (original_attribute, attribute) in enumerate(attributes):
            sign = "&" if idx < len(attributes) - 1 else "\\\\"
            print("{:<12} {}".format(game_info[original_attribute], sign), end=" ")

        print()


game_infos = {}
for game_name in ["Leduc", "Subgame3", "Subgame4", "BigLeduc"]:
    chief = ChiefBase(t_prof=None)
    cfr = VanillaCFR(
        name="cfr",
        game_cls=game_dict[game_name],
        agent_bet_set=bet_sets.B_3,
        other_agent_bet_set=bet_sets.B_2,
        chief_handle=chief,
    )
    tree = cfr._trees[0]
    info = {}
    n_hands = tree._env_bldr.rules.RANGE_SIZE

    board, hand, deck = get_cards(tree, tree.root)
    info["game_name"] = game_name
    info["history_size"] = 1
    info["max_length"] = 2
    info["history_size"] += hidden(deck + hand, hand)
    info["infostate_size"] = 0
    info["terminal_state_size"] = 0
    info["max_infostate_count"] = hidden(deck, hand)
    dfs(tree, tree.root, info, n_hands, tree.root, 3)
    game_infos[info["game_name"]] = info

output(game_infos)
