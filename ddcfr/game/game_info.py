from collections import defaultdict


def dfs(h, length, info):
    h_str = h.history_str()
    info["history_set"].add(h_str)
    if h.is_player_node():
        s_str = h.information_state_string()
        info["infostate_set"].add(s_str)
        info["infostate_count"][s_str].add(h_str)

    if h.is_terminal():
        info["max_length"] = max(info["max_length"], length)
        info["terminal_state"] += 1
        return

    for a in h.legal_actions():
        dfs(h.child(a), length + 1, info)


def calc_game_info(game_config):
    game = game_config.load_game()
    info = {
        "history_set": set(),
        "infostate_set": set(),
        "max_length": 0,
        "infostate_count": defaultdict(set),
        "terminal_state": 0,
    }
    h = game.new_initial_state()
    dfs(h, 1, info)

    history_size = len(info["history_set"])
    infostate_size = len(info["infostate_set"])
    max_length = info["max_length"]
    terminal_state_size = info["terminal_state"]
    max_infostate_count = max(len(j) for j in info["infostate_count"].values())
    result = {
        "game_name": game_config.name,
        "history_size": history_size,
        "infostate_size": infostate_size,
        "max_length": max_length,
        "terminal_state_size": terminal_state_size,
        "max_infostate_count": max_infostate_count,
    }
    return result
