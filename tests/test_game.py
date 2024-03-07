import pytest

from ddcfr.game import GoofSpiel4, KuhnPoker, LeducPoker


def test_game_config():
    game_config = KuhnPoker()
    assert game_config.name == "KuhnPoker"
    assert game_config.weight == 3

    cfr_conv = game_config.cfr_df.loc[10, "KuhnPoker/conv"]
    score = game_config.conv_to_score(cfr_conv, 10)
    assert score == 0
    dcfr_conv = game_config.dcfr_df.loc[10, "KuhnPoker/conv"]
    score = game_config.conv_to_score(dcfr_conv, 10)
    assert score == 1
    conv = 0.01
    score = game_config.conv_to_score(conv, 10)
    assert score == pytest.approx(1.745758617147705, abs=1e-10)


def test_game():
    game_config = GoofSpiel4()
    game = game_config.load_game()
    root = game.new_initial_state()
    embedding_size = len(root.information_state_tensor(0))
    assert embedding_size == 70
    assert game_config.num_nodes == 1077

    game_config = KuhnPoker()
    game = game_config.load_game()
    root = game.new_initial_state()
    embedding_size = len(root.information_state_tensor(0))
    assert embedding_size == 11
    assert game_config.num_nodes == 58

    game_config = LeducPoker()
    game = game_config.load_game()
    root = game.new_initial_state()
    embedding_size = len(root.information_state_tensor(0))
    assert embedding_size == 30
    assert game_config.num_nodes == 9457
