. "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate PokerRL
algo_names="CFR CFRPlus DCFR"
game_names="Subgame3 Subgame4 BigLeduc"
script_dir=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
main_dir=$script_dir/../
for algo_name in $algo_names;
do
    for game_name in $game_names;
    do
        conda activate PokerRL
        cd "$main_dir/PokerRL"
        nohup python -u scripts/run_cfr.py --iters 1000 --game $game_name --algo $algo_name --save 2>&1 &
    done
done
