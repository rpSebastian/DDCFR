. "/home/xuhang/miniconda3/etc/profile.d/conda.sh"
conda activate DDCFR
script_dir=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
main_dir=$script_dir/../
cd "$main_dir"
nohup python -u scripts/test_model.py --model_name model >/dev/null 2>&1 &

conda activate PokerRL
game_names="Subgame3 Subgame4 BigLeduc"
for game_name in $game_names;
do
    cd "$main_dir/PokerRL"
    nohup python -u scripts/run_ddcfr.py --model_name model --iters 1000 --game $game_name --save >/dev/null 2>&1 &
done
