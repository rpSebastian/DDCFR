. "/home/xuhang/miniconda3/etc/profile.d/conda.sh"
run_id=$1
model_id=$2

model_name=test_${run_id}_${model_id}
conda activate DDCFR
script_dir=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
main_dir=$script_dir/../
cd "$main_dir"
nohup python -u scripts/test_model.py --run_id $run_id --model_id $model_id --model_version rl >/dev/null 2>&1 &

conda activate PokerRL
game_names="Subgame3 Subgame4 BigLeduc"
for game_name in $game_names;
do
    cd "$main_dir/PokerRL"
    nohup python -u scripts/run_ddcfr.py --run_id $run_id --model_id $model_id --iters 1000 --game $game_name --model_version rl --save >/dev/null 2>&1 &
done
