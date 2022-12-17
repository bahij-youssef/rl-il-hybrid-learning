### Training ###
epochs=100
workers=4
framestack=1
l2=0.00001
data_dir=doom_recordings
experiments_dir=experiments
save_freq=1
repetitions=3

width=80
height=60

### Evaluation ###
model_dir=$experiments_dir/vizdoom-models
savedir=$experiments_dir/vizdoom-results
action=sampling
processes=4
games=50

mkdir -p $savedir

# Only evaluate last three epochs for the final performance
for epoch in $((epochs - 2)) $((epochs - 1)) $epochs
do
    python3 play_vizdoom.py $model_dir/vizdoom_defend_the_center_1_${epoch}.pt --config doom_scenarios/defend_the_center.cfg --processes $processes --games $games --framestack $framestack --save $savedir --width $width --height $height
    python3 play_vizdoom.py $model_dir/vizdoom_defend_the_center_2_${epoch}.pt --config doom_scenarios/defend_the_center.cfg --processes $processes --games $games --framestack $framestack --save $savedir --width $width --height $height
    python3 play_vizdoom.py $model_dir/vizdoom_defend_the_center_3_${epoch}.pt --config doom_scenarios/defend_the_center.cfg --processes $processes --games $games --framestack $framestack --save $savedir --width $width --height $height
done