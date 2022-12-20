width=80
height=60
model_dir=/home/model_testing
games=50

python3 play_vizdoom.py $model_dir --games $games --width $width --height $height
