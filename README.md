# Behavioural cloning experiments with video games

## ViZDoom experiments

To gather data from a human player:

```
mkdir -p doom_recordings
python3 record_vizdoom.py --config doom_scenarios/health_gathering_supreme.cfg --num-games 20 --output doom_recordings
python3 record_vizdoom.py --config doom_scenarios/deathmatch.cfg --num-games 10 --output doom_recordings
```

After this, `run_vizdoom.sh` will train models with behavioural cloning and evaluate their performance. Models and evaluation logs will appear under `experiments` directory.

To add support for more games, modify `utils/game_keyboard_mapping.py` to list what buttons should be tracked.
