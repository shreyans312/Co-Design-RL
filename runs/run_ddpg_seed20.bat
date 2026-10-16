@echo off
cd /d D:\Coding\MCD310
"D:\Coding\.venv\Scripts\python.exe" -u train_rl.py --model ddpg_lognormal --seed 20 --log-every 10 --output-directory runs\ddpg_lognormal_fixed\seed_20 > runs\ddpg_lognormal_fixed_seed20.log 2>&1
