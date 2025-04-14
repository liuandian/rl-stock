mkdir nohup
nohup python -u ./trader.py -m 'a2c' 
nohup python -u ./trader.py -m 'ddpg' 
nohup python -u ./trader.py -m 'ppo' 
nohup python -u ./trader.py -m 'td3'
nohup python -u ./trader.py -m 'sac' 
