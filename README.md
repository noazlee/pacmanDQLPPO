# pacmanDQLPPO
DQL and PPO implementation of pacman.

3 terminals:
1. python server.py --map data/fixed_classic.bmp --ghosts 2 --level 2
2. python viewer.py
3. python client.py

Flow:
train.py (creates) -> gym_pacman.py (provides states to) -> dql_agent.py (uses) -> dql_model.py (neural network)



Files not primarily written by me (env files):
* client.py
* game.py
* ghost1/2.py
* gym_observations.py
* gym_pacman.py
* mapa.py
* server.py,viewer.py