import retro
from baselines.ppo2 import ppo2

def main():
    env = retro.make(game='SuperMarioWorld-Snes')
    ppo2.learn(network='mlp',env=env,total_timesteps=500)
    env.close()


if __name__ == "__main__":
    main()