# Run `pip install "gymnasium[classic-control]"` for this example.
import random
import gymnasium as gym

AMOUNT_OF_AGENTS = 100
AMOUNT_OF_GENERATIONS = 100

class CartPoleGeneration:
    def __init__(self, amount_agents=10, agents=None):
        self.agents = agents.copy() if agents is not None else [CartPoleAgent() for _ in range(amount_agents)]
        self.amount_agents = amount_agents

    def add_agent(self, agent):
        self.agents.append(agent)

    def get_best_agents(self) -> list:
        percentage_agents = int(len(self.agents) * 0.1) # seleccionar el 10% de los mejores agentes
        sorted_agents = sorted(self.agents, key=lambda agent: agent.total_reward, reverse=True)
        return sorted_agents[:percentage_agents]

    def crossover(self, agents: list, crossover_agents: list) -> list:
        # Mezclar los mejores agentes para crear nuevos
        new_agents = []
        for parent1, parent2 in zip(agents, crossover_agents):
            child_agent = CartPoleAgent()
            for observation in parent1.get_observations():
                if observation in parent2.get_observations():
                    child_agent.observation_action[observation] = random.choice([
                        parent1.get_observations()[observation],
                        parent2.get_observations()[observation]
                    ])
                else:
                    child_agent.observation_action[observation] = parent1.get_observations()[observation]
            for observation in parent2.get_observations():
                if observation not in child_agent.get_observations():
                    child_agent.observation_action[observation] = parent2.get_observations()[observation]
            new_agents.append(child_agent)

        return new_agents
    
    def mutate(self, agents: list) -> list:
        mutation_rate = 0.5 # 50% de probabilidad de mutación
        copy_agents = agents.copy()
        for agent in copy_agents:
            for observation in agent.get_observations():
                if random.random() < mutation_rate:
                    agent.observation_action[observation] = random.choice([0, 1])
        return copy_agents
    
    def create_new_generation(self):
        best_agents = self.get_best_agents() # 10%
        shuffled_best_agents = best_agents.copy() * 3
        random.shuffle(shuffled_best_agents)
        crossover_best_agents = self.crossover(best_agents * 3, shuffled_best_agents) # 30%
        mutated_best_agents = self.mutate(best_agents * 3) # 30%
        mutated_crossover_agents = self.mutate(crossover_best_agents) # 30%
        final_agents = best_agents + crossover_best_agents + mutated_best_agents + mutated_crossover_agents
        return CartPoleGeneration(agents=final_agents)
    
    def calculate_generation_fitness(self) -> float:
        best_agent = self.get_best_agents()[0]
        return best_agent.total_reward


class CartPoleAgent:
    def __init__(self):
        self.action_space = [0, 1] # mover izquierda, mover derecha
        self.observation_action = {}
        self.total_reward = 0

    def act(self, observation: tuple) -> int:
        if observation not in self.observation_action:
            self.observation_action[observation] = random.choice(self.action_space)
        return self.observation_action[observation]

    def get_observations(self) -> dict:
        return self.observation_action
    
    def set_total_reward(self, reward: float):
        self.total_reward = reward

# Create our training environment - a cart with a pole that needs balancing
env = gym.make("CartPole-v1")

generation = CartPoleGeneration(amount_agents=AMOUNT_OF_AGENTS)

for _ in range(AMOUNT_OF_GENERATIONS):
    new_generation = generation.create_new_generation()
    for i in range(AMOUNT_OF_AGENTS):
        # en caso que sea la primera generación, crear agentes nuevos
        # caso contrario, usar los agentes de la nueva generación mezclados y mutados
        cart_pole_agent = new_generation.agents[i]
        finished = False
        total_reward = 0

        observation, info = env.reset()
        while not finished:
            action = cart_pole_agent.act(tuple(observation))

            # reward: +1 por cada paso realizado con el palo en equilibrio
            # terminated: True si el palo se cae (el agente ha fallado)
            # truncated: True si se alcanza el límite de tiempo (500 pasos)
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            finished = terminated or truncated

        cart_pole_agent.set_total_reward(total_reward)

    print(f"Mejor agente de la generación {_} tiene recompensa total: {new_generation.calculate_generation_fitness()}")
    generation = new_generation

    # simulacion del mejor agente de la generación actual
    env_best_agent = gym.make("CartPole-v1", render_mode="human")
    best_agent_current_gen = generation.get_best_agents()[0]
    observation, info = env_best_agent.reset()
    finished = False
    while not finished:
        action = best_agent_current_gen.act(tuple(observation))
        observation, reward, terminated, truncated, info = env_best_agent.step(action)
        finished = terminated or truncated
    env_best_agent.close()

best_agent = generation.get_best_agents()[0]

print("Recompensa total del mejor agente:", best_agent.total_reward)

env = gym.make("CartPole-v1", render_mode="human")
observation, info = env.reset()
finished = False
total_reward = 0
while not finished:
    action = generation.agents[0].act(tuple(observation))
    observation, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    finished = terminated or truncated
env.close()

print("Recompensa total del mejor agente en la simulación:", total_reward)