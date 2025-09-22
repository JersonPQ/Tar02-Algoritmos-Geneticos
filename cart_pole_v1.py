import gymnasium as gym
import numpy as np

AMOUNT_OF_AGENTS = 30
AMOUNT_OF_GENERATIONS = 50
MUTATION_RATE = 0.2

# Listas globales para promedios y maximos por generación
max_fitness_in_gen = []
avg_fitness_in_gen = []


class CartPoleGeneration:
    def __init__(self, amount_agents=10, agents=None):
        self.agents = (
            agents.copy()
            if agents is not None
            else [CartPoleAgent() for _ in range(amount_agents)]
        )
        self.amount_agents = amount_agents

    def get_best_agents(self) -> list:
        percentage_agents = int(
            len(self.agents) * 0.1
        )  # seleccionar el 10% de los mejores agentes
        sorted_agents = sorted(
            self.agents, key=lambda agent: agent.fitness, reverse=True
        )
        return sorted_agents[:percentage_agents]

    def crossover(self, agents: list, crossover_agents: list) -> list:
        new_agents = []
        for parent1, parent2 in zip(agents, crossover_agents):
            child_agent = CartPoleAgent()
            for i in range(len(parent1.weights)):
                child_agent.weights[i] = (
                    parent1.weights[i] if np.random.rand() < 0.5 else parent2.weights[i]
                )
            new_agents.append(child_agent)

        return new_agents

    def mutate(self, agents: list) -> list:
        copy_agents = agents.copy()
        for agent in copy_agents:
            for i in range(len(agent.weights)):
                if np.random.rand() < MUTATION_RATE:
                    agent.weights[i] = np.random.uniform(-1, 1)
        return copy_agents

    def create_new_generation(self):
        best_agents = self.get_best_agents()  # 10%
        shuffled_best_agents = best_agents.copy() * 3
        np.random.shuffle(shuffled_best_agents)
        crossover_best_agents = self.crossover(
            best_agents * 3, shuffled_best_agents
        )  # 30%
        mutated_best_agents = self.mutate(best_agents * 3)  # 30%
        mutated_crossover_agents = self.mutate(crossover_best_agents)  # 30%
        final_agents = (
            best_agents
            + crossover_best_agents
            + mutated_best_agents
            + mutated_crossover_agents
        )
        return CartPoleGeneration(agents=final_agents)

    def calculate_generation_fitness(self) -> float:
        best_agent = self.get_best_agents()[0]
        return best_agent.fitness


class CartPoleAgent:
    def __init__(self, weights=None):
        self.fitness = 0
        self.weights = (
            weights if weights is not None else np.random.uniform(-1, 1, 4)
        )  # pesos para las 4 observaciones del cart-pole

    def act(self, observation: tuple) -> int:
        action = 1 if np.dot(self.weights, observation) > 0 else 0
        return action

    def set_fitness(self, fitness: float):
        self.fitness = fitness


env = gym.make("CartPole-v1")

generation = CartPoleGeneration(amount_agents=AMOUNT_OF_AGENTS)

for _ in range(AMOUNT_OF_GENERATIONS):
    new_generation = generation.create_new_generation()
    fitness_values = []
    for i in range(AMOUNT_OF_AGENTS):
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

        cart_pole_agent.set_fitness(total_reward)
        fitness_values.append(total_reward)

    # Guardar estadísticas de esta generación
    if fitness_values:
        max_fitness_in_gen.append(float(np.max(fitness_values)))
        avg_fitness_in_gen.append(float(np.mean(fitness_values)))

    print(
        f"Mejor agente de la generación {_} tiene recompensa total: {new_generation.calculate_generation_fitness()}"
    )
    generation = new_generation

best_agent = generation.get_best_agents()[0]

print("Recompensa total del mejor agente:", best_agent.fitness)

env = gym.make("CartPole-v1", render_mode="human")
observation, info = env.reset()
finished = False
total_reward = 0
while not finished:
    action = best_agent.act(tuple(observation))
    observation, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    finished = terminated or truncated
env.close()

print("Recompensa total del mejor agente en la simulación:", total_reward)
