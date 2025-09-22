import matplotlib.pyplot as plt
import cart_pole_v1 as cp

# ESTE ARCHIVO ES PARA CORRER EL ENTORNO Y GRAFICAR LAS ESTADISTICAS
# Grafica segun lo configurado en AMOUNT_OF_AGENTS y AMOUNT_OF_GENERATIONS de cart_pole_v1.py
# y la imagen se nombra dependiendo de esos valores y MUTATION_RATE

# Al importar el módulo, se ejecuta la simulación configurada en cart_pole_v1
# y se rellenan las listas globales de métricas
max_fitness_in_gen = cp.max_fitness_in_gen
avg_fitness_in_gen = cp.avg_fitness_in_gen

# Imprimir estadísticas por generación
for i, (mx, avg) in enumerate(zip(max_fitness_in_gen, avg_fitness_in_gen)):
    print(f"[Gen {i}] Mejor recompensa: {mx:.2f} | Promedio: {avg:.2f}")

# ---- gráfica con matplotlib ----
plt.figure(figsize=(10, 5))
plt.plot(max_fitness_in_gen, label="Fitness Máximo", linewidth=2)
plt.plot(avg_fitness_in_gen, label="Fitness Promedio", linestyle="--")
plt.xlabel("Generación")
plt.ylabel("Recompensa")
plt.title(
    f"Evolución del Fitness en CartPole-v1 {cp.AMOUNT_OF_AGENTS} agentes, {cp.AMOUNT_OF_GENERATIONS} generaciones, tasa mutación {cp.MUTATION_RATE}"
)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(
    f"evolutiongraph_{cp.AMOUNT_OF_AGENTS}agents_{cp.AMOUNT_OF_GENERATIONS}gens_{cp.MUTATION_RATE}mut.png",
    dpi=150,
)  # Guarda la gráfica como un png
plt.show()
