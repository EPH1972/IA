import random

def calcular_fitness(individuo):
    """Cuenta cuántos pares de reinas se atacan."""
    ataques = 0
    n = len(individuo)
    for i in range(n):
        for j in range(i + 1, n):
            # Misma fila o misma diagonal
            if individuo[i] == individuo[j] or abs(individuo[i] - individuo[j]) == abs(i - j):
                ataques += 1
    return ataques

def genetic_8_queens(n=8, poblacion_size=100):
    # 1. Población inicial
    poblacion = [[random.randint(0, n-1) for _ in range(n)] for _ in range(poblacion_size)]
    
    for generacion in range(1000):
        # Sort por fitness (menor es mejor)
        poblacion = sorted(poblacion, key=lambda ind: calcular_fitness(ind))
        
        if calcular_fitness(poblacion[0]) == 0:
            print(f"✅ Solución encontrada en Gen {generacion}: {poblacion[0]}")
            return poblacion[0]

        # 2. Selección (los 20 mejores) y Crossover
        nueva_poblacion = poblacion[:20]
        while len(nueva_poblacion) < poblacion_size:
            padre1, padre2 = random.sample(poblacion[:50], 2)
            punto_cruce = random.randint(1, n - 1)  # punto de cruce aleatorio
            hijo = padre1[:punto_cruce] + padre2[punto_cruce:]

            # 3. Mutación (5% de probabilidad)
            if random.random() < 0.05:
                hijo[random.randint(0, n-1)] = random.randint(0, n-1)
            nueva_poblacion.append(hijo)
        poblacion = nueva_poblacion

    print(f"❌ No se encontró solución en 1000 generaciones. Mejor: {poblacion[0]} (ataques: {calcular_fitness(poblacion[0])})")
    return None

# Ejecutar
solucion = genetic_8_queens(8)
if solucion is None:
    print("Intenta aumentar el número de generaciones o el tamaño de la población.")