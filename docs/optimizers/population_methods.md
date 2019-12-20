## Particle Swarm Optimization

Particle swarm optimization works by initializing a number of positions at the same time and moving all of those closer to the best one after each iteration.

---

**Use case/properties:**
- If the search space is complex and large
- If you have enough time for many model evaluations

<p align="center">
<img src="./plots/search_paths/ParticleSwarm [('n_particles', 10)].svg" width= 49%/>
<img src="./plots/search_paths/ParticleSwarm [('n_particles', 20)].svg" width= 49%/>
</p>


## Evolution Strategy
Evolution strategy mutates and combines the best individuals of a population across a number of generations without transforming them into an array of bits (like genetic algorithms) but uses the real values of the positions.

---

**Use case/properties:**
- If the search space is very complex and large
- If you have enough time for many model evaluations

<p align="center">
<img src="./plots/search_paths/EvolutionStrategy [('individuals', 4)].svg" width= 49%/>
<img src="./plots/search_paths/EvolutionStrategy [('individuals', 10)].svg" width= 49%/>
<img src="./plots/search_paths/EvolutionStrategy [('individuals', 10), ('mutation_rate', 0.1), ('crossover_rate', 0.9)].svg" width= 49%/>
<img src="./plots/search_paths/EvolutionStrategy [('individuals', 10), ('mutation_rate', 0.9), ('crossover_rate', 0.1)].svg" width= 49%/>
</p>
