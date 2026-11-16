from qubovert import QUBO, boolean_var
from qubovert.sim import anneal_qubo


def read_graph(filename):
  """Legge il numero di nodi e la lista degli archi pesati da max_cut.txt."""
  edges = []
  with open(filename, 'r') as file:
    # Legge tutte le righe non vuote rimuovendo spazi inutili
    lines = [line.strip() for line in file if line.strip()]

  # La prima riga contiene il numero di nodi
  num_nodes = int(lines[0])

  # Le righe successive contengono: nodo1 nodo2 peso
  for line in lines[1:]:
    parts = line.split()
    u = int(parts[0])
    v = int(parts[1])
    # Usa float se i pesi possono essere decimali, oppure int se sono interi
    weight = float(parts[2]) if '.' in parts[2] else int(parts[2])
    edges.append((u, v, weight))

  return num_nodes, edges


def create_max_cut_qubo(num_nodes, edges):
  """Costruisce la formulazione QUBO generica per il Max-Cut."""
  model = QUBO()

  # Crea una variabile binaria x_i per ogni nodo del grafo
  # Nota: se nel file i nodi partono da 1 invece che da 0, usa range(1, num_nodes + 1)
  x = {i: boolean_var(f'x_{i}') for i in range(num_nodes)}

  # Aggiunge alla funzione obiettivo il termine w_ij * (2*x_i*x_j - x_i - x_j) per ogni arco
  for u, v, w in edges:
    model += w * (2 * x[u] * x[v] - x[u] - x[v])

  return model


def main():
  filename = 'C:\\Users\\gbett\\Quantum_Lab_4\\es01\\max_cut.txt'

  # 1. Lettura del grafo dal file
  num_nodes, edges = read_graph(filename)
  print(f'Grafo caricato con successo: {num_nodes} nodi e {len(edges)} archi.')

  # 2. Creazione del modello QUBO
  qubo_model = create_max_cut_qubo(num_nodes, edges)

  # 3. Risoluzione con Simulated Annealing
  res = anneal_qubo(qubo_model, num_anneals=50)

  # 4. Estrazione dei risultati
  best_state = res.best.state
  min_energy = res.best.value
  max_cut_weight = -min_energy

  # Divisione dei nodi nei due sottoinsiemi S1 (x_i = 0) e S2 (x_i = 1)
  s1 = [node for node in range(num_nodes) if best_state[f'x_{node}'] == 0]
  s2 = [node for node in range(num_nodes) if best_state[f'x_{node}'] == 1]

  print('\n--- Risultato Max-Cut ---')
  print(f'Energia minima (QUBO): {min_energy}')
  print(f'Peso massimo del taglio: {max_cut_weight}')
  print(f'Sottoinsieme S1 (0): {s1}')
  print(f'Sottoinsieme S2 (1): {s2}')


if __name__ == '__main__':
  main()