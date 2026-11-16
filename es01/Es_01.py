"""Esercizio 1: Max-Cut pesato mediante QUBO e simulated annealing.

Installazione: python -m pip install -r requirements.txt
Esecuzione:   python Es_01.py [percorso/max_cut.txt]

Per ogni nodo i, x_i = 0 indica S1 e x_i = 1 indica S2.
Il peso del taglio e' C(x) = sum_(i,j) w_ij (x_i + x_j - 2*x_i*x_j).
Si minimizza quindi f(x) = -C(x), senza vincoli o variabili ausiliarie.
Nella convenzione triangolare f(x) = sum_(i<=j) Q_ij*x_i*x_j:
Q_ii = -sum_j w_ij e Q_ij = 2*w_ij per i < j.
"""

import argparse
from itertools import product
from math import isclose, isfinite
from pathlib import Path

from qubovert import QUBO, boolean_var
from qubovert.sim import anneal_qubo


DEFAULT_INPUT = Path(__file__).resolve().parent / "Lab4_input_files/Lab4/max_cut.txt"


def read_graph(path):
    """Legge n e gli archi (i, j, peso), con nodi numerati da 0 a n-1."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    lines = [line.strip() for line in lines if line.strip()]
    if not lines:
        raise ValueError("Il file del grafo e' vuoto.")
    n = int(lines[0])
    if n < 1:
        raise ValueError("Il numero di nodi deve essere positivo.")
    edges = []
    for line_number, line in enumerate(lines[1:], start=2):
        fields = line.split()
        if len(fields) != 3:
            raise ValueError(f"Riga {line_number}: attesi nodo1 nodo2 peso.")
        i, j = map(int, fields[:2])
        weight = float(fields[2])
        if not (0 <= i < n and 0 <= j < n) or not isfinite(weight):
            raise ValueError(f"Riga {line_number}: nodo o peso non valido.")
        edges.append((i, j, weight))
    return n, edges


def build_qubo(n, edges):
    """Costruisce simbolicamente il QUBO per un grafo pesato generico."""
    x = [boolean_var(i) for i in range(n)]
    objective = QUBO()
    for i, j, weight in edges:
        objective += weight * (2 * x[i] * x[j] - x[i] - x[j])
    return objective


def cut_weight(state, edges):
    """Calcola direttamente il peso degli archi che attraversano il taglio."""
    return sum(weight for i, j, weight in edges if state[i] != state[j])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--num-anneals", type=int, default=100)
    parser.add_argument("--duration", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.num_anneals < 1 or args.duration < 1:
        parser.error("num-anneals e duration devono essere positivi")

    n, edges = read_graph(args.input)
    qubo = build_qubo(n, edges)
    if qubo:
        results = anneal_qubo(
            qubo, num_anneals=args.num_anneals,
            anneal_duration=args.duration, seed=args.seed,
        )
        # Le variabili assenti dal QUBO (ad esempio nodi isolati) sono libere.
        state = {i: results.best.state.get(i, 0) for i in range(n)}
        energy = results.best.value
    else:
        state = dict.fromkeys(range(n), 0)
        energy = 0

    weight = cut_weight(state, edges)
    if not isclose(energy, -weight, abs_tol=1e-9):
        raise RuntimeError("Energia QUBO e peso del taglio non coerenti.")

    print(f"Grafo: {n} nodi, {len(edges)} archi")
    print(f"QUBO: {qubo}")
    print(f"Soluzione simulated annealing (seed={args.seed}): {state}")
    print("S1:", [i for i in range(n) if state[i] == 0])
    print("S2:", [i for i in range(n) if state[i] == 1])
    print(f"Energia QUBO: {energy:g}")
    print(f"Peso del taglio: {weight:g}")
    print("Archi del taglio:", [(i, j, w) for i, j, w in edges if state[i] != state[j]])

    # Verifica esatta solo per piccole istanze; il solutore principale resta SA.
    # Fissare x_0=0 elimina la simmetria tra un taglio e il suo complementare.
    if n <= 20:
        optimum = max(
            cut_weight((0,) + bits, edges)
            for bits in product((0, 1), repeat=n - 1)
        )
        print(f"Ottimo esatto: {optimum:g}")
        print(f"Scarto dall'ottimo: {optimum - weight:g}")
        print("Ottimalita' verificata:", isclose(weight, optimum, abs_tol=1e-9))
    else:
        print("Verifica esaustiva omessa: piu' di 20 nodi.")


if __name__ == "__main__":
    main()
