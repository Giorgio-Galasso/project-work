import numpy as np
import random
from numba import njit
from numba.typed import List
from .utils import cost_func

# ==========================================
# 1. PRINS SPLIT (CORE)
# ==========================================

@njit
def optimal_split_cost(tour, dist_matrix, golds, alpha, beta, limit=30):
    """
    Calcola il costo ottimo di partizionamento di un Giant Tour in viaggi validi.
    Usa un approccio di Programmazione Dinamica (simile a Bellman-Ford su DAG).
    """
    n = len(tour)
    # V[i] memorizza il costo minimo per servire i primi i clienti
    V = np.full(n + 1, np.inf, dtype=np.float64)
    V[0] = 0.0
    
    for i in range(n):
        if V[i] == np.inf: continue
        load = 0.0
        cost_accum = 0.0
        prev_node = 0 
        
        # Ottimizzazione: controlliamo solo i prossimi 'limit' nodi per velocità
        max_j = min(n, i + limit)
        
        for j in range(i, max_j):
            curr_node = tour[j]
            gold = golds[curr_node]
            d_leg = dist_matrix[prev_node, curr_node]
            if d_leg == np.inf: break 
            
            # Calcolo incrementale del costo del viaggio corrente
            if prev_node == 0: 
                cost_accum += d_leg 
            else: 
                # Sottraiamo il vecchio arco e aggiungiamo quello nuovo pesato
                cost_accum = cost_accum - d_leg + cost_func(d_leg, load, alpha, beta)
            
            load += gold
            d_home = dist_matrix[curr_node, 0]
            
            if d_home == np.inf: 
                prev_node = curr_node
                continue
            
            # Costo totale se chiudessimo il viaggio qui (ritorno al deposito)
            cost_return = cost_func(d_home, load, alpha, beta)
            new_cost = V[i] + cost_accum + cost_return
            
            # Rilassamento (Bellman-Ford step)
            if new_cost < V[j + 1]: 
                V[j + 1] = new_cost
                
            prev_node = curr_node
    return V[n]

@njit
def reconstruct_split(tour, dist_matrix, golds, alpha, beta, limit=30):
    """
    Simile a optimal_split_cost, ma restituisce il vettore dei predecessori P.
    P[j] indica l'inizio dell'ultimo viaggio che finisce al nodo j.
    Serve per ricostruire i viaggi effettivi.
    """
    n = len(tour)
    V = np.full(n + 1, np.inf, dtype=np.float64)
    P = np.zeros(n + 1, dtype=np.int32)
    V[0] = 0.0
    
    for i in range(n):
        if V[i] == np.inf: continue
        load = 0.0
        cost_accum = 0.0
        prev_node = 0
        
        max_j = min(n, i + limit)
        
        for j in range(i, max_j):
            curr_node = tour[j]
            gold = golds[curr_node]
            d_leg = dist_matrix[prev_node, curr_node]
            if d_leg == np.inf: break
            
            if prev_node == 0: 
                cost_accum += d_leg
            else: 
                cost_accum = cost_accum - d_leg + cost_func(d_leg, load, alpha, beta)
            
            load += gold
            d_home = dist_matrix[curr_node, 0]
            
            if d_home == np.inf: 
                prev_node = curr_node
                continue
                
            cost_return = cost_func(d_home, load, alpha, beta)
            new_cost = V[i] + cost_accum + cost_return
            
            if new_cost < V[j + 1]: 
                V[j + 1] = new_cost
                P[j + 1] = i + 1  # Salviamo l'indice di partenza del viaggio
                
            prev_node = curr_node
    return P

# ==========================================
# 2. GENETIC OPERATORS
# ==========================================

@njit
def iox_crossover(p1, p2):
    """
    Inter-route Optimization Crossover (IOX).
    Preserva sottosequenze comuni mantenendo l'ordine relativo.
    """
    n = len(p1)
    child = np.full(n, -1, dtype=np.int32)
    
    # Scegliamo una sottostringa casuale dal genitore 1
    start = random.randint(0, n - 1)
    end = random.randint(0, n - 1)
    if start > end: start, end = end, start
    
    copied_set = set()
    for i in range(start, end + 1):
        val = p1[i]
        child[i] = val
        copied_set.add(val)
        
    # Riempiamo i buchi con l'ordine del genitore 2
    curr_p2 = 0
    for i in range(n):
        if i < start or i > end:
            while curr_p2 < n:
                cand = p2[curr_p2]
                curr_p2 += 1
                if cand not in copied_set:
                    child[i] = cand
                    break
    return child

@njit
def inversion_mutation(sequence):
    """
    Inverte una sottosequenza casuale del cromosoma.
    Ottimo per TSP (simile a una mossa 2-opt casuale).
    """
    n = len(sequence)
    if n < 3: return sequence
    seq = sequence.copy()
    i = random.randint(0, n-1)
    j = random.randint(0, n-1)
    if i > j: i, j = j, i
    
    # Inversione in-place
    p1, p2 = i, j
    while p1 < p2:
        seq[p1], seq[p2] = seq[p2], seq[p1]
        p1 += 1; p2 -= 1
    return seq

@njit
def swap_mutation(sequence):
    """
    Scambia due città a caso.
    """
    n = len(sequence)
    if n < 2: return sequence
    seq = sequence.copy()
    i = random.randint(0, n-1)
    j = random.randint(0, n-1)
    while i == j: 
        j = random.randint(0, n-1)
    
    tmp = seq[i]
    seq[i] = seq[j]
    seq[j] = tmp
    return seq

@njit
def fast_2opt(sequence, dist_matrix, max_iter=100):
    """
    Local Search 2-opt veloce.
    Scambia gli archi incrociati per rimuovere inefficienze geometriche.
    """
    best_seq = sequence.copy()
    n = len(sequence)
    improved = True
    count = 0
    
    while improved and count < max_iter:
        improved = False
        count += 1
        for i in range(n - 2):
            for j in range(i + 2, n):
                if j == n - 1: continue 
                
                # Valutiamo se scambiare gli archi (i, i+1) e (j, j+1) conviene
                nA, nB = best_seq[i], best_seq[i+1]
                nC, nD = best_seq[j], best_seq[j+1]
                
                # Check geometrico basato solo sulle distanze (più veloce)
                if dist_matrix[nA, nC] + dist_matrix[nB, nD] < dist_matrix[nA, nB] + dist_matrix[nC, nD] - 1e-6:
                    # Eseguiamo lo swap (inversione della sottostringa)
                    p1, p2 = i + 1, j
                    while p1 < p2:
                        best_seq[p1], best_seq[p2] = best_seq[p2], best_seq[p1]
                        p1 += 1; p2 -= 1
                    improved = True
    return best_seq

@njit
def nearest_neighbor_init(nodes, dist_matrix):
    """
    Genera un individuo iniziale usando l'euristica Nearest Neighbor.
    """
    n = len(nodes)
    if n == 0: return nodes
    tour = np.zeros(n, dtype=np.int32)
    visited = np.zeros(n, dtype=np.int8) 
    
    # Partenza casuale per diversificare la popolazione
    curr_idx = random.randint(0, n-1)
    tour[0] = nodes[curr_idx]
    visited[curr_idx] = 1
    
    for i in range(1, n):
        last_node = tour[i-1]
        best_node_idx = -1
        min_d = np.inf
        
        # Cerca il nodo più vicino non ancora visitato
        for j in range(n):
            if not visited[j]:
                d = dist_matrix[last_node, nodes[j]]
                if d < min_d:
                    min_d = d
                    best_node_idx = j
                    
        if best_node_idx != -1:
            tour[i] = nodes[best_node_idx]
            visited[best_node_idx] = 1
        else:
            # Fallback (non dovrebbe succedere in grafi connessi)
            for j in range(n):
                if not visited[j]:
                    tour[i] = nodes[j]
                    visited[j] = 1
                    break
    return tour

# ==========================================
# 3. LNS HELPERS (Large Neighborhood Search)
# ==========================================

@njit
def get_routes(tour, predecessors):
    """
    Estrae le rotte individuali dal Giant Tour usando il vettore dei predecessori P.
    """
    routes = List()
    # Dummy list per inizializzare il tipo List(List(int))
    dummy = List(); dummy.append(np.int32(0)); routes.append(dummy); routes.clear()
    
    curr = len(tour)
    while curr > 0:
        prev = predecessors[curr]
        start = prev - 1
        if start < 0: start = 0
        trip = List()
        for k in range(start, curr): 
            trip.append(tour[k])
        routes.append(trip)
        curr = prev - 1
        
    # Le rotte sono estratte al contrario, le invertiamo
    n_routes = len(routes)
    reversed_routes = List()
    dummy2 = List(); dummy2.append(np.int32(0)); reversed_routes.append(dummy2); reversed_routes.clear()
    
    for i in range(n_routes - 1, -1, -1):
        reversed_routes.append(routes[i])
    return reversed_routes

@njit
def calculate_single_route_cost(route, dist_matrix, golds, alpha, beta):
    """
    Calcola il costo esatto di una singola rotta (load dependent).
    """
    if len(route) == 0: return 0.0
    c = 0.0
    load = 0.0
    prev = 0
    for node in route:
        d = dist_matrix[prev, node]
        c += cost_func(d, load, alpha, beta)
        load += golds[node]
        prev = node
    # Ritorno al deposito
    d = dist_matrix[prev, 0]
    c += cost_func(d, load, alpha, beta)
    return c

@njit
def lns_destroy_repair(tour, predecessors, dist_matrix, golds, alpha, beta, num_remove):
    """
    Implementa la logica Destroy & Repair dell'LNS.
    1. Destroy: Rimuove 'num_remove' nodi casuali dalle rotte.
    2. Repair: Reinserisce i nodi usando una politica Greedy Best Insertion.
       Valuta l'incremento di costo REALE (con alpha e beta) per ogni posizione.
    """
    routes = get_routes(tour, predecessors)
    
    # Flattening delle rotte in una lista di nodi
    all_nodes = List()
    for r in routes:
        for n in r: all_nodes.append(n)
        
    total = len(all_nodes)
    if total <= num_remove: return tour
    
    # Selezione casuale dei nodi da rimuovere
    indices = np.arange(total)
    for i in range(total - 1, 0, -1):
        j = random.randint(0, i)
        indices[i], indices[j] = indices[j], indices[i]
        
    removed_set = set()
    for k in range(num_remove): 
        removed_set.add(indices[k])
        
    removed_nodes = List()
    for idx in range(total):
        if idx in removed_set: 
            removed_nodes.append(all_nodes[idx])
            
    # Costruzione delle rotte parziali (senza i nodi rimossi)
    partial_routes = List(); dummy = List(); dummy.append(np.int32(0)); partial_routes.append(dummy); partial_routes.clear()
    curr_idx = 0
    for r in routes:
        new_r = List()
        for n in r:
            if curr_idx not in removed_set: 
                new_r.append(n)
            curr_idx += 1
        if len(new_r) > 0: 
            partial_routes.append(new_r)
            
    # REPAIR: Reinserimento Greedy
    for node in removed_nodes:
        best_incr = np.inf; best_r = -1; best_p = -1
        
        # Prova a inserire il nodo in ogni posizione di ogni rotta esistente
        for r_idx in range(len(partial_routes)):
            r = partial_routes[r_idx]
            base_c = calculate_single_route_cost(r, dist_matrix, golds, alpha, beta)
            for pos in range(len(r) + 1):
                # Crea rotta temporanea
                temp_r = List()
                for k in range(pos): temp_r.append(r[k])
                temp_r.append(node)
                for k in range(pos, len(r)): temp_r.append(r[k])
                
                new_c = calculate_single_route_cost(temp_r, dist_matrix, golds, alpha, beta)
                incr = new_c - base_c
                if incr < best_incr:
                    best_incr = incr; best_r = r_idx; best_p = pos
                    
        # Considera anche la creazione di una nuova rotta dedicata (solo per questo nodo)
        single = List(); single.append(node)
        single_c = calculate_single_route_cost(single, dist_matrix, golds, alpha, beta)
        
        if single_c < best_incr: 
            best_incr = single_c; best_r = len(partial_routes); best_p = 0
            
        # Applica la mossa migliore
        if best_r == len(partial_routes): 
            new_r = List(); new_r.append(node); partial_routes.append(new_r)
        else:
            target = partial_routes[best_r]; final_r = List()
            for k in range(best_p): final_r.append(target[k])
            final_r.append(node)
            for k in range(best_p, len(target)): final_r.append(target[k])
            partial_routes[best_r] = final_r
            
    # Ricostruisce il Giant Tour finale concatenando le rotte riparate
    final_len = 0
    for r in partial_routes: final_len += len(r)
    new_tour = np.empty(final_len, dtype=np.int32)
    idx = 0
    for r in partial_routes:
        for n in r: 
            new_tour[idx] = n; idx += 1
    return new_tour