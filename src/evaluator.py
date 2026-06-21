import networkx as nx
import numpy as np
from .utils import optimal_fraction_size, cost_func
from .algorithms import (
    optimal_split_cost, reconstruct_split, iox_crossover, fast_2opt,
    inversion_mutation, swap_mutation, nearest_neighbor_init, lns_destroy_repair
)

def preprocess_bulk_removal(p):
    """
    Gestisce la strategia 'Bulk Removal' per valori di Beta > 1.
    In questi casi, il costo del trasporto cresce in modo super-lineare col peso.
    Conviene quindi fare viaggi dedicati 'avanti e indietro' dal deposito per ridurre il carico.
    
    Returns:
        fixed_path: Una lista di tuple (nodo, gold) che rappresenta i viaggi 'fissi' già decisi.
        ResidualProblem: Un oggetto problema ridotto con l'oro residuo da raccogliere col GA.
    """
    beta = getattr(p, 'beta', 1.0)
    # Se beta <= 1, la strategia Bulk non serve (il carico non pesa così tanto).
    if beta <= 1.0: return [], p 
    
    alpha = getattr(p, 'alpha', 0.1)
    
    # Pre-calcolo distanze dal deposito (nodo 0) per tutti i nodi
    try: dists_from_0 = nx.single_source_dijkstra_path_length(p.graph, 0, weight='dist')
    except: dists_from_0 = {n: 0.0 for n in p.graph.nodes}
    
    fixed_path = []
    res_graph = p.graph.copy()
    nodes_with_gold = [n for n in res_graph.nodes if n != 0 and res_graph.nodes[n]['gold'] > 0]
    
    for node in nodes_with_gold:
        d = dists_from_0.get(node, float('inf'))
        if d == float('inf'): continue
        
        # Calcola la dimensione ottimale del carico per minimizzare il costo specifico
        opt_size = optimal_fraction_size(alpha, beta, d)
        gold = int(res_graph.nodes[node]['gold'])
        
        # Se c'è più oro del carico ottimale, facciamo viaggi dedicati
        if gold > opt_size:
            chunk = int(opt_size)
            if chunk < 1: chunk = 1
            num_trips = int(gold // chunk)
            rem = gold - (num_trips * chunk)
            
            # Aggiustamento fine per non lasciare residui troppo piccoli
            if rem == 0 and num_trips > 0: num_trips -= 1; rem += chunk
            
            # Aggiorna l'oro residuo nel grafo
            res_graph.nodes[node]['gold'] = rem
            
            # Calcola il percorso reale nel grafo (shortest path)
            try:
                pto = nx.shortest_path(p.graph, 0, node, weight='dist')
                if p.graph.is_directed(): pfrom = nx.shortest_path(p.graph, node, 0, weight='dist')
                else: pfrom = pto[::-1]
            except: continue
            
            # Aggiungi i viaggi dedicati alla soluzione fissa
            for _ in range(num_trips):
                for step in pto[1:]: fixed_path.append((step, chunk if step == node else 0))
                for step in pfrom[1:]: fixed_path.append((step, 0))
                
    # Classe wrapper per ingannare il GA facendogli credere che questo sia il problema originale
    class ResidualProblem:
        def __init__(self, g, orig_p):
            self.graph = g; self.beta = orig_p.beta; self.alpha = getattr(orig_p, 'alpha', 1.0)
            
    return fixed_path, ResidualProblem(res_graph, p)

class SmartEvaluator:
    """
    Classe wrapper che interfaccia il problema basato su grafo (NetworkX)
    con gli algoritmi ottimizzati basati su array (Numba).
    Gestisce la matrice delle distanze e le chiamate alle funzioni veloci.
    """
    def __init__(self, problem):
        self.p = problem
        self.graph = problem.graph
        self.alpha = getattr(problem, 'alpha', 1.0)
        self.beta = getattr(problem, 'beta', 1.0)
        
        # Costruzione della matrice delle distanze (All-Pairs Shortest Path)
        # Questo permette lookup O(1) durante l'evoluzione, evitando chiamate lente a NetworkX
        max_node_id = max(self.graph.nodes())
        self.num_nodes = max_node_id + 1
        self.dist_matrix = np.full((self.num_nodes, self.num_nodes), np.inf, dtype=np.float64)
        all_pairs = dict(nx.all_pairs_dijkstra_path_length(self.graph, weight='dist'))
        
        for u, targets in all_pairs.items():
            for v, dist in targets.items(): self.dist_matrix[int(u), int(v)] = dist
            
        # Vettore dei 'golds' per accesso veloce
        self.golds = np.zeros(self.num_nodes, dtype=np.int32)
        for n in self.graph.nodes():
            try: self.golds[n] = int(self.graph.nodes[n]['gold'])
            except: pass
            
        # Configurazione del limite di lookahead per il Prins Split (ottimizzazione velocità)
        if self.num_nodes > 200: self.split_limit = 50 
        elif self.beta <= 1.0 + 1e-9: self.split_limit = min(self.num_nodes, 150)
        else: self.split_limit = 25 

    def evaluate(self, sequence):
        """Calcola la fitness (costo) di un individuo usando il Prins Split ottimizzato."""
        seq_arr = np.array(sequence, dtype=np.int32)
        return optimal_split_cost(seq_arr, self.dist_matrix, self.golds, self.alpha, self.beta, self.split_limit)
    
    def apply_smart_lns(self, sequence, pct):
        """Applica la mutazione Large Neighborhood Search."""
        seq_arr = np.array(sequence, dtype=np.int32)
        # Prima ricostruiamo come il tour è attualmente diviso in viaggi (P)
        P = reconstruct_split(seq_arr, self.dist_matrix, self.golds, self.alpha, self.beta, self.split_limit)
        num_rm = max(2, int(len(sequence) * pct))
        return lns_destroy_repair(seq_arr, P, self.dist_matrix, self.golds, self.alpha, self.beta, num_rm)
    
    def apply_crossover(self, p1, p2): 
        return iox_crossover(p1, p2)
    
    def apply_2opt(self, sequence): 
        seq_arr = np.array(sequence, dtype=np.int32)
        return fast_2opt(seq_arr, self.dist_matrix)
    
    def apply_inversion(self, sequence): 
        seq_arr = np.array(sequence, dtype=np.int32)
        return inversion_mutation(seq_arr)
    
    def apply_swap(self, sequence): 
        seq_arr = np.array(sequence, dtype=np.int32)
        return swap_mutation(seq_arr)
    
    def create_nn_ind(self, valid_nodes): 
        nodes_arr = np.array(valid_nodes, dtype=np.int32)
        return nearest_neighbor_init(nodes_arr, self.dist_matrix)
    
    def _get_live_path(self, u, v):
        """Recupera il percorso nodo-per-nodo dal grafo originale."""
        if u == v: return [u]
        try: return nx.shortest_path(self.graph, u, v, weight='dist')
        except: return None if self.graph.is_directed() else nx.shortest_path(self.graph, v, u, weight='dist')[::-1]
        
    def build_solution(self, tour):
        """
        Trasforma il Giant Tour (sequenza di clienti) nella soluzione finale richiesta:
        una lista di tuple [(nodo, gold_raccolto), ...].
        Usa 'reconstruct_split' per sapere dove tagliare i viaggi (ritorni al deposito).
        """
        tour_arr = np.array(tour, dtype=np.int32)
        P = reconstruct_split(tour_arr, self.dist_matrix, self.golds, self.alpha, self.beta, self.split_limit)
        final_path = []
        n = len(tour); curr = n; segments = []
        
        # Backtracking per ricostruire i segmenti di viaggio
        while curr > 0:
            start_idx = P[curr] - 1; segment = tour[start_idx : curr]; segments.append(segment); curr = start_idx
        segments = segments[::-1]
        
        for trip in segments:
            curr_real = 0 # Si parte sempre dal deposito (0)
            for node in trip:
                gold = self.golds[node]
                # Viaggio verso il nodo
                path_segment = self._get_live_path(curr_real, node)
                if path_segment:
                    # Registriamo il movimento; solo all'arrivo raccogliamo l'oro
                    for step in path_segment[1:]: final_path.append((step, gold if step == node else 0))
                curr_real = node
            
            # Ritorno al deposito alla fine del viaggio
            path_home = self._get_live_path(curr_real, 0)
            if path_home:
                for step in path_home[1:]: final_path.append((step, 0))
        return final_path
    
def calculate_real_total_cost(full_path, dist_matrix, alpha, beta):
    """
    Calcola il costo finale della soluzione completa (inclusi i percorsi fissi).
    Serve per verificare la qualità della soluzione 'intera'.
    """
    cost = 0.0
    load = 0.0
    curr = 0
    for node, gold in full_path:
        d = dist_matrix[curr, node]
        # USA LA FUNZIONE CENTRALIZZATA DI UTILS per coerenza matematica
        cost += cost_func(d, load, alpha, beta)
        load += gold
        curr = node
        # Se siamo al deposito, scarichiamo tutto
        if node == 0: load = 0.0
        
    # Safety check: se il percorso non finisce a 0, aggiungiamo il ritorno
    if curr != 0: 
        d_home = dist_matrix[curr, 0]
        cost += cost_func(d_home, load, alpha, beta)
    return cost