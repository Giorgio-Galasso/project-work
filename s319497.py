import random
import numpy as np
from tqdm import tqdm
from Problem import Problem

# Importa le classi e funzioni necessarie dalla cartella src/
# SmartEvaluator: Gestisce la valutazione della fitness e l'interfaccia con Numba
# preprocess_bulk_removal: Strategia euristica per beta > 1 (viaggi dedicati)
# calculate_real_total_cost: Calcola il costo totale 'vero' per monitorare i progressi
from src.evaluator import (
    SmartEvaluator, 
    preprocess_bulk_removal, 
    calculate_real_total_cost
)

def solution(p: Problem):
    """
    Funzione principale che risolve il problema di routing.
    
    Args:
        p (Problem): Istanza del problema generata da Problem.py.
        
    Returns:
        list: Lista di tuple [(nodo, gold_raccolto), ...] che rappresenta il percorso ottimo.
    """
    
    # ==========================================
    # 1. PREPROCESSING (Strategia Bulk Removal)
    # ==========================================
    # Se Beta > 1, il peso trasportato ha un impatto enorme sul costo.
    # In questi casi, conviene servire i nodi con molto oro facendo viaggi dedicati (spoke-hub).
    # Questa funzione identifica tali nodi e crea una parte di percorso "fisso".
    fixed_path, residual_problem = preprocess_bulk_removal(p)
    
    # Identifica i nodi che hanno ancora oro da raccogliere dopo il preprocessing
    valid_nodes = [n for n in residual_problem.graph.nodes() if n != 0 and residual_problem.graph.nodes[n]['gold'] > 0]
    
    # Se non ci sono più nodi validi (tutto risolto dal preprocessore o grafo vuoto),
    # ritorniamo subito il percorso fisso calcolato.
    if not valid_nodes:
        return fixed_path
    
    # ==========================================
    # 2. CONFIGURAZIONE PARAMETRI (Adattiva)
    # ==========================================
    evaluator = SmartEvaluator(residual_problem)
    beta = residual_problem.beta
    N = len(valid_nodes) + 1 
    IS_HUGE = N >= 500  # Flag per gestire istanze massive
    
    # Adattiamo i parametri dell'Algoritmo Genetico in base alla natura del problema
    if beta > 1.0:
        # SCENARIO ALTA PENALITÀ CARICO (Beta > 1):
        # Il problema è dominato dalla gestione del peso, non dalla geometria.
        # Riduciamo l'esplorazione geometrica (LNS inutile) e ci concentriamo 
        # sull'ordine di visita per scaricare prima i nodi pesanti.
        MU = int(max(10, N * 0.25)); LAMBDA = MU * 2
        MAX_GENERATIONS = int(max(10, N * 0.20))
        ELITISM = int(max(1, MU * 0.15))
        USE_LNS = False; LNS_PCT = 0.0
        EARLY_STOP_LIMIT = 50 
    else:
        # SCENARIO LINEARE O SUB-LINEARE (Beta <= 1):
        # Il problema è simile a un TSP/CVRP classico. La geometria conta molto.
        # Usiamo parametri standard robusti e attiviamo Large Neighborhood Search (LNS).
        if IS_HUGE: 
            # Parametri conservativi per istanze giganti (per stare nei tempi)
            MU = 30; LAMBDA = 60
            MAX_GENERATIONS = 300
            EARLY_STOP_LIMIT = 50
        else: 
            # Parametri aggressivi per istanze normali
            MU = 50; LAMBDA = 100
            MAX_GENERATIONS = 1000
            EARLY_STOP_LIMIT = 250
            
        ELITISM = 5
        USE_LNS = True; LNS_PCT = 0.20; LNS_PROB = 0.5 
    
    # ==========================================
    # 3. INIZIALIZZAZIONE POPOLAZIONE
    # ==========================================
    population = []
    attempts = 0
    # Proviamo a riempire la popolazione fino a MU individui validi
    while len(population) < MU and attempts < MU * 5:
        attempts += 1
        
        # Strategia Ibrida:
        # - 80% Nearest Neighbor (euristica golosa) -> Buona partenza geometrica
        # - 20% Random -> Garantisce diversità genetica
        if beta <= 1.0 and random.random() < 0.8: 
            seq = evaluator.create_nn_ind(valid_nodes)
            # Piccola perturbazione per non avere tutti cloni identici
            if random.random() < 0.5: seq = evaluator.apply_inversion(seq)
        else:
            seq = np.array(valid_nodes, dtype=np.int32)
            np.random.shuffle(seq)
            
        # Refinement iniziale veloce (solo per istanze gestibili)
        if beta <= 1.0 and not IS_HUGE: 
            seq = evaluator.apply_2opt(seq)
            
        # Valutazione fitness (costo del percorso partizionato ottimamente)
        cost = evaluator.evaluate(seq)
        if cost != float('inf'): population.append((seq, cost))
    
    # Fallback di sicurezza: se non riusciamo a generare nulla, torniamo il percorso base
    if not population:
        return fixed_path
        
    # Ordiniamo la popolazione per costo (migliori prima)
    population.sort(key=lambda x: x[1])
    population = population[:MU]
    
    # ==========================================
    # 4. LOOP PRINCIPALE (EVOLUZIONE)
    # ==========================================
    no_improve_counter = 0
    global_real_best_cost = float('inf')
    global_best_ind = population[0]
    
    # Setup barra di progresso (tqdm)
    try:
        # Calcolo densità solo per info a schermo
        d_val = getattr(p, 'density', None)
        if d_val is None:
            num_nodes = len(p.graph.nodes())
            num_edges = len(p.graph.edges())
            if num_nodes > 1: d_val = (2 * num_edges) / (num_nodes * (num_nodes - 1))
            else: d_val = 0.0
        dens_disp = f"{d_val:.1f}"
    except: dens_disp = "NA"
    
    pbar_desc = f"GA (N={N}, A={residual_problem.alpha}, B={beta}, D={dens_disp})"
    pbar = tqdm(range(MAX_GENERATIONS), desc=pbar_desc, leave=False)
    
    for gen in pbar:
        offspring = []
        
        # ELITISMO: I migliori individui passano direttamente alla generazione successiva
        offspring.extend(population[:ELITISM])
        
        # Generazione della prole fino a riempire LAMBDA
        while len(offspring) < LAMBDA:
            # SELEZIONE (Tournament Selection)
            # Scegliamo 2 genitori confrontando candidati casuali
            t_size = 4
            best_i = random.randint(0, len(population)-1)
            for _ in range(t_size-1):
                idx = random.randint(0, len(population)-1)
                if population[idx][1] < population[best_i][1]: best_i = idx
            p1 = population[best_i][0]
            
            best_i2 = random.randint(0, len(population)-1)
            for _ in range(t_size-1):
                idx = random.randint(0, len(population)-1)
                if population[idx][1] < population[best_i2][1]: best_i2 = idx
            p2 = population[best_i2][0]
            
            # CROSSOVER (IOX - Inter-route Optimization Crossover)
            child_seq = evaluator.apply_crossover(p1, p2)
            
            # MUTAZIONE IBRIDA
            # - LNS (Large Neighborhood Search): Distrugge e ripara parti del percorso (più costoso ma efficace)
            # - Mutazioni semplici (Swap/Inversion): Veloci perturbazioni locali
            if USE_LNS and random.random() < LNS_PROB: 
                child_seq = evaluator.apply_smart_lns(child_seq, LNS_PCT)
            elif random.random() < 0.1:
                if random.random() < 0.5: child_seq = evaluator.apply_inversion(child_seq)
                else: child_seq = evaluator.apply_swap(child_seq)
            
            # LOCAL SEARCH (2-opt)
            # Ottimizza geometricamente il figlio. Applicata con probabilità ridotta su istanze grandi.
            prob_2opt = 0.05 if IS_HUGE else 1.0
            if beta <= 1.0 and random.random() < prob_2opt: 
                child_seq = evaluator.apply_2opt(child_seq)
            
            # Valutazione del figlio
            cost = evaluator.evaluate(child_seq)
            offspring.append((child_seq, cost))
            
        # SELEZIONE SOPRAVVIVENTI (Mu + Lambda)
        # Uniamo genitori e figli, ordiniamo e teniamo solo i migliori MU
        total = population + offspring
        total.sort(key=lambda x: x[1])
        population = total[:MU]
        
        current_pop_best = population[0]
        
        # VERIFICA MIGLIORAMENTO GLOBALE
        # Ricostruiamo la soluzione completa per calcolare il costo reale esatto
        ga_path = evaluator.build_solution(current_pop_best[0])
        full_path = fixed_path + ga_path
        real_cost_now = calculate_real_total_cost(full_path, evaluator.dist_matrix, residual_problem.alpha, beta)
        
        # Se troviamo una nuova soluzione migliore globale, resettiamo il contatore di stagnazione
        if real_cost_now < global_real_best_cost - 1e-6:
            global_real_best_cost = real_cost_now
            global_best_ind = current_pop_best
            no_improve_counter = 0
        else:
            no_improve_counter += 1
            
        # EARLY STOPPING: Interrompiamo se non miglioriamo per troppe generazioni
        if no_improve_counter >= EARLY_STOP_LIMIT: break
        
        # Aggiornamento barra (ogni 5 gen per non rallentare)
        if gen % 5 == 0: 
            pbar.set_postfix({"Best": f"{global_real_best_cost:.1f}", "NoImp": no_improve_counter})
    
    # ==========================================
    # 5. RICOSTRUZIONE PERCORSO FINALE
    # ==========================================
    best_ga_path = []
    if global_best_ind is not None:
        # Trasformiamo il cromosoma (sequenza di clienti) in percorso completo (con ritorni al deposito)
        best_ga_path = evaluator.build_solution(global_best_ind[0])
    
    # Concateniamo i viaggi fissi (Bulk Removal) con quelli ottimizzati dal GA
    final_path = fixed_path + best_ga_path
    
    return final_path