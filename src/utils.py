from numba import njit

@njit(inline="always")
def cost_func(dist: float, weight: float, alpha: float, beta: float) -> float:
    """
    Calcola il costo di attraversamento di un arco secondo la formula del problema.
    
    Formula: Cost = dist + (dist * alpha * weight)^beta
    
    Decoratore: @njit(inline="always") forza il compilatore a inserire il codice 
    direttamente nel punto di chiamata (come una macro in C), eliminando l'overhead 
    della chiamata a funzione. Fondamentale per le performance dentro i loop.
    """
    return dist + (dist * alpha * weight) ** beta

@njit(inline="always")
def optimal_fraction_size(alpha: float, beta: float, distance: float) -> float:
    """
    Calcola la dimensione ottimale del carico (chunk) per minimizzare il costo 
    totale quando si fanno viaggi avanti-indietro dedicati (Bulk Removal).
    
    Teoria:
    Se Beta > 1, il costo cresce in modo super-lineare rispetto al peso.
    Invece di portare tutto l'oro in una volta (costo esplosivo), conviene 
    spezzarlo in piccoli carichi 'x' e fare più viaggi.
    
    Questa funzione trova la 'x' che minimizza la derivata della funzione di costo totale.
    
    Returns:
        float: La quantità di oro ottimale da portare per viaggio.
               Ritorna 1e9 (infinito) se Beta <= 1, perché in quel caso conviene 
               sempre portare tutto insieme.
    """
    # Se la penalità è lineare o sub-lineare, conviene portare il massimo possibile.
    # Ritorniamo un valore altissimo per indicare "nessun limite".
    if beta <= 1.0 + 1e-9: return 1e9
    
    # Derivazione matematica dell'ottimo:
    # Costo Totale ~ (Gold / x) * [ 2*dist + (dist * alpha * x)^beta ]
    # Ponendo la derivata rispetto a x = 0, si ottiene la formula seguente.
    
    term1 = 1.0 / alpha
    numerator = 2.0 * (distance ** (1.0 / beta))
    denominator = beta - 1.0
    
    if denominator <= 0: return 1e9
    
    base = numerator / denominator
    
    # Safety check per radici negative (non dovrebbe accadere con dist > 0)
    if base < 0: return 1e9
    
    return term1 * (base ** (1.0 / beta))