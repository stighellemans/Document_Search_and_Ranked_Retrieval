from helpers import tokens_to_term_freqs
from preprocessing import tokenize

# process the query: tokenize, calculate term frequencies, normalize
def process_query(query: str) -> dict:
    # step 1: tokenize the query
    tokens = tokenize(query)
    
    # step 2: calculate term frequencies using the helper function
    term_freqs = tokens_to_term_freqs(tokens)
    
    # step 3: normalize query term frequencies
    normalized_tfs = normalize(term_freqs)
    
    return normalized_tfs

# normalize the term frequencies by total term count
def normalize(term_freqs: dict) -> dict:
    total_terms = sum(term_freqs.values())
    return {term: tf / total_terms for term, tf in term_freqs.items()}