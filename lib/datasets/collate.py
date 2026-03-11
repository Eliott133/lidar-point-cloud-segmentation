def custom_collate_fn(batch):
    """
    Fonction de rassemblement pour gérer les nuages de points de tailles variables.
    """
    batched_data = {
        'points': [item['points'] for item in batch],
        'remissions': [item['remissions'] for item in batch]
    }
    
    if 'semantics' in batch[0]:
        batched_data['semantics'] = [item['semantics'] for item in batch]
        batched_data['instances'] = [item['instances'] for item in batch]
        
    return batched_data