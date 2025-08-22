import pandas as pd

def get_key_word_attentions(key_words: list[str], results: list[pd.DataFrame]):
    attention_results = {}

    for token in key_words:
        head_appearances = 0
        total_token_attention = 0
        
        print(len(results))
        print(results[2][results[2]["key_token"] == "_goblin"])
        
        for i in range(len(results)):
            token_rows = results[i][results[i]["key_token"] == token]
            
            if not token_rows.empty:
                total_token_attention += token_rows["attention"].mean()
                head_appearances += 1
        
        avg_token_attention = total_token_attention / max(1, head_appearances)
        
        attention_results[token] = {
            "avg_attention": avg_token_attention,
            "head_appearances": head_appearances
        }
    
    return attention_results 
    