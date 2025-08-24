import pandas as pd
import json, gzip
from pathlib import Path
from typing import Dict, List, Any, Iterable, Optional, Callable

def get_key_word_attentions(key_words: list[str], results: list[pd.DataFrame]):
    attention_results = {}

    for token in key_words:
        head_appearances = 0
        total_token_attention = 0
        
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

def load_prompts_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    records = []
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            s = line.strip()
            if not s:
                continue
            try:
                rec = json.loads(s)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{ln}: invalid JSON: {e}") from e
            
            records.append({
                "prompt_id": rec.get("id"),
                "prompt_len": rec.get("prompt_len"),
                "prompt": rec.get("prompt")
            })
    return records

    