import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer

class Model:
    def __init__(self, name: str, token: str = None):
        self.tokenizer = AutoTokenizer.from_pretrained(name, use_fast=False)
        self.model = AutoModelForCausalLM.from_pretrained(
            name,
            device_map="auto",
            torch_dtype=torch.float16, 
            trust_remote_code=True,
            attn_implementation="eager" 
        )

        try:
            self.model.config.attn_implementation = "eager"
        except Exception:
            print("can't do eager")
            pass

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def invoke_model(self, query: str, max_new_tokens: int = 128, chat: bool = True):
        if chat and hasattr(self.tokenizer, "apply_chat_template"):
            messages = [{"role": "user", "content": query}]
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            prompt = query

        input_dict = {}
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_tokens = inputs
        
        for k, v in inputs.items():
            input_dict[k] = v.to(self.model.device)
            
        inputs = input_dict

        with torch.no_grad():
            gen = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                return_dict_in_generate=True,
                output_attentions=True,
                output_scores=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        text_outputs = self.tokenizer.decode(gen.sequences[0], skip_special_tokens=True)
        
        gen_attentions = gen.attentions

        with torch.no_grad():
            fwd_out = self.model(**inputs, output_attentions=True)

        prompt_attentions = fwd_out.attentions

        return {
            "inputs": inputs,
            "fwd_out": fwd_out,
            "text_outputs": text_outputs,
            "prompt_attn": prompt_attentions,
            "gen_attn": gen_attentions,
            "gen_tokens": gen.sequences[0],
            "input_tokens": input_tokens
        }
        
    def get_model_output(self, results: dict):
        input_ids = results["inputs"]["input_ids"]
        if "attention_mask" in results["inputs"]:
            input_len = int(results["inputs"]["attention_mask"][0].sum().item())
        else:
            input_len = int(input_ids.shape[1])

        seq = results.get("gen_sequences", results.get("gen_tokens"))
        if hasattr(seq, "dim") and seq.dim() == 2:
            seq = seq[0]
        if hasattr(seq, "tolist"):
            seq = seq.tolist()

        new_ids = seq[input_len:]
        new_text = self.tokenizer.decode(new_ids, skip_special_tokens=True)

        return {
            "generated_tokens": new_ids,
            "generated_text": new_text
        }


    
    def prompt_attention(self, results: dict, layer: int, head: int, top_n: int):
        attention = results["prompt_attn"]
        attn_mat_tensor = attention[layer][0, head]
        attn_mat = attn_mat_tensor.detach().cpu().numpy()
        
        input_ids = results["inputs"]["input_ids"][0].tolist()
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
        tokens_len = len(tokens)

        rows, cols = attn_mat.shape
        
        pairs = []
        
        for i in range(rows):
            for j in range(cols):
                w = attn_mat[i, j]
                
                q_tok = tokens[i] if (i < len(tokens)) else f"context[{i}]"
                k_tok = tokens[j] if (j < len(tokens)) else f"context[{j}]"
                
                    
                pairs.append((w, q_tok, k_tok))
        pairs.sort(key=lambda x: x[0], reverse=True)
        
        for w, q_tok, k_tok in pairs[:top_n]:
            print(f"{q_tok:<12} → {k_tok:<12} | attn: {w:.4f}")

        return {
            "text": results["text_outputs"],
            "attn": attn_mat,
            "tokens_source_labels": tokens,
            "shape": attn_mat.shape,
            "input_length": tokens_len
        }
    
    def gen_attention(self, results: dict, layer: int, head: int, top_n: int, gen_step: int = -1):
        step_idx = gen_step if gen_step >= 0 else (len(results["gen_attn"]) - 1)
        step_tuple = results["gen_attn"][step_idx]
        layer_attn = step_tuple[layer]
        attn_mat_tensor = layer_attn[0, head]
        attn_mat = attn_mat_tensor.detach().cpu().numpy()
        
        input_ids = results["inputs"]["input_ids"][0].tolist()
        input_len = len(input_ids)
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
    
        rows, cols = attn_mat.shape
        
        pairs = []
        for i in range(rows):
            for j in range(cols):
                w = attn_mat[i, j]
                
                q_tok = f"gen_txt[{step_idx}]"
                k_tok = tokens[j] if j < len(tokens) else f"context[{j}]"
        
                pairs.append((w, q_tok, k_tok))
        pairs.sort(key=lambda x: x[0], reverse=True)
        
        for w, q_tok, k_tok in pairs[:top_n]:
            print(f"{q_tok:<12} → {k_tok:<12} | attn: {w:.4f}")

        return {
            "text": results["text_outputs"],
            "attn": attn_mat,
            "tokens_source_labels": tokens,
            "shape": attn_mat.shape,
            "input_length": input_len
        }
        
    def gen_attention_all_steps(self, results: dict, layer: int, head: int):
        num_gen_steps = len(results["gen_attn"])
        
        gen_attn_result = []
        
        for gen_step in range(num_gen_steps):
            
            print(f"Generation step {gen_step + 1} \n")
            
            step_tuple = results["gen_attn"][gen_step]
            layer_attn = step_tuple[layer]
            attn_mat_tensor = layer_attn[0, head]
            attn_mat = attn_mat_tensor.detach().cpu().numpy()
            
            input_ids = results["inputs"]["input_ids"][0].tolist()
            input_len = len(input_ids)
            tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
        
            rows, cols = attn_mat.shape
            
            pairs = []
            for i in range(rows):
                for j in range(cols):
                    w = attn_mat[i, j]
                    
                    q_tok = f"gen_txt[{gen_step}]"
                    k_tok = tokens[j] if j < len(tokens) else f"context[{j}]"
            
                    pairs.append((w, q_tok, k_tok))
            pairs.sort(key=lambda x: x[0], reverse=True)
            
            for w, q_tok, k_tok in pairs[:5]:
                print(f"{q_tok:<12} → {k_tok:<12} | attn: {w:.4f}")
            
            print("\n")
            
            curr_gen_step_result =  {
                "text": results["text_outputs"],
                "attn": attn_mat,
                "tokens_source_labels": tokens,
                "shape": attn_mat.shape,
                "input_length": input_len
            }
            
            gen_attn_result.append(curr_gen_step_result)
        
        return gen_attn_result

    def prompt_attn_all_heads(self, results: dict, layer: int, num_heads: int):
        
        results_list = []
        
        for curr_head in range(num_heads):
            print(f"HEAD {curr_head + 1}\n")
            
            prompt_attn_results = self.prompt_attention(results, layer, curr_head, 5)
            results_list.append(prompt_attn_results)
            
            print("\n")
        
        return results_list

    def gen_attn_all_heads(self, results: dict, layer: int, num_heads: int, gen_step: int = -1):
        
        results_list = []
        
        for curr_head in range(num_heads):
            print(f"HEAD {curr_head + 1}\n")
            
            
            gen_attn_results = self.gen_attention(results, layer, curr_head, 5)
            results_list.append(gen_attn_results)
            
            print("\n")
        
        return results_list

    def gen_attn_all_heads_all_gen_steps(self, results: dict, layer: int, num_heads: int):
        pd_results = []
        num_gen_steps = len(results["gen_attn"])
        
        for curr_head in range(num_heads):
            curr_rows = []
            
            for gen_step in range(num_gen_steps):
                
                step_tuple = results["gen_attn"][gen_step]
                layer_attn = step_tuple[layer]
                attn_mat_tensor = layer_attn[0, curr_head]
                attn_mat = attn_mat_tensor.detach().cpu().numpy()
                
                input_ids = results["inputs"]["input_ids"][0].tolist()
                input_len = len(input_ids)
                tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
            
                rows, cols = attn_mat.shape
                
                pairs = []
                for i in range(rows):
                    for j in range(cols):
                        w = attn_mat[i, j]
                        
                        q_tok = f"gen_txt[{gen_step}]"
                        k_tok = tokens[j] if j < len(tokens) else f"context[{j}]"
                
                        pairs.append((w, q_tok, k_tok))
                pairs.sort(key=lambda x: x[0], reverse=True)
                
                for w, q_tok, k_tok in pairs[:40]:
                    curr_rows.append((gen_step, w, q_tok, k_tok))
                    
            curr_df = pd.DataFrame(curr_rows, columns=["gen_step", "attention", "query_token", "key_token"])
            pd_results.append(curr_df)
        
        return pd_results
