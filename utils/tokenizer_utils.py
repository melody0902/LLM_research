class TokenizerUtils:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def decode_token_ids(self, token_ids):
        """將單一 token ID 或 token ID 列表解碼為文字"""
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)
    
    def analyze_tokenization(self, text):
        """分析一段文字的 tokenizer 分詞行為"""
        encoded = self.tokenizer.encode(text, add_special_tokens=False)
        tokens = self.tokenizer.convert_ids_to_tokens(encoded)
        
        results = []
        for i, (tid, tok) in enumerate(zip(encoded, tokens)):
            decoded = self.tokenizer.decode([tid])
            results.append({"index": i, "token_id": tid, "token": tok, "decoded": decoded})
        return results
    
    def split_prompt_and_natural_text(self, text, prompt_token_length=30):
        """將文字分割為 prompt 和 natural_text"""
        encoded = self.tokenizer.encode(text, add_special_tokens=False)
        
        prompt_ids = encoded[:prompt_token_length]
        natural_ids = encoded[prompt_token_length:]
        
        prompt_text = self.tokenizer.decode(prompt_ids, skip_special_tokens=True)
        natural_text = self.tokenizer.decode(natural_ids, skip_special_tokens=True)
        
        return prompt_text, natural_text
    
    def get_top_tokens_by_count(self, json_file_path, top_n=10):
        """從 JSON 檔案中獲取出現次數最多的 token"""
        import json
        
        with open(json_file_path, 'r') as file:
            data = json.load(file)
        
        tokens_data = [(id, item['total_tokens']) for id, item in data.items()]
        sorted_data = sorted(tokens_data, key=lambda x: x[1], reverse=True)
        
        top_results = sorted_data[:top_n]
        return [(id, count, self.decode_token_ids(int(id))) for id, count in top_results]