# =============================================
# ngram.py
# Description: n-gram signature collection and detection
# =============================================

import json
from typing import Optional, Set, List, Dict, Tuple, Any, Union
from visualize.data_for_visualization import DataForVisualization
from watermark.kgw.kgw import KGW
from watermark.sweet.sweet import SWEET
from watermark.unigram.unigram import Unigram
from watermark.signature.signature import SignatureSetCollector, KGWSignature, SweetSignature, UnigramSignature, WatermarkTokenAnalyzer
import os
import torch


class NGramSignatureSetUtils:
    @staticmethod
    def load(file_path: str) -> Tuple[int, Set[Tuple[int, ...]]]:
        """
        從文件加載 n-gram 簽名集。
        
        Args:
            file_path: 簽名集文件路徑
            
        Returns:
            Tuple[int, Set[Tuple[int, ...]]]: n 值和簽名集
            
        Raises:
            FileNotFoundError: 如果文件不存在
            json.JSONDecodeError: 如果 JSON 格式錯誤
        """
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            n = data.get("n", 3)  # 默認n為3
            ngram_signature_set = {tuple(ngram) for ngram in data.get("signatures", [])}
            
            print(f"已從 {file_path} 加載 {len(ngram_signature_set)} 個 {n}-gram 簽名")
            return n, ngram_signature_set
        except FileNotFoundError:
            raise FileNotFoundError(f"文件不存在: {file_path}")
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(f"JSON 格式錯誤: {e.msg}", e.doc, e.pos)
    
    @staticmethod
    def save(ngram_signature_set: Set[Tuple[int, ...]], n: int, save_path: str) -> None:
        """
        保存 n-gram 簽名集到文件。
        
        Args:
            ngram_signature_set: n-gram 簽名集
            n: n-gram 的 n 值
            save_path: 保存路徑
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # 將 tuple 轉換為可序列化的列表
        saveable_signatures = [list(ngram) for ngram in ngram_signature_set]
        
        with open(save_path, 'w') as f:
            json.dump({
                "n": n,
                "signatures": saveable_signatures
            }, f)
        print(f"已保存 {len(ngram_signature_set)} 個 {n}-gram 簽名到 {save_path}")


class NGramSignatureSetCollector(SignatureSetCollector):
    """
    用於收集和管理 n-gram 簽名集合的工具類。
    
    收集生成式水印中連續 n 個或更多的「紅字」tokens，用於後續檢測時提高準確性。
    """
    
    def __init__(self, watermark, n=3) -> None:
        """
        初始化 n-gram 簽名收集器。
        
        Args:
            watermark: 水印系統實例，用於獲取綠名單和其他信息
            n: 連續紅字的最小長度
        """
        super().__init__(watermark)
        self.n = n
        self.ngram_signature_set: Set[Tuple[int, ...]] = set()  # 存儲 n-gram 簽名
    
    def collect_from_text(self, text: str) -> None:
        """
        從單一文本收集符合 n-gram 條件的連續紅字。
        
        Args:
            text: 要分析的文本
        """
        encoded_text = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0].to(self.device)
        
        if isinstance(self.watermark, KGW):
            self._collect_ngram_from_kgw(encoded_text)
        elif isinstance(self.watermark, SWEET):
            self._collect_ngram_from_sweet(encoded_text)
        elif isinstance(self.watermark, Unigram):
            self._collect_ngram_from_unigram(encoded_text)
        else:
            raise NotImplementedError(f"不支援的水印類型: {type(self.watermark).__name__}")
    
    def _collect_ngram_from_kgw(self, encoded_text: torch.LongTensor) -> None:
        """從 KGW 水印文本中收集符合 n-gram 條件的連續紅字。"""
        # 1. 標記每個位置是紅字還是綠字
        red_flags = []
        for idx in range(self.prefix_length, len(encoded_text)):
            curr_token = encoded_text[idx].item()
            greenlist_ids = self.watermark.utils.get_greenlist_ids(encoded_text[:idx])
            red_flags.append(curr_token not in greenlist_ids)
        
        # 2. 收集連續 n 個或更多的紅字序列
        current_seq = []
        for idx, is_red in enumerate(red_flags):
            if is_red:
                # 如果是紅字，加入當前序列
                current_seq.append(encoded_text[idx + self.prefix_length].item())
                
                # 如果序列長度達到 n，就提取一個新的 n-gram
                if len(current_seq) >= self.n:
                    ngram = tuple(current_seq[-self.n:])  # 取最後 n 個元素
                    self.ngram_signature_set.add(ngram)
            else:
                # 遇到綠字時重置序列
                current_seq = []
    
    def _collect_ngram_from_sweet(self, encoded_text: torch.LongTensor) -> None:
        """
        從 SWEET 水印文本中收集符合 n-gram 條件的連續紅字。
        
        SWEET 的紅字判定需要同時滿足：
        1. 不在綠名單中
        2. 熵值高於閾值
        """
        # 1. 計算熵值
        entropy_list = self.watermark.utils.calculate_entropy(
            self.watermark.config.generation_model, 
            encoded_text
        )
        
        # 2. 標記每個位置是紅字還是綠字
        red_flags = []
        for idx in range(self.prefix_length, len(encoded_text)):
            curr_token = encoded_text[idx].item()
            greenlist_ids = self.watermark.utils.get_greenlist_ids(encoded_text[:idx])
            
            # 檢查熵值是否高於閾值
            is_high_entropy = idx < len(entropy_list) and entropy_list[idx] > self.watermark.config.entropy_threshold
            
            # 同時滿足：不在綠名單且熵值高
            red_flags.append(curr_token not in greenlist_ids and is_high_entropy)
        
        # 3. 收集連續 n 個或更多的紅字序列
        current_seq = []
        for idx, is_red in enumerate(red_flags):
            if is_red:
                # 如果是紅字，加入當前序列
                current_seq.append(encoded_text[idx + self.prefix_length].item())
                
                # 如果序列長度達到 n，就提取一個新的 n-gram
                if len(current_seq) >= self.n:
                    ngram = tuple(current_seq[-self.n:])  # 取最後 n 個元素
                    self.ngram_signature_set.add(ngram)
            else:
                # 遇到綠字時重置序列
                current_seq = []
    
    def _collect_ngram_from_unigram(self, encoded_text: torch.LongTensor) -> None:
        """ 從 Unigram 水印文本中收集符合 n-gram 條件的連續紅字。"""
        # 1. 標記每個位置是紅字還是綠字
        red_flags = []
        for idx in range(len(encoded_text)):
            curr_token = encoded_text[idx].item()
            red_flags.append(not self.watermark.utils.mask[curr_token])
        
        # 2. 收集連續 n 個或更多的紅字序列
        current_seq = []
        for idx, is_red in enumerate(red_flags):
            if is_red:
                # 如果是紅字，加入當前序列
                current_seq.append(encoded_text[idx].item())
                
                # 如果序列長度達到 n，就提取一個新的 n-gram
                if len(current_seq) >= self.n:
                    ngram = tuple(current_seq[-self.n:])  # 取最後 n 個元素
                    self.ngram_signature_set.add(ngram)
            else:
                # 遇到綠字時重置序列
                current_seq = []
    
    def save_ngram_signature_set(self, save_path: str) -> None:
        """保存 n-gram 簽名集到文件"""
        NGramSignatureSetUtils.save(self.ngram_signature_set, self.n, save_path)
    
    def load_ngram_signature_set(self, file_path: str) -> None:
        """從文件加載 n-gram 簽名集"""
        self.n, self.ngram_signature_set = NGramSignatureSetUtils.load(file_path)


class KGWNGramSignature(KGWSignature):  
    """
    KGW水印的 n-gram 簽名感知版本，根據連續紅字規則進行檢測。
    """
    
    def __init__(
        self, 
        algorithm_config: str, 
        transformers_config: Optional[Any] = None, 
        n: int = 3,
        signature_set: Optional[Set[int]] = None, 
        signature_file: Optional[str] = None,
        ngram_signature_set: Optional[Set[Tuple[int, ...]]] = None, 
        ngram_signature_file: Optional[str] = None, 
        *args, 
        **kwargs
    ) -> None:
        """
        初始化 n-gram 簽名感知的KGW水印。
        
        Args:
            algorithm_config: 算法配置文件路徑或配置對象
            transformers_config: Transformers配置
            n: 連續紅字的最小長度
            signature_set: 簽名集合
            signature_file: 簽名文件路徑
            ngram_signature_set: n-gram 簽名集合
            ngram_signature_file: n-gram 簽名文件路徑
        """
        super().__init__(algorithm_config, transformers_config, signature_set, signature_file, *args, **kwargs)
        
        self.n = n
        self.ngram_signature_set: Set[Tuple[int, ...]] = set()
        
        if ngram_signature_set:
            self.ngram_signature_set = set(ngram_signature_set)
        elif ngram_signature_file:
            self.load_ngram_signature_set(ngram_signature_file)

    def load_ngram_signature_set(self, file_path: str) -> None:
        """從文件加載 n-gram 簽名集"""
        self.n, self.ngram_signature_set = NGramSignatureSetUtils.load(file_path)
    
    def save_ngram_signature_set(self, save_path: str) -> None:
        """保存 n-gram 簽名集到文件"""
        NGramSignatureSetUtils.save(self.ngram_signature_set, self.n, save_path)
    
    def detect_watermark(self, text: str, return_dict: bool = True, *args, **kwargs) -> Union[Dict[str, Any], Tuple[bool, float]]:
        """使用 n-gram 規則進行水印檢測"""
        encoded_text = self.config.generation_tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0].to(self.config.device)
        
        z_score, green_tokens = self.ngram_score_sequence(encoded_text)
        
        is_watermarked = z_score > self.config.z_threshold
        
        if return_dict:
            return {
                "is_watermarked": is_watermarked, 
                "score": z_score,
                "ngram_applied": True,
                "n": self.n,
                "ngram_signature_size": len(self.ngram_signature_set)
            }
        else:
            return (is_watermarked, z_score)
    
    def ngram_score_sequence(self, input_ids: torch.LongTensor) -> Tuple[float, List[int]]:
        """
        應用 n-gram 連續紅字規則進行評分。
        
        Args:
            input_ids: 編碼後的文本張量
        
        Returns:
            Tuple[float, List[int]]: z-score 值和標記列表
            標記列表中：-1 表示在 prefix 或 signature 中，1 表示綠字，0 表示紅字
        """
        if len(input_ids) == 0:
            return 0.0, []
        
        # 1. 先標記所有 token 為綠字
        token_flags = [1] * len(input_ids)
        
        # 2. 標記 prefix 為 -1
        prefix_length = self.config.prefix_length  # 從 config 獲取 prefix_length
        for i in range(min(prefix_length, len(input_ids))):
            token_flags[i] = -1
        
        # 3. 檢查每個可能的 n-gram 序列是否匹配完整的 signature
        for i in range(prefix_length, len(input_ids) - self.n + 1):
            current_ngram = tuple(input_ids[i:i+self.n].tolist())
            if current_ngram in self.ngram_signature_set:
                # 找到完整的 signature，將整個序列標記為 -1
                for j in range(i, i + self.n):
                    token_flags[j] = -1
        
        # 4. 對於未被標記為 signature 的 token，根據 greenlist 判斷紅綠字
        for i in range(prefix_length, len(input_ids)):
            if token_flags[i] != -1:  # 如果不是 signature
                # 獲取當前位置的綠名單
                greenlist = self.utils.get_greenlist_ids(input_ids[:i])
                curr_token = input_ids[i].item()
                token_flags[i] = 1 if curr_token in greenlist else 0
        
        # 5. 計算 z-score（只考慮不在 prefix 和 signature 中的 token）
        green_count = sum(1 for flag in token_flags[prefix_length:] if flag == 1)
        valid_count = sum(1 for flag in token_flags[prefix_length:] if flag != -1)

        print(f"{self.n} gram signature N: {valid_count}, {self.n} gram signature NG: {green_count}")
        
        if valid_count == 0:
            return 0.0, token_flags
        
        z_score = self.utils._compute_z_score(green_count, valid_count)
        
        return z_score, token_flags
    
    def get_data_for_visualization(self, text: str, *args, **kwargs) -> tuple[list[str], list[int]]:
        """Get data for visualization."""
        
        # Encode text
        encoded_text = self.config.generation_tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0].to(self.config.device)
        
        z_score, highlight_values = self.ngram_score_sequence(encoded_text)

        print(f'z_score: {z_score}, highlight_values: {highlight_values}, len(highlight_values): {len(highlight_values)}')
        red_ratio = sum(1 for value in highlight_values if value == 0) / len(highlight_values)
        print(f'red_ratio: {red_ratio:.2f}')
        green_ratio = sum(1 for value in highlight_values if value == 1) / len(highlight_values)
        print(f'green_ratio: {green_ratio:.2f}')
        ignore_ratio = sum(1 for value in highlight_values if value == -1) / len(highlight_values)
        print(f'ignore_ratio: {ignore_ratio:.2f}')
        
        # decode single tokens
        decoded_tokens = []
        for token_id in encoded_text:
            token = self.config.generation_tokenizer.decode(token_id.item())
            decoded_tokens.append(token)
        
        return DataForVisualization(decoded_tokens, highlight_values)    
    
    @property
    def ngram_signature_set_size(self) -> int:
        """
        返回 n-gram 簽名集大小。
        
        Returns:
            int: n-gram 簽名集中的序列數量
        """
        return len(self.ngram_signature_set)
    
class SweetNGramSignature(SweetSignature):
    """
    SWEET水印的 n-gram 簽名感知版本，根據連續紅字規則進行檢測。
    """
    
    def __init__(
        self, 
        algorithm_config: str, 
        transformers_config: Optional[Any] = None, 
        n: int = 3,
        signature_set: Optional[Set[int]] = None, 
        signature_file: Optional[str] = None,
        ngram_signature_set: Optional[Set[Tuple[int, ...]]] = None, 
        ngram_signature_file: Optional[str] = None, 
        *args, 
        **kwargs
    ) -> None:
        """
        初始化 n-gram 簽名感知的SWEET水印。
        
        Args:
            algorithm_config: 算法配置文件路徑或配置對象
            transformers_config: Transformers配置
            n: 連續紅字的最小長度
            signature_set: 簽名集合
            signature_file: 簽名文件路徑
            ngram_signature_set: n-gram 簽名集合
            ngram_signature_file: n-gram 簽名文件路徑
        """
        super().__init__(algorithm_config, transformers_config, signature_set, signature_file, *args, **kwargs)
        
        self.n = n
        self.ngram_signature_set: Set[Tuple[int, ...]] = set()
        
        if ngram_signature_set:
            self.ngram_signature_set = set(ngram_signature_set)
        elif ngram_signature_file:
            self.load_ngram_signature_set(ngram_signature_file)
    
    def load_ngram_signature_set(self, file_path: str) -> None:
        """從文件加載 n-gram 簽名集"""
        self.n, self.ngram_signature_set = NGramSignatureSetUtils.load(file_path)
    
    def save_ngram_signature_set(self, save_path: str) -> None:
        """保存 n-gram 簽名集到文件"""
        NGramSignatureSetUtils.save(self.ngram_signature_set, self.n, save_path)
    
    def detect_watermark(self, text: str, return_dict: bool = True, *args, **kwargs) -> Union[Dict[str, Any], Tuple[bool, float]]:
        """使用 n-gram 規則進行水印檢測"""
        encoded_text = self.config.generation_tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0].to(self.config.device)
        
        z_score, green_tokens = self.ngram_score_sequence(encoded_text)
        
        is_watermarked = z_score > self.config.z_threshold
        
        if return_dict:
            return {
                "is_watermarked": is_watermarked, 
                "score": z_score,
                "ngram_applied": True,
                "n": self.n,
                "ngram_signature_size": len(self.ngram_signature_set)
            }
        else:
            return (is_watermarked, z_score)
    
    def ngram_score_sequence(self, input_ids: torch.LongTensor) -> Tuple[float, List[int]]:
        """
        應用 n-gram 連續紅字規則進行評分。
        
        Args:
            input_ids: 編碼後的文本張量
        
        Returns:
            Tuple[float, List[int]]: z-score 值和標記列表
            標記列表中：-1 表示在 prefix、signature 中或是低熵字，1 表示綠字，0 表示紅字
        """
        if len(input_ids) == 0:
            return 0.0, []
        
        # 1. 先標記所有 token 為綠字
        token_flags = [1] * len(input_ids)
        
        # 2. 標記 prefix 為 -1
        prefix_length = self.config.prefix_length  # 從 config 獲取 prefix_length
        for i in range(min(prefix_length, len(input_ids))):
            token_flags[i] = -1
        
        # 3. 計算熵值
        entropy_list = self.utils.calculate_entropy(
            self.config.generation_model, 
            input_ids
        )
        
        # 4. 檢查每個可能的 n-gram 序列是否匹配完整的 signature
        for i in range(prefix_length, len(input_ids) - self.n + 1):
            current_ngram = tuple(input_ids[i:i+self.n].tolist())
            if current_ngram in self.ngram_signature_set:
                # 找到完整的 signature，將整個序列標記為 -1
                for j in range(i, i + self.n):
                    token_flags[j] = -1
        
        # 5. 對於未被標記為 signature 的 token，根據 greenlist 和熵值判斷紅綠字
        for i in range(prefix_length, len(input_ids)):
            if token_flags[i] != -1:  # 如果不是 signature
                # 檢查熵值
                is_high_entropy = i < len(entropy_list) and entropy_list[i] > self.config.entropy_threshold
                
                if not is_high_entropy:
                    # 低熵字不參與水印檢測，標記為 -1
                    token_flags[i] = -1
                    continue
                    
                # 對於高熵字，獲取當前位置的綠名單並判斷
                greenlist = self.utils.get_greenlist_ids(input_ids[:i])
                curr_token = input_ids[i].item()
                
                # 在綠名單中為綠字，否則為紅字
                token_flags[i] = 1 if curr_token in greenlist else 0
        
        # 6. 計算 z-score（只考慮高熵且不在 prefix 和 signature 中的 token）
        green_count = sum(1 for flag in token_flags if flag == 1)
        valid_count = sum(1 for flag in token_flags if flag == 0 or flag == 1)

        print(f"{self.n} gram signature N: {valid_count}, {self.n} gram signature NG: {green_count}")
        
        if valid_count == 0:
            return 0.0, token_flags
        
        z_score = self.utils._compute_z_score(green_count, valid_count)
        
        return z_score, token_flags
    
    @property
    def ngram_signature_set_size(self) -> int:
        """
        返回 n-gram 簽名集大小。
        
        Returns:
            int: n-gram 簽名集中的序列數量
        """
        return len(self.ngram_signature_set)
    
class UnigramNGramSignature(UnigramSignature): 
    """
    Unigram水印的 n-gram 簽名感知版本，根據連續紅字規則進行檢測。
    """
    
    def __init__(
        self, 
        algorithm_config: str, 
        transformers_config: Optional[Any] = None, 
        n: int = 3,
        signature_set: Optional[Set[int]] = None, 
        signature_file: Optional[str] = None,
        ngram_signature_set: Optional[Set[Tuple[int, ...]]] = None, 
        ngram_signature_file: Optional[str] = None, 
        *args, 
        **kwargs
    ) -> None:
        """
        初始化 n-gram 簽名感知的Unigram水印。
        
        Args:
            algorithm_config: 算法配置文件路徑或配置對象
            transformers_config: Transformers配置
            n: 連續紅字的最小長度
            signature_set: 簽名集合
            signature_file: 簽名文件路徑
            ngram_signature_set: n-gram 簽名集合
            ngram_signature_file: n-gram 簽名文件路徑
        """
        super().__init__(algorithm_config, transformers_config, signature_set, signature_file, *args, **kwargs)
        
        self.n = n
        self.ngram_signature_set: Set[Tuple[int, ...]] = set()
        
        if ngram_signature_set:
            self.ngram_signature_set = set(ngram_signature_set)
        elif ngram_signature_file:
            self.load_ngram_signature_set(ngram_signature_file)

    def load_ngram_signature_set(self, file_path: str) -> None:
        """從文件加載 n-gram 簽名集"""
        self.n, self.ngram_signature_set = NGramSignatureSetUtils.load(file_path)
    
    def save_ngram_signature_set(self, save_path: str) -> None:
        """保存 n-gram 簽名集到文件"""
        NGramSignatureSetUtils.save(self.ngram_signature_set, self.n, save_path)
    
    def detect_watermark(self, text: str, return_dict: bool = True, *args, **kwargs) -> Union[Dict[str, Any], Tuple[bool, float]]:
        """使用 n-gram 規則進行水印檢測"""
        encoded_text = self.config.generation_tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0].to(self.config.device)
        
        z_score, green_tokens = self.ngram_score_sequence(encoded_text)
        
        is_watermarked = z_score > self.config.z_threshold
        
        if return_dict:
            return {
                "is_watermarked": is_watermarked, 
                "score": z_score,
                "ngram_applied": True,
                "n": self.n,
                "ngram_signature_size": len(self.ngram_signature_set)
            }
        else:
            return (is_watermarked, z_score)
    
    def ngram_score_sequence(self, input_ids: torch.LongTensor) -> Tuple[float, List[int]]:
        """
        應用 n-gram 連續紅字規則進行評分。
        
        Args:
            input_ids: 編碼後的文本張量
        
        Returns:
            Tuple[float, List[int]]: z-score 值和標記列表
            標記列表中：-1 表示在 signature 中，1 表示綠字，0 表示紅字
        """
        if len(input_ids) == 0:
            return 0.0, []
        
        # 1. 先標記所有 token 為綠字
        token_flags = [1] * len(input_ids)
        
        # 2. 檢查每個可能的 n-gram 序列是否匹配完整的 signature
        for i in range(len(input_ids) - self.n + 1):
            current_ngram = tuple(input_ids[i:i+self.n].tolist())
            if current_ngram in self.ngram_signature_set:
                # 找到完整的 signature，將整個序列標記為 -1
                for j in range(i, i + self.n):
                    token_flags[j] = -1
        
        # 3. 對於未被標記為 signature 的 token，根據 mask 判斷紅綠字
        for i in range(len(input_ids)):
            if token_flags[i] != -1:  # 如果不是 signature
                token_flags[i] = 1 if self.utils.mask[input_ids[i].item()] else 0
        
        # 4. 計算 z-score（只考慮不在 signature 中的 token）
        green_count = sum(1 for flag in token_flags if flag == 1)
        valid_count = sum(1 for flag in token_flags if flag != -1)

        print(f"{self.n} gram signature N: {valid_count}, {self.n} gram signature NG: {green_count}")
        
        if valid_count == 0:
            return 0.0, token_flags
        
        z_score = self.utils._compute_z_score(green_count, valid_count)
        
        return z_score, token_flags
    
    @property
    def ngram_signature_set_size(self) -> int:
        """
        返回 n-gram 簽名集大小。
        
        Returns:
            int: n-gram 簽名集中的序列數量
        """
        return len(self.ngram_signature_set)

class NGramWatermarkTokenAnalyzer(WatermarkTokenAnalyzer):
    """
    分析水印文本中 n-gram 連續紅字/綠字的分布。
    繼承自 WatermarkTokenAnalyzer。
    """
    
    def __init__(self, watermark: Union[KGW, SWEET, Unigram, 'KGWNGramSignature'], n: int = 3) -> None:
        """
        初始化 n-gram 分析器。
        
        Args:
            watermark: 水印系統實例，用於判斷綠字和紅字
            n: n-gram 的 n 值
        """
        super().__init__(watermark)
        self.n = n
        # n-gram 紅字和綠字序列的計數器
        self.ngram_stats: Dict[Tuple[int, ...], Dict[str, int]] = {}
        
    def analyze_text(self, text: str) -> None:
        """
        分析文本中每個 token 的綠字和紅字次數，同時分析 n-gram 序列。
        
        Args:
            text: 水印文本
        """
        encoded_text = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0].to(self.device)
        
        if isinstance(self.watermark, KGW) or hasattr(self.watermark, 'ngram_utils'):
            self._analyze_kgw_with_ngram(encoded_text)
        elif isinstance(self.watermark, SWEET):
            self._analyze_sweet_with_ngram(encoded_text)
        elif isinstance(self.watermark, Unigram):
            self._analyze_unigram_with_ngram(encoded_text)
        else:
            raise NotImplementedError(f"不支援的水印類型: {type(self.watermark).__name__}")
    
    def _analyze_kgw_with_ngram(self, encoded_text: torch.LongTensor) -> None:
        """
        分析KGW水印文本中綠字和紅字，同時分析n-gram
        
        Args:
            encoded_text: 編碼後的文本張量
        """
        # 標記每個位置是紅字還是綠字
        token_labels = []  # 1 為綠字，0 為紅字
        tokens = []
        
        # 使用父類方法分析單個 token
        self._analyze_kgw(encoded_text)
        
        # 另外收集 token 序列和標記
        for idx in range(self.prefix_length, len(encoded_text)):
            curr_token = encoded_text[idx].item()
            tokens.append(curr_token)
            
            # 獲取綠名單ID
            greenlist_ids = self.watermark.utils.get_greenlist_ids(encoded_text[:idx])
            
            # 根據是否在綠名單中判斷
            if curr_token in greenlist_ids:
                token_labels.append(1)  # 綠字
            else:
                token_labels.append(0)  # 紅字
        
        # 分析連續 n 個 token 的 n-gram
        for i in range(len(token_labels) - self.n + 1):
            ngram = tuple(tokens[i:i+self.n])
            # n-gram 的標記：如果全部都是綠字，則為綠字n-gram，否則為紅字n-gram
            ngram_label = 1 if all(label == 1 for label in token_labels[i:i+self.n]) else 0
            
            if ngram not in self.ngram_stats:
                self.ngram_stats[ngram] = {"green_count": 0, "red_count": 0}
            
            if ngram_label == 1:
                self.ngram_stats[ngram]["green_count"] += 1
            else:
                self.ngram_stats[ngram]["red_count"] += 1
    
    def _analyze_sweet_with_ngram(self, encoded_text: torch.LongTensor) -> None:
        """
        分析SWEET水印文本中綠字和紅字，同時分析n-gram
        
        Args:
            encoded_text: 編碼後的文本張量
        """
        # 標記每個位置是紅字還是綠字
        token_labels = []  # 1 為綠字，0 為紅字
        tokens = []
        
        # 使用父類方法分析單個 token
        self._analyze_sweet(encoded_text)
        
        # 另外收集 token 序列和標記
        entropy_list = self.watermark.utils.calculate_entropy(
            self.watermark.config.generation_model, 
            encoded_text
        )
        
        for idx in range(self.prefix_length, len(encoded_text)):
            curr_token = encoded_text[idx].item()
            tokens.append(curr_token)
            
            # 獲取綠名單
            greenlist_ids = self.watermark.utils.get_greenlist_ids(encoded_text[:idx])
            
            # 根據是否在綠名單中判斷
            if curr_token in greenlist_ids:
                token_labels.append(1)  # 綠字
            else:
                token_labels.append(0)  # 紅字
        
        # 分析連續 n 個 token 的 n-gram
        for i in range(len(token_labels) - self.n + 1):
            ngram = tuple(tokens[i:i+self.n])
            # n-gram 的標記：如果全部都是綠字，則為綠字n-gram，否則為紅字n-gram
            ngram_label = 1 if all(label == 1 for label in token_labels[i:i+self.n]) else 0
            
            if ngram not in self.ngram_stats:
                self.ngram_stats[ngram] = {"green_count": 0, "red_count": 0}
            
            if ngram_label == 1:
                self.ngram_stats[ngram]["green_count"] += 1
            else:
                self.ngram_stats[ngram]["red_count"] += 1
    
    def _analyze_unigram_with_ngram(self, encoded_text: torch.LongTensor) -> None:
        """
        分析Unigram水印文本中綠字和紅字，同時分析n-gram
        
        Args:
            encoded_text: 編碼後的文本張量
        """
        # 標記每個位置是紅字還是綠字
        token_labels = []  # 1 為綠字，0 為紅字
        tokens = []
        
        # 使用父類方法分析單個 token
        self._analyze_unigram(encoded_text)
        
        # 另外收集 token 序列和標記
        for idx in range(len(encoded_text)):
            curr_token = encoded_text[idx].item()
            tokens.append(curr_token)
            
            # 根據 mask 判斷
            if self.utils.mask[curr_token]:
                token_labels.append(1)  # 綠字
            else:
                token_labels.append(0)  # 紅字
        
        # 分析連續 n 個 token 的 n-gram
        for i in range(len(token_labels) - self.n + 1):
            ngram = tuple(tokens[i:i+self.n])
            # n-gram 的標記：如果全部都是綠字，則為綠字n-gram，否則為紅字n-gram
            ngram_label = 1 if all(label == 1 for label in token_labels[i:i+self.n]) else 0
            
            if ngram not in self.ngram_stats:
                self.ngram_stats[ngram] = {"green_count": 0, "red_count": 0}
            
            if ngram_label == 1:
                self.ngram_stats[ngram]["green_count"] += 1
            else:
                self.ngram_stats[ngram]["red_count"] += 1
    
    def get_ngram_stats(self) -> List[Dict[str, Any]]:
        """
        獲取 n-gram 統計資訊。
        
        Returns:
            List[Dict]: 包含每個 n-gram 的令牌序列、綠字次數和紅字次數的列表
        """
        result = []
        for ngram, counts in self.ngram_stats.items():
            total_count = counts["green_count"] + counts["red_count"]
            
            # 嘗試解碼 n-gram 顯示
            try:
                decoded_ngram = "".join([self.tokenizer.decode(token) for token in ngram])
            except:
                decoded_ngram = "<無法解碼>"
                
            result.append({
                "ngram": list(ngram),  # 轉換為列表以便 JSON 序列化
                "decoded": decoded_ngram,
                "green_count": counts["green_count"],
                "red_count": counts["red_count"],
                "total_count": total_count,
                "green_ratio": counts["green_count"] / total_count if total_count > 0 else 0
            })
        
        # 按總出現次數排序
        result.sort(key=lambda x: x["total_count"], reverse=True)
        return result
    
    def save_ngram_stats(self, save_path: str) -> None:
        """
        保存 n-gram 統計到 JSON 文件。
        
        Args:
            save_path: 保存路徑
        """
        stats = self.get_ngram_stats()
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump({
                "n": self.n,
                "ngram_stats": stats
            }, f, ensure_ascii=False, indent=2)
        
        print(f"已保存 {len(stats)} 個 {self.n}-gram 的統計資訊到 {save_path}")
    
    def clear_stats(self) -> None:
        """
        清除所有統計資訊。
        """
        super().clear_stats()  # 清除父類的 token_stats
        self.ngram_stats.clear()  # 清除 n-gram 統計