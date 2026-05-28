# =============================================
# signature.py
# Description: Signature collection and signature-aware watermarking
# =============================================

import os
import json
from typing import Optional, Set, List, Dict, Tuple, Any, Union
import torch
from watermark.ewd.ewd import EWD
from watermark.kgw.kgw import KGW
from watermark.sweet.sweet import SWEET
from watermark.unigram.unigram import Unigram

class SignatureSetUtils:
    @staticmethod
    def load(file_path: str) -> Set[int]:
        """從文件加載簽名集"""
        try:
            with open(file_path, 'r') as f:
                signature_set = set(json.load(f))
            print(f"已從 {file_path} 加載 {len(signature_set)} 個簽名")
            return signature_set
        except FileNotFoundError:
            raise FileNotFoundError(f"文件不存在: {file_path}")
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(f"JSON 格式錯誤: {e.msg}", e.doc, e.pos)
    
    @staticmethod
    def save(signature_set: Set[int], save_path: str) -> None:
        """保存簽名集到文件"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(list(signature_set), f)
        print(f"已保存 {len(signature_set)} 個簽名到 {save_path}")

class SignatureSetCollector:
    """
    用於收集和管理簽名集合的工具類。
    
    收集生成式水印中的「紅字」tokens，用於後續檢測時提高準確性。
    """
    
    def __init__(self, watermark: Union[KGW, SWEET, Unigram]) -> None:
        """
        初始化簽名收集器。
        
        Args:
            watermark: 水印系統實例，用於獲取綠名單和其他信息
        """
        self.watermark = watermark
        self.signature_set: Set[int] = set()
        self.tokenizer = watermark.config.generation_tokenizer
        self.prefix_length = getattr(watermark.config, 'prefix_length', 0)
        self.device = watermark.config.device
        
    def collect_from_text(self, text: str) -> None:
        """
        從單一文本收集紅字。
        
        Args:
            text: 要分析的文本
        
        Raises:
            NotImplementedError: 如果水印類型不支援
        """
        encoded_text = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0].to(self.device)
        
        if isinstance(self.watermark, KGW):
            self._collect_from_kgw(encoded_text)
        elif isinstance(self.watermark, SWEET):
            self._collect_from_sweet(encoded_text)
        elif isinstance(self.watermark, Unigram):
            self._collect_from_unigram(encoded_text)
        else:
            raise NotImplementedError(f"不支援的水印類型: {type(self.watermark).__name__}")
    
    def _collect_from_kgw(self, encoded_text: torch.LongTensor) -> None:
        """
        從 KGW 水印文本中收集紅字。
        
        Args:
            encoded_text: 編碼後的文本張量
        """
        for idx in range(self.prefix_length, len(encoded_text)):
            curr_token = encoded_text[idx].item()
            # 獲取綠名單ID
            greenlist_ids = self.watermark.utils.get_greenlist_ids(encoded_text[:idx])
            # 如果不在綠名單中，就是紅字，加入signature_set
            if curr_token not in greenlist_ids:
                self.signature_set.add(curr_token)
    
    def _collect_from_sweet(self, encoded_text: torch.LongTensor) -> None:
        """
        從 SWEET 水印文本中收集高熵紅字。
        
        Args:
            encoded_text: 編碼後的文本張量
        """
        # 計算熵值
        entropy_list = self.watermark.utils.calculate_entropy(
            self.watermark.config.generation_model, 
            encoded_text
        )
        
        # 收集高熵紅字
        for idx in range(self.prefix_length, len(encoded_text)):
            curr_token = encoded_text[idx].item()
            
            # 獲取綠名單
            greenlist_ids = self.watermark.utils.get_greenlist_ids(encoded_text[:idx])
            
            # 檢查熵值是否高於閾值
            is_high_entropy = entropy_list[idx] > self.watermark.config.entropy_threshold
            
            # 如果不在綠名單中且熵值高，就是我們要收集的簽名
            if curr_token not in greenlist_ids and is_high_entropy:
                self.signature_set.add(curr_token)
    
    def _collect_from_unigram(self, encoded_text: torch.LongTensor) -> None:
        """
        從 Unigram 水印文本中收集紅字。
        
        Args:
            encoded_text: 編碼後的文本張量
        """
        for idx in range(len(encoded_text)):
            curr_token = encoded_text[idx].item()
            # 檢查是否在綠名單中（即 mask 值為 True）
            if not self.watermark.utils.mask[curr_token]:
                # 不在綠名單中，即為紅字，加入 signature_set
                self.signature_set.add(curr_token)
    
    def collect_from_file(self, file_path: str) -> None:
        """
        從文件讀取文本並收集簽名。
        
        Args:
            file_path: 文本文件路徑
        
        Raises:
            FileNotFoundError: 如果文件不存在
            IOError: 如果讀取文件時出錯
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            self.collect_from_text(text)
        except FileNotFoundError:
            raise FileNotFoundError(f"文件不存在: {file_path}")
        except IOError as e:
            raise IOError(f"讀取文件時出錯: {e}")
    
    def save_signature_set(self, save_path: str) -> None:
        """保存簽名集到文件"""
        SignatureSetUtils.save(self.signature_set, save_path)
    
    def load_signature_set(self, file_path: str) -> None:
        """從文件加載簽名集"""
        self.signature_set = SignatureSetUtils.load(file_path)

class KGWSignature(KGW):
    """KGW水印的簽名感知版本，可在檢測時排除簽名集中的tokens。"""
    
    def __init__(
        self, 
        algorithm_config: str, 
        transformers_config: Optional[Any] = None, 
        signature_set: Optional[Set[int]] = None, 
        signature_file: Optional[str] = None, 
        *args, 
        **kwargs
    ) -> None:
        """
        初始化簽名感知的KGW水印。
        
        Args:
            algorithm_config: 算法配置文件路徑或配置對象
            transformers_config: Transformers配置
            signature_set: 簽名集合
            signature_file: 簽名文件路徑
        """
        super().__init__(algorithm_config, transformers_config, *args, **kwargs)
        
        self.signature_set: Set[int] = set()
        if signature_set:
            self.signature_set = set(signature_set)
        elif signature_file:
            self.load_signature_set(signature_file)
    
    def load_signature_set(self, file_path: str) -> None:
        """從文件加載簽名集"""
        self.signature_set = SignatureSetUtils.load(file_path)
    
    def save_signature_set(self, save_path: str) -> None:
        """保存簽名集到文件"""
        SignatureSetUtils.save(self.signature_set, save_path)
    
    def detect_watermark(self, text: str, return_dict: bool = True, *args, **kwargs) -> Union[Dict[str, Any], Tuple[bool, float]]:
        """
        重寫偵測方法，考慮簽名集。
        
        Args:
            text: 要檢測的文本
            return_dict: 是否返回字典格式的結果
        
        Returns:
            Union[Dict[str, Any], Tuple[bool, float]]: 檢測結果
        """
        encoded_text = self.config.generation_tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0].to(self.config.device)
        
        # 過濾掉簽名集中的token
        z_score, green_tokens = self.signature_score_sequence(encoded_text)
        
        is_watermarked = z_score > self.config.z_threshold
        
        if return_dict:
            return {
                "is_watermarked": is_watermarked, 
                "score": z_score,
                "signature_filtered": len(self.signature_set) > 0
            }
        else:
            return (is_watermarked, z_score)
    
    def signature_score_sequence(self, input_ids: torch.LongTensor) -> Tuple[float, List[int]]:
        """
        考慮 signature 的評分方法，排除簽名集中的 tokens。
        
        Args:
            input_ids: 編碼後的文本張量
        
        Returns:
            Tuple[float, List[int]]: z-score 值和綠色標記列表
        """
        valid_positions = []
        green_token_count = 0
        green_token_flags = [-1 for _ in range(self.config.prefix_length)]
        
        filtered_count = 0  # 記錄被過濾的token數量
        
        for idx in range(self.config.prefix_length, len(input_ids)):
            curr_token = input_ids[idx].item()
            
            # 如果token在簽名集中，跳過
            if curr_token in self.signature_set:
                green_token_flags.append(-1)  # 標記為不計算
                filtered_count += 1
                continue
            
            valid_positions.append(idx)
            greenlist_ids = self.utils.get_greenlist_ids(input_ids[:idx])
            if curr_token in greenlist_ids:
                green_token_count += 1
                green_token_flags.append(1)
            else:
                green_token_flags.append(0)
        
        # 計算實際評分的token數量
        num_tokens_scored = len(valid_positions)
        print(f"signature N: {num_tokens_scored}, signature NG: {green_token_count}")
        if num_tokens_scored < 1:
            return 0.0, green_token_flags  # 太少token無法評分
        
        # 使用 utils 的 _compute_z_score 函數計算 z-score
        z_score = self.utils._compute_z_score(green_token_count, num_tokens_scored)
        
        return z_score, green_token_flags
    
    @property
    def signature_set_size(self) -> int:
        """
        返回簽名集大小。
        
        Returns:
            int: 簽名集中token的數量
        """
        return len(self.signature_set)


class SweetSignature(SWEET):
    """SWEET水印的簽名感知版本，可在檢測時排除簽名集中的tokens。"""
    
    def __init__(
        self, 
        algorithm_config: str, 
        transformers_config: Optional[Any] = None, 
        signature_set: Optional[Set[int]] = None, 
        signature_file: Optional[str] = None, 
        *args, 
        **kwargs
    ) -> None:
        """
        初始化簽名感知的SWEET水印。
        
        Args:
            algorithm_config: 算法配置文件路徑或配置對象
            transformers_config: Transformers配置
            signature_set: 簽名集合
            signature_file: 簽名文件路徑
        """
        super().__init__(algorithm_config, transformers_config, *args, **kwargs)
        
        self.signature_set: Set[int] = set()
        if signature_set:
            self.signature_set = set(signature_set)
        elif signature_file:
            self.load_signature_set(signature_file)
    
    def load_signature_set(self, file_path: str) -> None:
        """從文件加載簽名集"""
        self.signature_set = SignatureSetUtils.load(file_path)
    
    def save_signature_set(self, save_path: str) -> None:
        """保存簽名集到文件"""
        SignatureSetUtils.save(self.signature_set, save_path)
    
    def detect_watermark(self, text: str, return_dict: bool = True, *args, **kwargs) -> Union[Dict[str, Any], Tuple[bool, float]]:
        """
        重寫偵測方法，考慮簽名集。
        
        Args:
            text: 要檢測的文本
            return_dict: 是否返回字典格式的結果
        
        Returns:
            Union[Dict[str, Any], Tuple[bool, float]]: 檢測結果
        """
        encoded_text = self.config.generation_tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0].to(self.config.device)
        
        # 計算熵值
        entropy_list = self.utils.calculate_entropy(self.config.generation_model, encoded_text)
        
        # 過濾掉簽名集中的token
        z_score, green_tokens, weights = self.signature_score_sequence(encoded_text, entropy_list)
        
        is_watermarked = z_score > self.config.z_threshold
        
        if return_dict:
            return {
                "is_watermarked": is_watermarked, 
                "score": z_score,
                "signature_filtered": len(self.signature_set) > 0
            }
        else:
            return (is_watermarked, z_score)
    
    def signature_score_sequence(self, input_ids: torch.LongTensor, entropy_list: List[float]) -> Tuple[float, List[int], List[int]]:
        """
        考慮 signature 的評分方法，排除簽名集中的 tokens。
        
        Args:
            input_ids: 編碼後的文本張量
            entropy_list: 文本中每個token的熵值列表
        
        Returns:
            Tuple[float, List[int], List[int]]: z-score 值、綠色標記列表和權重列表
        """
        # 初始化標記列表
        green_token_flags = [-1 for _ in range(self.config.prefix_length)]
        weights = [-1 for _ in range(self.config.prefix_length)]
        
        # 處理每個 token
        valid_positions = []
        green_token_count = 0
        
        for idx in range(self.config.prefix_length, len(input_ids)):
            curr_token = input_ids[idx].item()
            
            # 獲取綠名單
            greenlist_ids = self.utils.get_greenlist_ids(input_ids[:idx])
            
            # 首先，根據熵值決定權重
            # 這與原始邏輯一致：熵值高的設置為1，否則為0
            if entropy_list[idx] > self.config.entropy_threshold:
                weights.append(1)
            else:
                weights.append(0)
            
            # 如果 token 在簽名集中，標記為 -1 並跳過評分
            if curr_token in self.signature_set:
                green_token_flags.append(-1)
                weights[-1] = -1  # 在簽名集中的 token 權重設為 -1
                continue
            
            # 處理非簽名集中的 token
            if entropy_list[idx] > self.config.entropy_threshold:
                valid_positions.append(idx)
                
                # 檢查是否在綠名單中
                if curr_token in greenlist_ids:
                    green_token_flags.append(1)
                    green_token_count += 1
                else:
                    green_token_flags.append(0)
            else:
                # 熵值低的 token 不計入綠色標記統計
                green_token_flags.append(-1)
        
        # 計算 z-score
        num_tokens_scored = len(valid_positions)
        print(f"signature N: {num_tokens_scored}, signature NG: {green_token_count}")
        if num_tokens_scored < 1:
            return 0.0, green_token_flags, weights
        
        z_score = self.utils._compute_z_score(green_token_count, num_tokens_scored)
        return z_score, green_token_flags, weights
    
    @property
    def signature_set_size(self) -> int:
        """
        返回簽名集大小。
        
        Returns:
            int: 簽名集中token的數量
        """
        return len(self.signature_set)


class UnigramSignature(Unigram):
    """Unigram水印的簽名感知版本，可在檢測時排除簽名集中的tokens。"""
    
    def __init__(
        self, 
        algorithm_config: str, 
        transformers_config: Optional[Any] = None, 
        signature_set: Optional[Set[int]] = None, 
        signature_file: Optional[str] = None, 
        *args, 
        **kwargs
    ) -> None:
        """
        初始化簽名感知的Unigram水印。
        
        Args:
            algorithm_config: 算法配置文件路徑或配置對象
            transformers_config: Transformers配置
            signature_set: 簽名集合
            signature_file: 簽名文件路徑
        """
        super().__init__(algorithm_config, transformers_config, *args, **kwargs)
        
        self.signature_set: Set[int] = set()
        if signature_set:
            self.signature_set = set(signature_set)
        elif signature_file:
            self.load_signature_set(signature_file)
    
    def load_signature_set(self, file_path: str) -> None:
        """從文件加載簽名集"""
        self.signature_set = SignatureSetUtils.load(file_path)
    
    def save_signature_set(self, save_path: str) -> None:
        """保存簽名集到文件"""
        SignatureSetUtils.save(self.signature_set, save_path)
    
    def detect_watermark(self, text: str, return_dict: bool = True, *args, **kwargs) -> Union[Dict[str, Any], Tuple[bool, float]]:
        """
        重寫偵測方法，考慮簽名集。
        
        Args:
            text: 要檢測的文本
            return_dict: 是否返回字典格式的結果
        
        Returns:
            Union[Dict[str, Any], Tuple[bool, float]]: 檢測結果
        """
        encoded_text = self.config.generation_tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0].to(self.config.device)
        
        # 過濾掉簽名集中的token
        z_score, green_tokens = self.signature_score_sequence(encoded_text)
        
        is_watermarked = z_score > self.config.z_threshold
        
        if return_dict:
            return {
                "is_watermarked": is_watermarked, 
                "score": z_score,
                "signature_filtered": len(self.signature_set) > 0
            }
        else:
            return (is_watermarked, z_score)
    
    def signature_score_sequence(self, input_ids: torch.LongTensor) -> Tuple[float, List[int]]:
        """
        考慮 signature 的評分方法，排除簽名集中的 tokens。
        
        Args:
            input_ids: 編碼後的文本張量
        
        Returns:
            Tuple[float, List[int]]: z-score 值和綠色標記列表
        """
        valid_positions = []
        green_token_count = 0
        green_token_flags = []
        
        filtered_count = 0  # 記錄被過濾的token數量
        
        for idx in range(len(input_ids)):
            curr_token = input_ids[idx].item()
            
            # 如果token在簽名集中，跳過
            if curr_token in self.signature_set:
                green_token_flags.append(-1)  # 標記為不計算
                filtered_count += 1
                continue
            
            valid_positions.append(idx)
            if self.utils.mask[curr_token] == True:
                green_token_count += 1
                green_token_flags.append(1)
            else:
                green_token_flags.append(0)
        
        # 計算實際評分的token數量
        num_tokens_scored = len(valid_positions)
        print(f"signature N: {num_tokens_scored}, signature NG: {green_token_count}")
        if num_tokens_scored < 1:
            return 0.0, green_token_flags  # 太少token無法評分
        
        # 使用 utils 的 _compute_z_score 函數計算 z-score
        z_score = self.utils._compute_z_score(green_token_count, num_tokens_scored)
        
        return z_score, green_token_flags
    
    @property
    def signature_set_size(self) -> int:
        """
        返回簽名集大小。
        
        Returns:
            int: 簽名集中token的數量
        """
        return len(self.signature_set)
    
class EWDSignature(EWD):
    """EWD水印的簽名感知版本，可在檢測時排除簽名集中的tokens。"""
    
    def __init__(
        self, 
        algorithm_config: str, 
        transformers_config: Optional[Any] = None, 
        signature_set: Optional[Set[int]] = None, 
        signature_file: Optional[str] = None, 
        *args, 
        **kwargs
    ) -> None:
        """
        初始化簽名感知的KGW水印。
        
        Args:
            algorithm_config: 算法配置文件路徑或配置對象
            transformers_config: Transformers配置
            signature_set: 簽名集合
            signature_file: 簽名文件路徑
        """
        super().__init__(algorithm_config, transformers_config, *args, **kwargs)
        
        self.signature_set: Set[int] = set()
        if signature_set:
            self.signature_set = set(signature_set)
        elif signature_file:
            self.load_signature_set(signature_file)
    
    def load_signature_set(self, file_path: str) -> None:
        """從文件加載簽名集"""
        self.signature_set = SignatureSetUtils.load(file_path)
    
    def save_signature_set(self, save_path: str) -> None:
        """保存簽名集到文件"""
        SignatureSetUtils.save(self.signature_set, save_path)
    
    def detect_watermark(self, text: str, return_dict: bool = True, *args, **kwargs) -> Union[Dict[str, Any], Tuple[bool, float]]:
        """
        重寫偵測方法，考慮簽名集。
        
        Args:
            text: 要檢測的文本
            return_dict: 是否返回字典格式的結果
        
        Returns:
            Union[Dict[str, Any], Tuple[bool, float]]: 檢測結果
        """
        encoded_text = self.config.generation_tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0].to(self.config.device)
        
        # 過濾掉簽名集中的token
        z_score, green_tokens = self.signature_score_sequence(encoded_text)
        
        is_watermarked = z_score > self.config.z_threshold
        
        if return_dict:
            return {
                "is_watermarked": is_watermarked, 
                "score": z_score,
                "signature_filtered": len(self.signature_set) > 0
            }
        else:
            return (is_watermarked, z_score)
    
    def signature_score_sequence(self, input_ids: torch.LongTensor, entropy_list: List[float]) -> Tuple[float, List[int], List[float]]:
        """
        考慮 signature 的評分方法，排除簽名集中的 tokens。
        
        Args:
            input_ids: 編碼後的文本張量
            entropy_list: 文本中每個token的熵值列表
        
        Returns:
            Tuple[float, List[int], List[float]]: z-score 值、綠色標記列表和權重列表
        """
        # 檢查是否有足夠的 tokens 進行評分
        num_tokens_scored = len(input_ids) - self.config.prefix_length
        if num_tokens_scored < 1:
            return 0.0, [], []  # 太少 token 無法評分
        
        # 初始化綠色標記列表
        green_token_flags = [-1 for _ in range(self.config.prefix_length)]
        
        # 初始化權重列表
        weights = [-1 for _ in range(self.config.prefix_length)]
        
        # 處理每個 token
        for idx in range(self.config.prefix_length, len(input_ids)):
            curr_token = input_ids[idx].item()
            
            # 如果 token 在簽名集中，跳過
            if curr_token in self.signature_set:
                green_token_flags.append(-1)  # 標記為不計算
                weights.append(-1)  # 權重也標記為不計算
                continue
            
            # 獲取綠名單並判斷當前 token
            greenlist_ids = self.utils.get_greenlist_ids(input_ids[:idx])
            if curr_token in greenlist_ids:
                green_token_flags.append(1)
            else:
                green_token_flags.append(0)
            
            # 計算權重
            if idx >= self.config.prefix_length:
                weights.append(entropy_list[idx])
        
        # 過濾掉被標記為 -1 的位置
        valid_weights = [w for w, f in zip(weights[self.config.prefix_length:], 
                                         green_token_flags[self.config.prefix_length:]) 
                        if f != -1]
        valid_flags = [f for f in green_token_flags[self.config.prefix_length:] 
                      if f != -1]
        
        if not valid_weights:  # 如果沒有有效的權重
            return 0.0, green_token_flags, weights
        
        # 計算綠色 token 的加權計數
        green_token_count = sum(w for w, f in zip(valid_weights, valid_flags) if f == 1)
        print(f"signature N: {len(valid_weights)}, signature NG: {green_token_count}")
        # 使用 utils 的 _compute_z_score 函數計算 z-score
        z_score = self.utils._compute_z_score(green_token_count, valid_weights)
        
        return z_score, green_token_flags, weights
    
    @property
    def signature_set_size(self) -> int:
        """
        返回簽名集大小。
        
        Returns:
            int: 簽名集中token的數量
        """
        return len(self.signature_set)

class WatermarkTokenAnalyzer:
    """
    分析水印文本中token的紅字和綠字次數。
    """
    
    def __init__(self, watermark: Union[KGW, SWEET, Unigram]) -> None:
        """
        初始化分析器。
        
        Args:
            watermark: 水印系統實例，用於判斷綠字和紅字
        """
        self.watermark = watermark
        self.tokenizer = watermark.config.generation_tokenizer
        self.prefix_length = getattr(watermark.config, 'prefix_length', 0)
        self.device = watermark.config.device
        
        # 紅字和綠字的計數器
        self.token_stats: Dict[int, Dict[str, int]] = {}
        
    def analyze_text(self, text: str) -> None:
        """
        分析文本中每個token的綠字和紅字次數。
        
        Args:
            text: 水印文本
        """
        encoded_text = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0].to(self.device)
        
        if isinstance(self.watermark, KGW):
            self._analyze_kgw(encoded_text)
        elif isinstance(self.watermark, SWEET):
            self._analyze_sweet(encoded_text)
        elif isinstance(self.watermark, Unigram):
            self._analyze_unigram(encoded_text)
        else:
            raise NotImplementedError(f"不支援的水印類型: {type(self.watermark).__name__}")
    
    def _analyze_kgw(self, encoded_text: torch.LongTensor) -> None:
        """分析KGW水印文本中綠字和紅字"""
        for idx in range(self.prefix_length, len(encoded_text)):
            curr_token = encoded_text[idx].item()
            
            # 獲取綠名單ID
            greenlist_ids = self.watermark.utils.get_greenlist_ids(encoded_text[:idx])
            
            # 初始化或更新統計
            if curr_token not in self.token_stats:
                self.token_stats[curr_token] = {"green_count": 0, "red_count": 0}
            
            # 根據是否在綠名單中更新計數
            if curr_token in greenlist_ids:
                self.token_stats[curr_token]["green_count"] += 1
            else:
                self.token_stats[curr_token]["red_count"] += 1
    
    def _analyze_sweet(self, encoded_text: torch.LongTensor) -> None:
        """分析SWEET水印文本中綠字和紅字"""
        entropy_list = self.watermark.utils.calculate_entropy(
            self.watermark.config.generation_model, 
            encoded_text
        )
        
        for idx in range(self.prefix_length, len(encoded_text)):
            curr_token = encoded_text[idx].item()
            
            # 獲取綠名單
            greenlist_ids = self.watermark.utils.get_greenlist_ids(encoded_text[:idx])
            
            # 初始化或更新統計
            if curr_token not in self.token_stats:
                self.token_stats[curr_token] = {"green_count": 0, "red_count": 0}
            
            # 根據是否在綠名單中更新計數
            if curr_token in greenlist_ids:
                self.token_stats[curr_token]["green_count"] += 1
            else:
                self.token_stats[curr_token]["red_count"] += 1
    
    def _analyze_unigram(self, encoded_text: torch.LongTensor) -> None:
        """分析Unigram水印文本中綠字和紅字"""
        for idx in range(len(encoded_text)):
            curr_token = encoded_text[idx].item()
            
            # 初始化或更新統計
            if curr_token not in self.token_stats:
                self.token_stats[curr_token] = {"green_count": 0, "red_count": 0}
            
            # 根據是否在綠名單中更新計數（Unigram直接使用mask）
            if self.watermark.utils.mask[curr_token]:
                self.token_stats[curr_token]["green_count"] += 1
            else:
                self.token_stats[curr_token]["red_count"] += 1
    
    def analyze_file(self, file_path: str) -> None:
        """
        從文件讀取文本並分析。
        
        Args:
            file_path: 文本文件路徑
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            self.analyze_text(text)
        except Exception as e:
            print(f"分析文件時出錯: {e}")
    
    def analyze_watermarked_texts_json(self, file_path: str, text_key: str = 'watermarked_text') -> None:
        """
        從包含多個水印文本的JSON文件中分析token統計。
        
        Args:
            file_path: JSON文件路徑，每項應該包含水印文本
            text_key: 水印文本的鍵名
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if not isinstance(data, list):
                print(f"錯誤: {file_path} 不是一個有效的文本列表")
                return
                
            for entry in data:
                if isinstance(entry, dict) and text_key in entry:
                    text = entry[text_key]
                    self.analyze_text(text)
                else:
                    print(f"警告: 找不到文本鍵 '{text_key}'")
                    
            print(f"已分析 {len(data)} 個文本")
        
        except Exception as e:
            print(f"分析JSON文件時出錯: {e}")
    
    def get_token_stats(self) -> List[Dict[str, Any]]:
        """
        獲取token統計資訊。
        
        Returns:
            List[Dict]: 包含每個token的ID、綠字次數和紅字次數的列表
        """
        result = []
        for token_id, counts in self.token_stats.items():
            total_count = counts["green_count"] + counts["red_count"]
            result.append({
                "token_id": token_id,
                "green_count": counts["green_count"],
                "red_count": counts["red_count"],
                "total_count": total_count,
                "green_ratio": counts["green_count"] / total_count if total_count > 0 else 0
            })
        
        # 按總出現次數排序
        result.sort(key=lambda x: x["total_count"], reverse=True)
        return result
    
    def save_stats(self, save_path: str) -> None:
        """
        保存token統計到JSON文件。
        
        Args:
            save_path: 保存路徑
        """
        stats = self.get_token_stats()
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        print(f"已保存 {len(stats)} 個token的統計資訊到 {save_path}")
    
    def clear_stats(self) -> None:
        """
        清除所有統計資訊。
        """
        self.token_stats.clear()