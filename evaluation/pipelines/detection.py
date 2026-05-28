# Copyright 2024 THU-BPM MarkLLM.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# =============================================
# detection.py
# Description: Pipeline for watermark detection
# =============================================

import json
import os
from tqdm import tqdm
from enum import Enum, auto
from utils.timer import timer
from watermark.base import BaseWatermark
from evaluation.dataset import BaseDataset
from evaluation.tools.text_editor import TextEditor
from exceptions.exceptions import InvalidTextSourceModeError
import logging
from typing import Optional, List, Dict, Any, Union, Tuple
from watermark.ewd.ewd import EWD
from watermark.kgw.kgw import KGW
from watermark.sweet.sweet import SWEET
from watermark.unigram.unigram import Unigram
from watermark.exp.exp import EXP
from evaluation.tools.success_rate_calculator import FundamentalSuccessRateCalculator
import torch
from watermark.signature.signature import SignatureSetCollector, KGWSignature, SweetSignature, UnigramSignature
from watermark.signature.ngram import KGWNGramSignature, SweetNGramSignature, UnigramNGramSignature
from math import log

class DetectionPipelineReturnType(Enum):
    """Return type of the watermark detection pipeline."""
    FULL = auto()
    SCORES = auto()
    IS_WATERMARKED = auto()


class WatermarkDetectionResult:
    """Result of watermark detection."""

    def __init__(self, generated_or_retrieved_text, edited_text, detect_result) -> None:
        """
            Initialize the watermark detection result.

            Parameters:
                generated_or_retrieved_text: The generated or retrieved text.
                edited_text: The edited text.
                detect_result: The detection result.
        """
        self.generated_or_retrieved_text = generated_or_retrieved_text
        self.edited_text = edited_text
        self.detect_result = detect_result
        pass


class WatermarkDetectionPipeline:
    """Pipeline for watermark detection."""

    def __init__(self, dataset: BaseDataset, text_editor_list: list[TextEditor] = [], 
                 show_progress: bool = True, return_type: DetectionPipelineReturnType = DetectionPipelineReturnType.SCORES) -> None:
        """
            Initialize the watermark detection pipeline.

            Parameters:
                dataset (BaseDataset): The dataset for the pipeline.
                text_editor_list (list[TextEditor]): The list of text editors.
                show_progress (bool): Whether to show progress bar.
                return_type (DetectionPipelineReturnType): The return type of the pipeline.
        """
        self.dataset = dataset
        self.text_editor_list = text_editor_list
        self.show_progress = show_progress
        self.return_type = return_type
       
    def _edit_text(self, text: str, prompt: str = None):
        """Edit text using text editors."""
        for text_editor in self.text_editor_list:
            text = text_editor.edit(text, prompt)
        return text
    
    def _generate_or_retrieve_text(self, dataset_index: int, watermark: BaseWatermark):
        """Generate or retrieve text from dataset."""
        pass

    def _detect_watermark(self, text: str, watermark: BaseWatermark):
        """Detect watermark in text."""
        detect_result = watermark.detect_watermark(text, return_dict=True)
        return detect_result

    def _get_iterable(self):
        """Return an iterable for the dataset."""
        pass

    def _get_progress_bar(self, iterable):
        """Return an iterable possibly wrapped with a progress bar."""
        if self.show_progress:
            return tqdm(iterable, desc="Processing", leave=True)
        return iterable

    def evaluate(self, watermark: BaseWatermark):
        """Conduct evaluation utilizing the pipeline."""
        evaluation_result = []
        bar = self._get_progress_bar(self._get_iterable())

        for index in bar:
            generated_or_retrieved_text = self._generate_or_retrieve_text(index, watermark)
            edited_text = self._edit_text(generated_or_retrieved_text, self.dataset.get_prompt(index))
            detect_result = self._detect_watermark(edited_text, watermark)
            evaluation_result.append(WatermarkDetectionResult(generated_or_retrieved_text, edited_text, detect_result))
            # print(f"generated_or_retrieved_text: {generated_or_retrieved_text}")
            # print(f"edited_text: {edited_text}")
            print(f"detect_result: {detect_result}")
            print("--------------------------------")
        if self.return_type == DetectionPipelineReturnType.FULL:
            return evaluation_result
        elif self.return_type == DetectionPipelineReturnType.SCORES:
            return [result.detect_result['score'] for result in evaluation_result]
        elif self.return_type == DetectionPipelineReturnType.IS_WATERMARKED:
            return [result.detect_result['is_watermarked'] for result in evaluation_result]
        
class WatermarkedTextDetectionPipeline(WatermarkDetectionPipeline):
    """Pipeline for detecting watermarked text."""

    def __init__(self, dataset, text_editor_list=[],
                 show_progress=True, return_type=DetectionPipelineReturnType.SCORES, *args, **kwargs) -> None:
        super().__init__(dataset, text_editor_list, show_progress, return_type)

    def _get_iterable(self):
        """Return an iterable for the prompts."""
        return range(self.dataset.prompt_nums)
    
    def _generate_or_retrieve_text(self, dataset_index, watermark):
        """Generate watermarked text from the dataset."""
        prompt = self.dataset.get_prompt(dataset_index)
        return watermark.generate_watermarked_text(prompt)


class UnWatermarkedTextDetectionPipeline(WatermarkDetectionPipeline):
    """Pipeline for detecting unwatermarked text."""

    def __init__(self, dataset, text_editor_list=[], text_source_mode='natural',
                 show_progress=True, return_type=DetectionPipelineReturnType.SCORES, *args, **kwargs) -> None:
        # Validate text_source_mode
        if text_source_mode not in ['natural', 'generated']:
            raise InvalidTextSourceModeError(text_source_mode)
        
        super().__init__(dataset, text_editor_list, show_progress, return_type)
        self.text_source_mode = text_source_mode

    def _get_iterable(self):
        """Return an iterable for the natural texts or prompts."""
        if self.text_source_mode == 'natural':
            return range(self.dataset.natural_text_nums)
        else:
            return range(self.dataset.prompt_nums)
    
    def _generate_or_retrieve_text(self, dataset_index, watermark):
        """Retrieve unwatermarked text from the dataset."""
        if self.text_source_mode == 'natural':
            return self.dataset.get_natural_text(dataset_index)
        else:
            prompt = self.dataset.get_prompt(dataset_index)
            return watermark.generate_unwatermarked_text(prompt)

class WMTextDetectionPipeline(WatermarkedTextDetectionPipeline):
    """Pipeline for detecting watermarked text. (Should provide watermarked texts.)"""

    def __init__(self, dataset, watermarked_texts_path, output_dir, text_editor_list=[],
                 show_progress=True, return_type=DetectionPipelineReturnType.SCORES, *args, **kwargs) -> None:
        super().__init__(dataset, text_editor_list, show_progress, return_type)
        self.watermarked_texts_path = watermarked_texts_path
        self.output_dir = output_dir

    def _get_iterable(self):
        """Return an iterable for the prompts."""
        return range(self.dataset.prompt_nums)
    
    def _load_watermarked_data(self, watermarked_texts_path: str):
        """載入水印文本"""
        with open(watermarked_texts_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    
    def _get_generated_text(self, dataset_index, watermark):
        """從載入的數據中獲取對應的水印文本"""
        prompt = self.dataset.get_prompt(dataset_index)
        data = self._load_watermarked_data(self.watermarked_texts_path)
        
        # 查找匹配當前 prompt 的水印文本
        for item in data:
            if item["prompt"] == prompt:
                return item["watermarked_text"]
        
        # 如果找不到匹配的 prompt，返回空字符串或其他默認值
        return ""

    def _save_edited_text(self, prompts: list[str], edited_texts: list[str], output_dir: str):
        """保存編輯後的文本"""
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, "edited_watermarked_texts.json")
        
        # 構建與示例格式一致的數據結構
        output_data = []
        for i, (prompt, text) in enumerate(zip(prompts, edited_texts)):
            output_data.append({
                "prompt": prompt,
                "watermarked_text": text
            })
        
        # 保存為JSON格式
        try:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            logging.info(f"已保存 {len(edited_texts)} 個水印文本到 {output_file}")
            return output_file
        except Exception as e:
            logging.error(f"保存水印文本時出錯: {str(e)}")
            print(f"保存水印文本失敗: {str(e)}")
            return ""

    def evaluate(self, watermark: BaseWatermark):
        """Conduct evaluation utilizing the pipeline."""
        evaluation_result = []
        bar = self._get_progress_bar(self._get_iterable())

        edit_texts = []
        for index in bar:
            generated_or_retrieved_text = self._get_generated_text(index, watermark)
            edited_text = self._edit_text(generated_or_retrieved_text, self.dataset.get_prompt(index))
            edit_texts.append(edited_text)
            print(f"generated_or_retrieved_text: {generated_or_retrieved_text}")
            print("--------------------------------")
            print(f"edited_text: {edited_text}")
            detect_result = self._detect_watermark(edited_text, watermark)
            evaluation_result.append(WatermarkDetectionResult(generated_or_retrieved_text, edited_text, detect_result))
            print(f"detect_result: {detect_result}")
            print("--------------------------------")
        self._save_edited_text(self.dataset.prompts, edit_texts, self.output_dir)
        if self.return_type == DetectionPipelineReturnType.FULL:
            return evaluation_result
        elif self.return_type == DetectionPipelineReturnType.SCORES:
            return [result.detect_result['score'] for result in evaluation_result]
        elif self.return_type == DetectionPipelineReturnType.IS_WATERMARKED:
            return [result.detect_result['is_watermarked'] for result in evaluation_result]
        

class WatermarkedTextDetectionPipeline_V2(WatermarkDetectionPipeline):
    """用於處理和檢測帶水印文本的增強版 pipeline"""

    def __init__(
        self,
        dataset: BaseDataset,
        watermark: BaseWatermark,
        output_dir: str,
        text_editor_list: List[TextEditor] = [],
        show_progress: bool = True,
        return_type: DetectionPipelineReturnType = DetectionPipelineReturnType.SCORES,
        extract_colors: bool = False,
        watermarked_texts_path: Optional[str] = None,
        generation_mode: str = 'load'  # 'load' 或 'generate'
    ) -> None:
        super().__init__(dataset, text_editor_list, show_progress, return_type)
        self.watermark = watermark
        self.output_dir = output_dir
        self.extract_colors = extract_colors
        self.watermarked_texts_path = watermarked_texts_path
        self.generation_mode = generation_mode
        os.makedirs(self.output_dir, exist_ok=True)

    def _get_iterable(self):
        """Return an iterable for the prompts."""
        return range(self.dataset.prompt_nums)

    def _load_watermarked_texts(self) -> List[Dict[str, str]]:
        """從文件載入水印文本"""
        try:
            with open(self.watermarked_texts_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logging.info(f"已載入 {len(data)} 個水印文本")
            return data
        except Exception as e:
            logging.error(f"載入水印文本時出錯: {str(e)}")
            raise

    def _generate_watermarked_text(self, prompt: str) -> str:
        """生成帶水印的文本"""
        try:
            return self.watermark.generate_watermarked_text(prompt)
        except Exception as e:
            logging.error(f"生成水印文本時出錯: {str(e)}")
            return ""

    def _generate_or_retrieve_text(self, dataset_index: int, watermark: BaseWatermark = None) -> str:
        """根據模式獲取水印文本"""
        prompt = self.dataset.get_prompt(dataset_index)
        
        if self.generation_mode == 'load' and self.watermarked_texts_path:
            # print(f"使用 load 模式，載入水印文本")
            data = self._load_watermarked_texts()
            for item in data:
                if item["prompt"] == prompt:
                    return item["watermarked_text"]
            print(f"找不到匹配的水印文本，將重新生成")
            logging.warning(f"找不到匹配的水印文本，將重新生成")
            return self._generate_watermarked_text(prompt)
        else:
            return self._generate_watermarked_text(prompt)

    def _save_texts(self, texts: List[str], prompts: List[str]) -> str:
        """保存水印文本"""
        # 如果是 load 模式，不需要保存文本
        if self.generation_mode == 'load':
            print(f"使用 load 模式，跳過保存水印文本")
            logging.info("使用 load 模式，跳過保存水印文本")
            return ""
        
        output_file = os.path.join(self.output_dir, "watermarked_texts.json")
        
        output_data = [
            {"prompt": prompt, "watermarked_text": text}
            for prompt, text in zip(prompts, texts)
        ]
        
        try:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            logging.info(f"已保存 {len(texts)} 個水印文本到 {output_file}")
            return output_file
        except Exception as e:
            logging.error(f"保存水印文本時出錯: {str(e)}")
            return ""

    def _extract_token_colors(self, text: str) -> List[Tuple]:
        """提取文本中每個 token 的顏色信息"""
        tokenizer = self.watermark.config.generation_tokenizer
        device = self.watermark.config.device
        
        encoded_text = tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0].to(device)

        if isinstance(self.watermark, EXP):
            # 將 encoded_text 轉換為 numpy array，與 EXP detect_watermark 方法保持一致
            encoded_array = encoded_text.cpu().numpy()
            
            exp_scores = []
            
            for i in range(len(encoded_array)):
                if i < self.watermark.config.prefix_length:
                    exp_scores.append("prefix")
                    continue
                # Seed RNG with the prefix of the encoded text
                self.watermark.utils.seed_rng(encoded_array[:i])
                
                # Generate random numbers for each token in the vocabulary
                random_numbers = torch.rand(self.watermark.config.vocab_size, generator=self.watermark.utils.rng)
                
                # Calculate score for the current token
                r = random_numbers[encoded_array[i]]
                score = log(1 / (1 - r))
                exp_scores.append(score)
            
            entropy_list = self._calculate_entropy(encoded_text)
            
            return [
                (token_id.item(), 
                 exp_scores[i],
                 entropy_list[i])
                for i, token_id in enumerate(encoded_text)
            ]
        
        if isinstance(self.watermark, (SWEET, EWD)):
            entropy_list = self.watermark.utils.calculate_entropy(
                self.watermark.config.generation_model,
                encoded_text
            )
            _, green_token_flags, _ = self.watermark.utils.score_sequence(encoded_text, entropy_list)
        else:
            if isinstance(self.watermark, (KGW, Unigram)):
                _, green_token_flags = self.watermark.utils.score_sequence(encoded_text)
            entropy_list = self._calculate_entropy(encoded_text)
        
        return [
            (token_id.item(), 
             "green" if flag == 1 else "red" if flag == 0 else "prefix",
             entropy_list[i] if i < len(entropy_list) else 0.0)
            for i, (token_id, flag) in enumerate(zip(encoded_text, green_token_flags))
        ]

    def _calculate_entropy(self, tokenized_text: torch.Tensor) -> List[float]:
        """計算每個 token 的熵值"""
        with torch.no_grad():
            output = self.watermark.config.generation_model(
                torch.unsqueeze(tokenized_text, 0),
                return_dict=True
            )
            probs = torch.softmax(output.logits, dim=-1)
            entropy = -torch.where(probs > 0, probs * probs.log(), probs.new([0.0])).sum(dim=-1)
            entropy = entropy[0].cpu().tolist()
            entropy.insert(0, -10000.0)
            return entropy[:-1]

    def _save_token_colors(self, texts: List[str]) -> str:
        """保存 token 顏色信息"""
        if not self.extract_colors:
            return ""
            
        output_file = os.path.join(self.output_dir, "watermarked_token_colors.json")
        
        all_results = []
        for i, text in enumerate(texts):
            try:
                token_colors = self._extract_token_colors(text)
                all_results.append({
                    "text_index": i,
                    "token_colors": token_colors
                })
            except Exception as e:
                logging.error(f"提取文本 {i+1} 的 token 顏色時出錯: {str(e)}")
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, ensure_ascii=False, separators=(",", ":"))
            logging.info(f"已保存 token 顏色信息到 {output_file}")
            return output_file
        except Exception as e:
            logging.error(f"保存 token 顏色時出錯: {str(e)}")
            return ""
    
    @timer
    def evaluate(self) -> Union[List[WatermarkDetectionResult], List[float], List[bool]]:
        """執行評估流程"""
        evaluation_results = []
        processed_texts = []
        prompts = []
        
        bar = self._get_progress_bar(self._get_iterable())
        
        print("========= watermarked text detection ==========")
        for index in bar:
            try:
                # 獲取或生成文本
                text = self._generate_or_retrieve_text(index)
                if not text:
                    continue
                    
                # 編輯文本
                prompt = self.dataset.get_prompt(index)
                edited_text = self._edit_text(text, prompt)
                
                # 檢測水印
                detect_result = self._detect_watermark(edited_text, self.watermark)
                
                # 保存結果
                evaluation_results.append(
                    WatermarkDetectionResult(text, edited_text, detect_result)
                )
                processed_texts.append(edited_text)
                prompts.append(prompt)
                
                if self.show_progress:
                    print(f"檢測結果: {detect_result}")
                    print("-" * 60)
                    
            except Exception as e:
                logging.error(f"處理文本 {index} 時出錯: {str(e)}")
                continue
        print("========= watermarked text detection end ==========")
        # 保存處理後的文本
        self._save_texts(processed_texts, prompts)
        
        # 提取並保存 token 顏色
        if self.extract_colors:
            self._save_token_colors(processed_texts)
        
        # 根據返回類型返回結果
        if self.return_type == DetectionPipelineReturnType.FULL:
            return evaluation_results
        elif self.return_type == DetectionPipelineReturnType.SCORES:
            return [result.detect_result['score'] for result in evaluation_results]
        else:  # DetectionPipelineReturnType.IS_WATERMARKED
            # return [result.detect_result['is_watermarked'] for result in evaluation_results]
            return [bool(result.detect_result['is_watermarked']) for result in evaluation_results]


class UnwatermarkedTextDetectionPipeline_V2(WatermarkDetectionPipeline):
    """用於處理和檢測無水印文本的增強版 pipeline"""

    def __init__(
        self,
        dataset: BaseDataset,
        watermark: BaseWatermark,
        output_dir: str,
        text_editor_list: List[TextEditor] = [],
        show_progress: bool = True,
        return_type: DetectionPipelineReturnType = DetectionPipelineReturnType.SCORES,
        extract_colors: bool = False,
        text_source_mode: str = 'natural'  # 'natural' 或 'generated'
    ) -> None:
        # 驗證 text_source_mode
        if text_source_mode not in ['natural', 'generated']:
            raise InvalidTextSourceModeError(text_source_mode)
            
        super().__init__(dataset, text_editor_list, show_progress, return_type)
        self.watermark = watermark
        self.output_dir = output_dir
        self.extract_colors = extract_colors
        self.text_source_mode = text_source_mode
        os.makedirs(self.output_dir, exist_ok=True)

    def _get_iterable(self):
        """Return an iterable for the natural texts or prompts."""
        if self.text_source_mode == 'natural':
            return range(self.dataset.natural_text_nums)
        else:
            return range(self.dataset.prompt_nums)

    def _extract_token_colors(self, text: str) -> List[Tuple]:
        """提取文本中每個 token 的顏色信息"""
        tokenizer = self.watermark.config.generation_tokenizer
        device = self.watermark.config.device
        
        encoded_text = tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0].to(device)

        if isinstance(self.watermark, EXP):
            # 將 encoded_text 轉換為 numpy array，與 EXP detect_watermark 方法保持一致
            encoded_array = encoded_text.cpu().numpy()
            
            exp_scores = []
            
            for i in range(len(encoded_array)):
                if i < self.watermark.config.prefix_length:
                    exp_scores.append("prefix")
                    continue
                # Seed RNG with the prefix of the encoded text
                self.watermark.utils.seed_rng(encoded_array[:i])
                
                # Generate random numbers for each token in the vocabulary
                random_numbers = torch.rand(self.watermark.config.vocab_size, generator=self.watermark.utils.rng)
                
                # Calculate score for the current token
                r = random_numbers[encoded_array[i]]
                score = log(1 / (1 - r))
                exp_scores.append(score)
            
            entropy_list = self._calculate_entropy(encoded_text)
            
            return [
                (token_id.item(), 
                 exp_scores[i],
                 entropy_list[i])
                for i, token_id in enumerate(encoded_text)
            ]
        
        if isinstance(self.watermark, (SWEET, EWD)):
            entropy_list = self.watermark.utils.calculate_entropy(
                self.watermark.config.generation_model,
                encoded_text
            )
            _, green_token_flags, _ = self.watermark.utils.score_sequence(encoded_text, entropy_list)
        else:
            if isinstance(self.watermark, (KGW, Unigram)):
                _, green_token_flags = self.watermark.utils.score_sequence(encoded_text)
            entropy_list = self._calculate_entropy(encoded_text)
        
        return [
            (token_id.item(), 
             "green" if flag == 1 else "red" if flag == 0 else "prefix",
             entropy_list[i] if i < len(entropy_list) else 0.0)
            for i, (token_id, flag) in enumerate(zip(encoded_text, green_token_flags))
        ]

    def _calculate_entropy(self, tokenized_text: torch.Tensor) -> List[float]:
        """計算每個 token 的熵值"""
        with torch.no_grad():
            output = self.watermark.config.generation_model(
                torch.unsqueeze(tokenized_text, 0),
                return_dict=True
            )
            probs = torch.softmax(output.logits, dim=-1)
            entropy = -torch.where(probs > 0, probs * probs.log(), probs.new([0.0])).sum(dim=-1)
            entropy = entropy[0].cpu().tolist()
            entropy.insert(0, -10000.0)
            return entropy[:-1]

    def _save_token_colors(self, texts: List[str]) -> str:
        """保存 token 顏色信息"""
        if not self.extract_colors:
            return ""
            
        output_file = os.path.join(self.output_dir, "unwatermarked_token_colors.json")
        
        all_results = []
        for i, text in enumerate(texts):
            try:
                token_colors = self._extract_token_colors(text)
                all_results.append({
                    "text_index": i,
                    "token_colors": token_colors
                })
            except Exception as e:
                logging.error(f"提取文本 {i+1} 的 token 顏色時出錯: {str(e)}")
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, ensure_ascii=False, separators=(",", ":"))
            logging.info(f"已保存 token 顏色信息到 {output_file}")
            return output_file
        except Exception as e:
            logging.error(f"保存 token 顏色時出錯: {str(e)}")
            return ""

    def _generate_or_retrieve_text(self, dataset_index, watermark):
        """Retrieve unwatermarked text from the dataset."""
        if self.text_source_mode == 'natural':
            return self.dataset.get_natural_text(dataset_index)
        else:
            prompt = self.dataset.get_prompt(dataset_index)
            return watermark.generate_unwatermarked_text(prompt)

    def evaluate(self) -> Union[List[WatermarkDetectionResult], List[float], List[bool]]:
        """執行評估流程"""
        evaluation_results = []
        processed_texts = []
        
        bar = self._get_progress_bar(self._get_iterable())
        
        print("========= unwatermarked text detection ==========")
        for index in bar:
            try:
                # 獲取文本
                text = self._generate_or_retrieve_text(index, self.watermark)
                if not text:
                    continue
                    
                # 編輯文本
                prompt = self.dataset.get_prompt(index) if self.text_source_mode == 'generated' else None
                edited_text = self._edit_text(text, prompt)
                
                # 檢測水印
                detect_result = self._detect_watermark(edited_text, self.watermark)
                
                # 保存結果
                evaluation_results.append(
                    WatermarkDetectionResult(text, edited_text, detect_result)
                )
                processed_texts.append(edited_text)
                
                if self.show_progress:
                    print(f"檢測結果: {detect_result}")
                    print("-" * 60)
                    
            except Exception as e:
                logging.error(f"處理文本 {index} 時出錯: {str(e)}")
                continue
        print("========= unwatermarked text detection end ==========")
        # 提取並保存 token 顏色
        if self.extract_colors:
            self._save_token_colors(processed_texts)
        
        # 根據返回類型返回結果
        if self.return_type == DetectionPipelineReturnType.FULL:
            return evaluation_results
        elif self.return_type == DetectionPipelineReturnType.SCORES:
            return [result.detect_result['score'] for result in evaluation_results]
        else:  # DetectionPipelineReturnType.IS_WATERMARKED
            # return [result.detect_result['is_watermarked'] for result in evaluation_results]
            return [bool(result.detect_result['is_watermarked']) for result in evaluation_results]
        

class SignatureAwareWatermarkDetectionPipeline_V2(WatermarkedTextDetectionPipeline_V2):
    """支援簽名感知的水印檢測管道 V2"""
    
    def __init__(
        self,
        dataset: BaseDataset,
        watermark: BaseWatermark,
        output_dir: str,
        text_editor_list: List[TextEditor] = [],
        show_progress: bool = True,
        return_type: DetectionPipelineReturnType = DetectionPipelineReturnType.SCORES,
        extract_colors: bool = False,
        watermarked_texts_path: Optional[str] = None,
        generation_mode: str = 'load',  # 'load' 或 'generate'
        signature_config: Optional[Dict] = None,
    ) -> None:
        """
        初始化簽名感知的水印檢測管道。
        
        Args:
            dataset: 用於評估的數據集
            watermark: 水印生成器
            output_dir: 輸出目錄
            text_editor_list: 文本編輯器列表
            show_progress: 是否顯示進度條
            return_type: 檢測結果的返回類型
            extract_colors: 是否提取 token 顏色
            watermarked_texts_path: 預加載的水印文本路徑
            generation_mode: 生成模式 ('load' 或 'generate')
            signature_config: 簽名配置，包含：
                - use_ngram: 是否使用 n-gram
                - n: n-gram 的 n 值（如果使用 n-gram）
        """
        super().__init__(
            dataset=dataset,
            watermark=watermark,
            output_dir=output_dir,
            text_editor_list=text_editor_list,
            show_progress=show_progress,
            return_type=return_type,
            extract_colors=extract_colors,
            watermarked_texts_path=watermarked_texts_path,
            generation_mode=generation_mode,
        )
        
        self.signature_config = signature_config or {}
        self.signature_collector = None
        self.ngram_collector = None
        self.signature_detector = None
        
        # 設置簽名相關路徑
        self.signature_file = os.path.join(output_dir, "signature_set.json")
        if self.signature_config.get('use_ngram'):
            self.ngram_signature_file = os.path.join(output_dir, f"ngram{self.signature_config['n']}_signature_set.json")
    
    def _setup_signature_detector(self, texts: List[str]) -> None:
        """設置簽名檢測器"""
        # 收集基本簽名
        self.signature_collector = SignatureSetCollector(self.watermark)
        logging.info("收集簽名集...")
        print(f"收集簽名集...")
        for text in texts:
            self.signature_collector.collect_from_text(text)
        self.signature_collector.save_signature_set(self.signature_file)
        
        # 如果啟用 n-gram，收集 n-gram 簽名
        if self.signature_config.get('use_ngram'):
            from watermark.signature.ngram import NGramSignatureSetCollector
            n = self.signature_config['n']
            self.ngram_collector = NGramSignatureSetCollector(self.watermark, n=n)
            
            logging.info(f"收集 {n}-gram 簽名集...")
            print(f"收集 {n}-gram 簽名集...")
            for text in texts:
                self.ngram_collector.collect_from_text(text)
            self.ngram_collector.save_ngram_signature_set(self.ngram_signature_file)
        
        # 創建簽名檢測器
        self._create_signature_detector()
    
    def _create_signature_detector(self) -> None:
        """創建簽名檢測器"""
        if self.signature_config.get('use_ngram'):
            # 使用 n-gram 簽名檢測器
            if isinstance(self.watermark, KGW):
                self.signature_detector = KGWNGramSignature(
                    algorithm_config=f'config/{self.watermark.__class__.__name__}.json',
                    transformers_config=self.watermark.config.transformers_config,
                    signature_set=self.signature_collector.signature_set,
                    n=self.signature_config['n'],
                    ngram_signature_set=self.ngram_collector.ngram_signature_set
                )
            elif isinstance(self.watermark, SWEET):
                self.signature_detector = SweetNGramSignature(
                    algorithm_config=f'config/{self.watermark.__class__.__name__}.json',
                    transformers_config=self.watermark.config.transformers_config,
                    signature_set=self.signature_collector.signature_set,
                    n=self.signature_config['n'],
                    ngram_signature_set=self.ngram_collector.ngram_signature_set
                )
            elif isinstance(self.watermark, Unigram):
                self.signature_detector = UnigramNGramSignature(
                    algorithm_config=f'config/{self.watermark.__class__.__name__}.json',
                    transformers_config=self.watermark.config.transformers_config,
                    signature_set=self.signature_collector.signature_set,
                    n=self.signature_config['n'],
                    ngram_signature_set=self.ngram_collector.ngram_signature_set
                )
            # 可以添加其他水印類型的支援
        else:
            # 使用基本簽名檢測器
            if isinstance(self.watermark, KGW):
                self.signature_detector = KGWSignature(
                    algorithm_config=f'config/{self.watermark.__class__.__name__}.json',
                    transformers_config=self.watermark.config.transformers_config,
                    signature_set=self.signature_collector.signature_set
                )
            elif isinstance(self.watermark, SWEET):
                self.signature_detector = SweetSignature(
                    algorithm_config=f'config/{self.watermark.__class__.__name__}.json',
                    transformers_config=self.watermark.config.transformers_config,
                    signature_set=self.signature_collector.signature_set
                )
            elif isinstance(self.watermark, Unigram):
                self.signature_detector = UnigramSignature(
                    algorithm_config=f'config/{self.watermark.__class__.__name__}.json',
                    transformers_config=self.watermark.config.transformers_config,
                    signature_set=self.signature_collector.signature_set
                )
            # 可以添加其他水印類型的支援
    
    def _compare_detection_results(self, text: str) -> Dict[str, Any]:
        """比較標準水印和簽名水印的檢測結果"""
        # 標準水印檢測
        standard_result = self.watermark.detect_watermark(text)
        
        # 簽名水印檢測
        signature_result = self.signature_detector.detect_watermark(text)

        detection_results = {
            "standard": standard_result,
            "signature": signature_result,
        }

        return detection_results
    
    @timer
    def evaluate(self) -> Union[List[WatermarkDetectionResult], List[float], List[bool]]:
        """執行評估流程"""
        evaluation_results = []
        processed_texts = []
        prompts = []
        
        bar = self._get_progress_bar(self._get_iterable())
        
        # 首先獲取所有文本
        texts = []
        for index in bar:
            text = self._generate_or_retrieve_text(index)
            if text:
                texts.append(text)
        
        # 設置簽名檢測器
        self._setup_signature_detector(texts)
        
        print("========= watermarked signature detection ==========")
        # 進行檢測
        for index, text in enumerate(texts):
            try:
                # 編輯文本
                prompt = self.dataset.get_prompt(index)
                edited_text = self._edit_text(text, prompt)
                
                # 比較檢測結果
                detection_results = self._compare_detection_results(edited_text)
                
                # 保存結果
                evaluation_results.append(
                    WatermarkDetectionResult(
                        text, 
                        edited_text, 
                        detection_results
                    )
                )
                processed_texts.append(edited_text)
                prompts.append(prompt)
                
                if self.show_progress:
                    print(f"檢測結果: {detection_results}")
                    print("-" * 60)
                    
            except Exception as e:
                logging.error(f"處理文本 {index} 時出錯: {str(e)}")
                continue
        print("========= watermarked signature detection end ==========")
        # 保存處理後的文本
        self._save_texts(processed_texts, prompts)
        
        # 提取並保存 token 顏色
        if self.extract_colors:
            self._save_token_colors(processed_texts)
        
        # 根據返回類型返回結果
        if self.return_type == DetectionPipelineReturnType.FULL:
            return evaluation_results
        elif self.return_type == DetectionPipelineReturnType.SCORES:
            return [result.detect_result['signature']['score'] for result in evaluation_results]
        else:  # DetectionPipelineReturnType.IS_WATERMARKED
            return [result.detect_result['signature']['is_watermarked'] for result in evaluation_results]


class SignatureAwareUnwatermarkedTextDetectionPipeline_V2(UnwatermarkedTextDetectionPipeline_V2):
    """支援簽名感知的非水印文本檢測管道 V2"""
    
    def __init__(
        self,
        dataset: BaseDataset,
        watermark: BaseWatermark,
        output_dir: str,
        text_editor_list: List[TextEditor] = [],
        show_progress: bool = True,
        return_type: DetectionPipelineReturnType = DetectionPipelineReturnType.SCORES,
        extract_colors: bool = False,
        text_source_mode: str = 'natural',  # 'natural' 或 'generated'
        signature_config: Optional[Dict] = None,
    ) -> None:
        """
        初始化簽名感知的非水印文本檢測管道。
        
        Args:
            dataset: 用於評估的數據集
            watermark: 水印生成器
            output_dir: 輸出目錄
            text_editor_list: 文本編輯器列表
            show_progress: 是否顯示進度條
            return_type: 檢測結果的返回類型
            extract_colors: 是否提取 token 顏色
            text_source_mode: 文本來源模式 ('natural' 或 'generated')
            signature_config: 簽名配置，包含：
                - use_ngram: 是否使用 n-gram
                - n: n-gram 的 n 值（如果使用 n-gram）
        """
        # 驗證 text_source_mode
        if text_source_mode not in ['natural', 'generated']:
            raise InvalidTextSourceModeError(text_source_mode)
            
        super().__init__(
            dataset=dataset,
            watermark=watermark,
            output_dir=output_dir,
            text_editor_list=text_editor_list,
            show_progress=show_progress,
            return_type=return_type,
            extract_colors=extract_colors,
            text_source_mode=text_source_mode
        )
        
        self.signature_config = signature_config or {}
        self.signature_detector = None
        
        # 設置簽名相關路徑 - 使用與 SignatureAwareWatermarkDetectionPipeline_V2 相同的路徑
        self.signature_file = os.path.join(output_dir, "signature_set.json")
        if self.signature_config.get('use_ngram'):
            self.ngram_signature_file = os.path.join(output_dir, f"ngram{self.signature_config['n']}_signature_set.json")
    
    def _load_signature_set(self) -> Dict:
        """載入由 SignatureAwareWatermarkDetectionPipeline_V2 產生的簽名集"""
        try:
            with open(self.signature_file, 'r', encoding='utf-8') as f:
                signature_set = json.load(f)
            logging.info(f"已載入簽名集: {self.signature_file}")
            print(f"已載入簽名集: {self.signature_file}")
            return signature_set
        except FileNotFoundError:
            logging.error(f"找不到簽名集文件: {self.signature_file}")
            print(f"找不到簽名集文件: {self.signature_file}")
            return {}
    
    def _load_ngram_signature_set(self) -> Dict:
        """載入由 SignatureAwareWatermarkDetectionPipeline_V2 產生的 n-gram 簽名集"""
        if not self.signature_config.get('use_ngram'):
            return set()
        
        try:
            from watermark.signature.ngram import NGramSignatureSetUtils
            loaded_n, loaded_ngram_set = NGramSignatureSetUtils.load(self.ngram_signature_file)
            
            # 確認加載的 n 與配置中的 n 一致
            configured_n = self.signature_config.get('n', 1)
            if loaded_n != configured_n:
                print(f"Warning: Loaded n-gram set has n={loaded_n}, but configuration expects n={configured_n}")
            
            logging.info(f"已載入 {len(loaded_ngram_set)} 個 {loaded_n}-gram 簽名: {self.ngram_signature_file}")
            print(f"已載入 {len(loaded_ngram_set)} 個 {loaded_n}-gram 簽名: {self.ngram_signature_file}")
            
            return loaded_ngram_set  # 返回 Set[Tuple[int, ...]]
        except FileNotFoundError:
            logging.error(f"找不到 n-gram 簽名集文件: {self.ngram_signature_file}")
            print(f"找不到 n-gram 簽名集文件: {self.ngram_signature_file}")
            return set()
        except Exception as e:
            logging.error(f"加載 n-gram 簽名集時出錯: {str(e)}")
            return set()
    
    def _setup_signature_detector(self) -> None:
        """設置簽名檢測器 - 使用已產生的簽名集"""
        # 載入簽名集
        signature_set = self._load_signature_set()
        ngram_signature_set = self._load_ngram_signature_set()
        
        # 創建簽名檢測器
        if self.signature_config.get('use_ngram'):
            # 使用 n-gram 簽名檢測器
            if isinstance(self.watermark, KGW):
                self.signature_detector = KGWNGramSignature(
                    algorithm_config=f'config/{self.watermark.__class__.__name__}.json',
                    transformers_config=self.watermark.config.transformers_config,
                    signature_set=signature_set,
                    n=self.signature_config['n'],
                    ngram_signature_set=ngram_signature_set
                )
            elif isinstance(self.watermark, SWEET):
                self.signature_detector = SweetNGramSignature(
                    algorithm_config=f'config/{self.watermark.__class__.__name__}.json',
                    transformers_config=self.watermark.config.transformers_config,
                    signature_set=signature_set,
                    n=self.signature_config['n'],
                    ngram_signature_set=ngram_signature_set
                )
            elif isinstance(self.watermark, Unigram):
                self.signature_detector = UnigramNGramSignature(
                    algorithm_config=f'config/{self.watermark.__class__.__name__}.json',
                    transformers_config=self.watermark.config.transformers_config,
                    signature_set=signature_set,
                    n=self.signature_config['n'],
                    ngram_signature_set=ngram_signature_set
                )
        else:
            # 使用基本簽名檢測器
            if isinstance(self.watermark, KGW):
                self.signature_detector = KGWSignature(
                    algorithm_config=f'config/{self.watermark.__class__.__name__}.json',
                    transformers_config=self.watermark.config.transformers_config,
                    signature_set=signature_set
                )
            elif isinstance(self.watermark, SWEET):
                self.signature_detector = SweetSignature(
                    algorithm_config=f'config/{self.watermark.__class__.__name__}.json',
                    transformers_config=self.watermark.config.transformers_config,
                    signature_set=signature_set
                )
            elif isinstance(self.watermark, Unigram):
                self.signature_detector = UnigramSignature(
                    algorithm_config=f'config/{self.watermark.__class__.__name__}.json',
                    transformers_config=self.watermark.config.transformers_config,
                    signature_set=signature_set
                )
    
    def _compare_detection_results(self, text: str) -> Dict[str, Any]:
        """比較標準水印和簽名水印的檢測結果"""
        # 標準水印檢測
        standard_result = self.watermark.detect_watermark(text)
        
        # 簽名水印檢測
        signature_result = self.signature_detector.detect_watermark(text)

        detection_results = {
            "standard": standard_result,
            "signature": signature_result,
        }

        return detection_results
    
    @timer
    def evaluate(self) -> Union[List[WatermarkDetectionResult], List[float], List[bool]]:
        """執行評估流程"""
        evaluation_results = []
        processed_texts = []
        
        # 首先設置簽名檢測器，使用已產生的簽名集
        self._setup_signature_detector()
        
        bar = self._get_progress_bar(self._get_iterable())

        print("========= unwatermarked signature detection ==========")
        # 進行檢測
        for index in bar:
            try:
                # 獲取文本
                text = self._generate_or_retrieve_text(index, self.watermark)
                if not text:
                    continue
                    
                # 編輯文本
                prompt = self.dataset.get_prompt(index) if self.text_source_mode == 'generated' else None
                edited_text = self._edit_text(text, prompt)
                
                # 比較檢測結果
                detection_results = self._compare_detection_results(edited_text)
                
                # 保存結果
                evaluation_results.append(
                    WatermarkDetectionResult(
                        text, 
                        edited_text, 
                        detection_results
                    )
                )
                processed_texts.append(edited_text)
                
                if self.show_progress:
                    print(f"檢測結果: {detection_results}")
                    print("-" * 60)
                    
            except Exception as e:
                logging.error(f"處理文本 {index} 時出錯: {str(e)}")
                continue
        print("========= unwatermarked signature detection end ==========")
        # 提取並保存 token 顏色
        if self.extract_colors:
            self._save_token_colors(processed_texts)
        
        # 根據返回類型返回結果
        if self.return_type == DetectionPipelineReturnType.FULL:
            return evaluation_results
        elif self.return_type == DetectionPipelineReturnType.SCORES:
            return [result.detect_result['signature']['score'] for result in evaluation_results]
        else:  # DetectionPipelineReturnType.IS_WATERMARKED
            return [result.detect_result['signature']['is_watermarked'] for result in evaluation_results]