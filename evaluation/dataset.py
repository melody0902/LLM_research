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

# ===========================================
# dataset.py
# Description: Dataset classes for evaluation
# ===========================================

import json


class BaseDataset:
    """Base class for dataset."""

    def __init__(self, max_samples: int = 200):
        """
        Initialize the dataset.
        
        Parameters:
            max_samples (int): Maximum number of samples to load. Default is 200.
        """
        self.max_samples = max_samples
        self.prompts = []
        self.natural_texts = []
        self.references = []
    def __getitem__(self, index):
        return {
            "prompt": self.prompts[index],
            "natural_text": self.natural_texts[index] if index < len(self.natural_texts) else None,
            "reference": self.references[index] if index < len(self.references) else None
        }

    def __len__(self):
        return len(self.prompts)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


    @property
    def prompt_nums(self):
        """Return the number of prompts."""
        return len(self.prompts)

    @property
    def natural_text_nums(self):
        """Return the number of natural texts."""
        return len(self.natural_texts)

    @property
    def reference_nums(self):
        """Return the number of references."""
        return len(self.references)

    def get_prompt(self, index):
        """Return the prompt at the specified index."""
        return self.prompts[index]

    def get_natural_text(self, index):
        """Return the natural text at the specified index."""
        return self.natural_texts[index]

    def get_reference(self, index):
        """Return the reference at the specified index."""
        return self.references[index]

    def load_data(self):
        """Load and process data to populate prompts, natural_texts, and references."""
        pass


class C4Dataset(BaseDataset):
    """Dataset class for C4 dataset."""

    def __init__(self, data_source: str, max_samples: int = 200):
        """
            Initialize the C4 dataset.

            Parameters:
                data_source (str): The path to the C4 dataset file.
        """
        super().__init__(max_samples)
        self.data_source = data_source
        self.load_data()
    
    def load_data(self):
        """Load data from the C4 dataset file."""
        with open(self.data_source, 'r') as f:
           lines = f.readlines()
        for line in lines[:self.max_samples]:
            item = json.loads(line)
            self.prompts.append(item['prompt'])
            self.natural_texts.append(item['natural_text'])


class WMT16DE_ENDataset(BaseDataset):
    """Dataset class for WMT16 DE-EN dataset."""

    def __init__(self, data_source: str, max_samples: int = 200) -> None:
        """
            Initialize the WMT16 DE-EN dataset.

            Parameters:
                data_source (str): The path to the WMT16 DE-EN dataset file.
        """
        super().__init__(max_samples)
        self.data_source = data_source
        self.load_data()
    
    def load_data(self):
        """Load data from the WMT16 DE-EN dataset file."""
        with open(self.data_source, 'r') as f:
            lines = f.readlines()
        for line in lines[:self.max_samples]:
            item = json.loads(line)
            self.prompts.append(item['de'])
            self.references.append(item['en'])


class HumanEvalDataset(BaseDataset):
    """Dataset class for HumanEval dataset."""

    def __init__(self, data_source: str, max_samples: int = 200) -> None:
        """
            Initialize the HumanEval dataset.

            Parameters:
                data_source (str): The path to the HumanEval dataset file.
        """
        super().__init__(max_samples)
        self.data_source = data_source
        self.load_data()
    
    def load_data(self):
        """Load data from the HumanEval dataset file."""
        with open(self.data_source, 'r') as f:
            lines = f.readlines()
        for line in lines[:self.max_samples]:
            item = json.loads(line)
            # process prompt
            prompt = item['prompt']
            sections = prompt.split(">>>")
            prompt = sections[0]
            if len(sections) > 1:
                prompt += '\"\"\"'

            self.prompts.append(prompt)
            self.references.append({'task': prompt, 'test': item['test'], 'entry_point': item['entry_point']})

class TraditionalChineseDataset(BaseDataset):
    """Dataset class for Traditional Chinese data."""

    def __init__(self, data_source: str, tokenizer, max_samples: int = 200):
        """
        Initialize the Traditional Chinese dataset.

        Parameters:
            data_source (str): The path to the Traditional Chinese dataset file.
            tokenizer: The tokenizer used to process prompts.
        """
        super().__init__(max_samples)
        self.data_source = data_source
        self.tokenizer = tokenizer  # 儲存 tokenizer
        self.load_data()

    def load_data(self):
        with open(self.data_source, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        sys_prompt = "你是一個來自台灣的AI助理，你的名字是 TAIDE，樂於以台灣人的立場幫助使用者，會用繁體中文回答問題。"
        for line in lines[:self.max_samples]:
            item = json.loads(line)
            chat = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": item['prompt']},
            ]
            prompt = self.tokenizer.apply_chat_template(chat, tokenize=False)
            self.prompts.append(prompt)

class ZHTWC4Dataset(BaseDataset):
    """Dataset class for ZHTW dataset in normal JSON or JSONL format."""

    def __init__(self, data_source: str, tokenizer=None, max_samples: int = None):
        """
            Initialize the ZHTW dataset from a JSONL or JSON file.

            Parameters:
                data_source (str): The path to the ZHTW dataset file.
                tokenizer: The tokenizer used to process prompts (optional).
                max_samples: Maximum number of samples to load.
        """ 
        super().__init__(max_samples)
        self.data_source = data_source
        self.tokenizer = tokenizer
        self.load_data()

    def load_data(self):
        with open(self.data_source, "r", encoding="utf-8") as f:
            # 支援 JSON Lines 格式，每行是一筆 JSON
            data = [json.loads(line) for line in f if line.strip()]

        for item in data[:self.max_samples]:
            # 支援 {"text": "..."} 格式
            text = item.get("prompt", "").strip()
            if text:
                self.prompts.append(text)

if __name__ == '__main__':
    d1 = C4Dataset('dataset/c4/processed_c4.json', max_samples=100)
    d2 = WMT16DE_ENDataset('dataset/wmt16_de_en/validation.jsonl', max_samples=100)
    d3 = HumanEvalDataset('dataset/HumanEval/test.jsonl', max_samples=100)
    d4 = TraditionalChineseDataset('dataset/zhtw/TraditionalChinese.jsonl', max_samples=100)