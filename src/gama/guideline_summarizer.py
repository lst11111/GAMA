"""
BioNER Guideline Summarization Agent — Pipeline Wrapper

Usage in a pipeline:
    from gama.guideline_summarizer import BioNERSummarizeAgent

    agent = BioNERSummarizeAgent(api_key=os.getenv("BIO_NER_API_KEY"), base_url="...")
    guidelines = agent.summarize(
        train_data=[{"text": "...", "entities": [{"text": "BRCA1", "type": "GENE"}]}],
        output_dir="guidelines_output",
        top_k=10,
        max_workers=8,
    )
"""

import json
import os
import random
import re
import time
import tempfile
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from openai import OpenAI

# Regex to extract JSON / list from LLM output
result_pattern = re.compile(r'\{.*\}', re.DOTALL)
valid_pattern = re.compile(r'\[\[.*?\]\]', re.DOTALL)


class BioNERSummarizeAgent:
    """
    Data-driven annotation guidelines summarization agent.
    Three-step pipeline: Summarize → Self-Verify → Aggregate.
    """

    def __init__(self, api_key: str, model: str = "",
                 base_url: str = "",
                 verbose: bool = True):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.verbose = verbose
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    # ───────────────────────── Public API ─────────────────────────

    def summarize(self, train_data, output_dir="guidelines_output",
                  top_k=10, max_workers=8, sample_size=None,
                  skip_step1=False, skip_step2=False):
        """
        train_data: path to JSON file OR list of dicts with 'text' and 'entities'
        output_dir: directory to save intermediate results and final guidelines
        Returns: dict of {entity_type: [rule1, rule2, ...]}
        """
        # Resolve data
        if isinstance(train_data, str):
            data_path = train_data
        else:
            # Write to temp file so the pipeline steps can read/write consistently
            tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.json',
                                              delete=False, encoding='utf-8')
            json.dump(train_data, tmp, ensure_ascii=False)
            tmp.close()
            data_path = tmp.name

        summarizer = _GuidelineSummarizerImpl(
            self.client, self.model, self.verbose
        )

        guidelines = summarizer.run(
            train_data_path=data_path,
            output_dir=output_dir,
            top_k=top_k,
            max_workers=max_workers,
            skip_step1=skip_step1,
            skip_step2=skip_step2,
            sample_size=sample_size,
        )

        # Clean up temp file
        if not isinstance(train_data, str) and os.path.exists(data_path):
            os.remove(data_path)

        return guidelines


# ───────────────────────── Internal Implementation ─────────────────────────


class _GuidelineSummarizerImpl:
    def __init__(self, client, model, verbose=True):
        self.client = client
        self.model = model
        self.verbose = verbose

    def log(self, message, level="INFO"):
        if self.verbose:
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp}] [{level}] {message}")

    # ───────────────────────── Prompts ─────────────────────────

    def _summarize_prompt(self, input_text, input_annotations):
        return f"""\
Task: You are an expert in biomedical named entity recognition (BioNER). Summarize the generic annotation rules for each entity category based on the provided biomedical text and its annotations.
The output must be a JSON object where keys are entity categories and values are lists of short descriptive patterns.

Guidelines:
(1) Do NOT include specific entity names (e.g., gene names, disease names) — describe general patterns or semantic features.
(2) Only summarize rules for categories that appear in the annotations.
(3) For each annotation, generate exactly one rule.
(4) The order and count of rules must match the annotations exactly.

Examples:
Input: "Immunohistochemical staining was positive for S-100 in all 9 cases stained."
Annotations: [["S-100", "GENE"]]
Output: {{"GENE": ["protein biomarker identifier"]}}

Input: "BRCA1 gene mutations increase the risk of breast cancer."
Annotations: [["BRCA1", "GENE"], ["breast cancer", "DISEASE"]]
Output: {{"GENE": ["gene symbol abbreviation"], "DISEASE": ["cancer type"]}}

Input: "Chloramphenicol acetyltransferase assays examining the ability of IE86 to repress the HCMV promoter demonstrated functional integrity."
Annotations: [["Chloramphenicol acetyltransferase", "GENE"], ["IE86", "GENE"], ["HCMV", "GENE"]]
Output: {{"GENE": ["enzyme full name", "protein accession shorthand", "virus gene abbreviation"]}}

Now summarize:
Input: {input_text}
Annotations: {input_annotations}
Output:
"""

    def _validate_prompt(self, input_text, summarized_rules):
        return f"""\
Task: You are an expert in biomedical named entity recognition (BioNER). Identify all named entities in the text using ONLY the provided rules.
Rules are in JSON format: keys = entity categories, values = list of pattern descriptions.
Output must be a flat list of [entity_text, entity_type] pairs in order of appearance in the text.

Examples:
Input: "Immunohistochemical staining was positive for S-100 in all 9 cases stained."
Rules: {{"GENE": ["protein biomarker identifier"]}}
Output: [["S-100", "GENE"]]

Input: "BRCA1 gene mutations increase the risk of breast cancer."
Rules: {{"GENE": ["gene symbol abbreviation"], "DISEASE": ["cancer type"]}}
Output: [["BRCA1", "GENE"], ["breast cancer", "DISEASE"]]

Input: "Chloramphenicol acetyltransferase assays examining the ability of IE86 to repress the HCMV promoter demonstrated functional integrity."
Rules: {{"GENE": ["enzyme full name", "protein accession shorthand", "virus gene abbreviation"]}}
Output: [["Chloramphenicol acetyltransferase", "GENE"], ["IE86", "GENE"], ["HCMV", "GENE"]]

Now identify:
Input: {input_text}
Rules: {summarized_rules}
Output:
"""

    # ───────────────────────── API Call ─────────────────────────

    def _call_llm(self, prompt, max_retries=3):
        for attempt in range(max_retries):
            try:
                start_time = time.time()
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.8,
                    max_tokens=256,
                )
                elapsed = time.time() - start_time
                if self.verbose:
                    token_info = ""
                    if hasattr(response, 'usage') and response.usage:
                        token_info = (f", tokens: prompt={response.usage.prompt_tokens}, "
                                      f"completion={response.usage.completion_tokens}")
                    self.log(f"API 响应成功，耗时: {elapsed:.2f} 秒{token_info}")
                return response.choices[0].message.content
            except Exception as e:
                if attempt == max_retries - 1:
                    self.log(f"API 调用失败 (已重试 {max_retries} 次): {str(e)}", level="ERROR")
                    return None
                wait = 2 ** attempt
                self.log(f"API 调用失败，{wait} 秒后重试 ({attempt+1}/{max_retries}): {str(e)}",
                         level="WARNING")
                time.sleep(wait)

    # ───────────────────────── Helpers ─────────────────────────

    @staticmethod
    def _extract_entity_labels(entities):
        return [[e["text"], e["type"]] for e in entities]

    @staticmethod
    def _type_num_equal(labels, result):
        label_counts = {}
        for label in labels:
            t = label[-1]
            label_counts[t] = label_counts.get(t, 0) + 1
        result_counts = {}
        for k, v in result.items():
            result_counts[k] = result_counts.get(k, 0) + len(v)
        return label_counts == result_counts

    @staticmethod
    def _get_correspondings(labels, result):
        type_counter = {}
        pairs = []
        for label in labels:
            t = label[-1]
            idx = type_counter.get(t, 0)
            type_counter[t] = idx + 1
            rule = result[t][idx]
            pairs.append([label, rule])
        return pairs

    # ───────────────────────── Step 1 ─────────────────────────

    def _summarize_one(self, item):
        text = item["text"]
        entities = item.get("entities", [])
        if len(entities) == 0:
            return None

        entity_labels = self._extract_entity_labels(entities)
        prompt = self._summarize_prompt(text, entity_labels)
        response_text = self._call_llm(prompt)

        if response_text is None:
            return None

        result = result_pattern.search(response_text)
        record = {"text": text, "labels": entity_labels}

        if result is not None:
            try:
                rules = json.loads(result.group())
                record["status"] = "success"
                record["predicted_rules"] = rules
                return record
            except (json.JSONDecodeError, TypeError):
                pass

        record["status"] = "parse_error"
        record["predicted_rules"] = {}
        return record

    def _summarize_step(self, train_data, output_file, max_workers):
        self.log(f"步骤 1/3: 归纳模式规则，共 {len(train_data)} 条训练样本...")
        valid_items = [item for item in train_data if len(item.get("entities", [])) > 0]

        success_count = 0
        fail_count = 0

        with open(output_file, 'w', encoding='utf-8') as fw:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(self._summarize_one, item): item
                           for item in valid_items}
                for future in tqdm(as_completed(futures), total=len(futures), desc="Summarizing"):
                    result = future.result()
                    if result is not None:
                        fw.write(json.dumps(result, ensure_ascii=False) + "\n")
                        if result["status"] == "success":
                            success_count += 1
                        else:
                            fail_count += 1

        self.log(f"Step 1 完成: 成功 {success_count} 条, 失败 {fail_count} 条", level="SUCCESS")

    # ───────────────────────── Step 2 ─────────────────────────

    def _validate_one(self, line):
        line_json = json.loads(line)
        text = line_json["text"]
        entity_labels = line_json["labels"]
        rules = line_json.get("predicted_rules", {})

        if len(entity_labels) == 0 or not isinstance(rules, dict) or len(rules) == 0:
            return None
        if not self._type_num_equal(entity_labels, rules):
            return None

        prompt = self._validate_prompt(text, json.dumps(rules, ensure_ascii=False))
        response_text = self._call_llm(prompt)

        corres = self._get_correspondings(entity_labels, rules)
        right_rules = []
        wrong_rules = []

        record = {"text": text, "label": entity_labels, "original_rules": rules}

        if response_text is not None:
            result = valid_pattern.search(response_text)
            if result is not None:
                try:
                    predicted_entities = json.loads(result.group())
                    pred_set = set()
                    for ent in predicted_entities:
                        if len(ent) >= 2:
                            pred_set.add((ent[0], ent[1]))

                    for cor in corres:
                        label = cor[0]
                        rule_text = cor[1]
                        entity_type = label[-1]
                        rule_dict = {entity_type: rule_text}

                        if tuple(label) in pred_set:
                            right_rules.append(rule_dict)
                        else:
                            wrong_rules.append(rule_dict)

                    record["right_rules"] = right_rules
                    record["wrong_rules"] = wrong_rules
                    record["status"] = "success"
                    record["predict_labels"] = predicted_entities
                    return record
                except (json.JSONDecodeError, TypeError):
                    pass

        record["right_rules"] = []
        record["wrong_rules"] = []
        record["status"] = "parse_error"
        record["predict_labels"] = []
        return record

    def _validate_step(self, rules_file, output_file, max_workers):
        self.log("步骤 2/3: 自验证规则...")

        with open(rules_file, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]

        self.log(f"读取到 {len(lines)} 条待验证规则")

        right_total = 0
        wrong_total = 0

        with open(output_file, 'w', encoding='utf-8') as fw:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(self._validate_one, line): line
                           for line in lines}
                for future in tqdm(as_completed(futures), total=len(futures), desc="Validating"):
                    result = future.result()
                    if result is not None:
                        fw.write(json.dumps(result, ensure_ascii=False) + "\n")
                        right_total += len(result.get("right_rules", []))
                        wrong_total += len(result.get("wrong_rules", []))

        self.log(f"Step 2 完成: 验证通过 {right_total} 条, 未通过 {wrong_total} 条", level="SUCCESS")

    # ───────────────────────── Step 3 ─────────────────────────

    def _aggregate_step(self, validated_file, output_file, top_k=20):
        self.log(f"步骤 3/3: 聚合规则 (top_k={top_k})...")
        type_rule_counts = {}

        with open(validated_file, 'r', encoding='utf-8') as f:
            for line in f:
                line_json = json.loads(line)
                right_rules = line_json.get("right_rules", [])
                for rule_dict in right_rules:
                    for entity_type, rule_text in rule_dict.items():
                        if entity_type not in type_rule_counts:
                            type_rule_counts[entity_type] = {}
                        if rule_text not in type_rule_counts[entity_type]:
                            type_rule_counts[entity_type][rule_text] = 0
                        type_rule_counts[entity_type][rule_text] += 1

        guidelines = {}
        for entity_type, rule_freq in type_rule_counts.items():
            sorted_rules = sorted(rule_freq.items(), key=lambda x: x[1], reverse=True)
            guidelines[entity_type] = [rule for rule, freq in sorted_rules[:top_k]]
            self.log(f"  - {entity_type}: {len(sorted_rules)} 条唯一规则, "
                     f"选取 top {min(top_k, len(sorted_rules))} 条")

        with open(output_file, 'w', encoding='utf-8') as fw:
            json.dump(guidelines, fw, indent=2, ensure_ascii=False)

        self.log(f"Step 3 完成: 指南写入 {output_file}", level="SUCCESS")
        return guidelines

    # ───────────────────────── Pipeline ─────────────────────────

    def run(self, train_data_path, output_dir, top_k=20, max_workers=8,
            skip_step1=False, skip_step2=False, sample_size=None):
        os.makedirs(output_dir, exist_ok=True)

        summarize_file = os.path.join(output_dir, 'summarized_rules.jsonl')
        validated_file = os.path.join(output_dir, 'validated_rules.jsonl')
        guidelines_file = os.path.join(output_dir, 'guidelines.json')

        self.log(f"加载训练数据: {train_data_path}")
        with open(train_data_path, 'r', encoding='utf-8') as f:
            if train_data_path.endswith('.jsonl'):
                train_data = [json.loads(line) for line in f if line.strip()]
            else:
                train_data = json.load(f)

        self.log(f"加载 {len(train_data)} 条训练样本")

        if sample_size is not None and sample_size < len(train_data):
            random.seed(42)
            train_data = random.sample(train_data, sample_size)
            self.log(f"随机采样 {sample_size} 条样本 (原始 {len(train_data)} 条)")

        entity_types = set()
        for item in train_data:
            for e in item.get("entities", []):
                entity_types.add(e["type"])
        self.log(f"实体类型: {', '.join(sorted(entity_types))}")

        if skip_step1 and os.path.exists(summarize_file):
            self.log(f"跳过 Step 1, 使用已有文件 {summarize_file}")
        else:
            self._summarize_step(train_data, summarize_file, max_workers)

        if skip_step2 and os.path.exists(validated_file):
            self.log(f"跳过 Step 2, 使用已有文件 {validated_file}")
        else:
            self._validate_step(summarize_file, validated_file, max_workers)

        guidelines = self._aggregate_step(validated_file, guidelines_file, top_k)

        self.log("=" * 60, level="SUCCESS")
        self.log("最终指南摘要:", level="SUCCESS")
        for etype, rules in guidelines.items():
            self.log(f"\n  [{etype}] ({len(rules)} 条规则)", level="SUCCESS")
            for i, rule in enumerate(rules[:5], 1):
                self.log(f"    {i}. {rule}", level="SUCCESS")
            if len(rules) > 5:
                self.log(f"    ... 还有 {len(rules) - 5} 条", level="SUCCESS")
        self.log("=" * 60, level="SUCCESS")

        return guidelines
