import requests
import json
import os
import time
from datetime import datetime
from typing import List, Dict, Any

class BioNERPlanningAgent:
    """
    Planning Agent for BioNER.
    Generates entity text + type + confidence + rationale.
    Uses retrieved exemplars (Dex) as in-context guidance.
    """

    def __init__(self, api_key, model="qwen3.5-122b-a10b", base_url=None, max_retries=3, verbose=True):
        """
        api_key: str, API密钥
        model: str, 模型名称
        base_url: str, API端点
        max_retries: int, 最大重试次数
        verbose: bool, 是否输出详细日志
        """
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.max_retries = max_retries
        self.verbose = verbose
    
    def log(self, message, level="INFO"):
        """输出日志信息"""
        if self.verbose:
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp}] [{level}] [PlanningAgent] {message}")
    
    def _call_llm(self, prompt):
        """使用requests直接调用API"""
        url = f"{self.base_url}/chat/completions"
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
            "max_tokens": 10000
        }
        
        for attempt in range(1, self.max_retries + 1):
            try:
                self.log(f"调用API (尝试 {attempt}/{self.max_retries})...")
                
                response = requests.post(url, headers=headers, json=payload, timeout=120)
                
                if response.status_code == 200:
                    result = response.json()
                    content = result['choices'][0]['message']['content'].strip()
                    self.log(f"API调用成功，返回长度: {len(content)} 字符")
                    return content
                else:
                    self.log(f"API返回错误: {response.status_code}", level="WARNING")
                    if response.text:
                        self.log(f"错误详情: {response.text[:200]}", level="WARNING")
                    
            except requests.exceptions.Timeout:
                self.log(f"请求超时 (尝试 {attempt}/{self.max_retries})", level="WARNING")
            except requests.exceptions.ConnectionError as e:
                self.log(f"连接错误: {e}", level="WARNING")
            except Exception as e:
                self.log(f"调用失败: {e}", level="WARNING")
            
            if attempt < self.max_retries:
                wait_time = 2 * attempt
                self.log(f"等待 {wait_time} 秒后重试...")
                time.sleep(wait_time)
        
        raise RuntimeError("LLM request failed after multiple retries.")
    
    def _clean_json_output(self, text: str) -> str:
        """清理LLM输出的JSON文本"""
        text = text.strip()
        
        # 移除markdown代码块标记
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        
        if text.endswith("```"):
            text = text[:-3]
        
        return text.strip()
    
    def score_demonstrations(
        self,
        text: str,
        schema: List[str],
        exemplars: List[Dict],
        trfs: Dict[str, List[str]] = None,
        threshold: float = 2.5,
    ) -> List[Dict]:
        """
        Demonstration Discriminator with Self-Reflection.
        Scores each exemplar's helpfulness for predicting entities in the target sentence.
        Filters out exemplars below the threshold score.
        Inspired by CMAS (WWW 25) paper's demonstration discriminator agent.

        Args:
            text: Target sentence to score exemplars against.
            schema: Entity types.
            exemplars: List of exemplar dicts to score.
            trfs: Optional TRFs for the target sentence (provides richer context for scoring).
            threshold: Minimum helpfulness score (1-5 scale) to keep an exemplar.

        Returns:
            List of exemplars that passed the threshold, each with an added 'helpfulness_score' field.
        """
        if not exemplars:
            self.log("无示例可供判别，跳过", level="WARNING")
            return []

        self.log(f"开始判别示例帮助度 (threshold={threshold})，共 {len(exemplars)} 个示例")

        # Format exemplars for the prompt
        exemplar_entries = []
        for idx, ex in enumerate(exemplars):
            ex_text = ex.get("text", "")
            ex_entities = ex.get("entities", [])
            ex_type = ex.get("type", "")
            ent_str = ", ".join(
                [f"'{e.get('text', '')}' -> {e.get('type', '')}" for e in ex_entities]
            ) if ex_entities else "(none)"
            exemplar_entries.append(
                f"ID: {idx} | Type: {ex_type} | Text: {ex_text} | Entities: {ent_str}"
            )
        exemplars_block = "\n".join(exemplar_entries)

        # Format TRFs if available
        trfs_block = ""
        if trfs:
            trf_parts = []
            for etype, features in trfs.items():
                if features:
                    features_joined = ', '.join(features)
                    trf_parts.append(f"  {etype}: {features_joined}")
            if trf_parts:
                trfs_block = "\n".join(trf_parts)

        trfs_section = f"""
Type-Related Features in target sentence:
{trfs_block}""" if trfs_block else ""

        prompt = f"""\
You are a BioNER Demonstration Discriminator with self-reflection capability.

TASK:
Evaluate how HELPFUL each exemplar sentence would be for recognizing entities in the target sentence.
A helpful exemplar is one that:
- Contains entity types similar to those likely in the target sentence
- Has similar entity boundary patterns
- Provides useful annotation guidance for the target sentence's domain/style

Score each exemplar from 1 to 5:
1 = Completely irrelevant or misleading for the target sentence
2 = Minimal relevance, unlikely to help
3 = Moderately relevant, might provide some guidance
4 = Quite relevant, likely to help with entity recognition
5 = Highly relevant, directly applicable to the target sentence

CRITICAL: Be strict. Do not give high scores to exemplars that only share superficial similarity.

Entity Schema: {schema}
Target Sentence: "{text}"
{trfs_section}

Exemplars to evaluate:
{exemplars_block}

Return ONLY a JSON object mapping exemplar ID to helpfulness score:
{{
    "0": 3,
    "1": 5,
    ...
}}
"""
        self.log(f"Discriminator prompt 长度: {len(prompt)} 字符")

        try:
            llm_output = self._call_llm(prompt)
            llm_output = self._clean_json_output(llm_output)
            self.log(f"Discriminator LLM 输出: {llm_output[:200]}..." if len(llm_output) > 200 else f"Discriminator LLM 输出: {llm_output}")

            scores = json.loads(llm_output)

            # Apply scores to exemplars
            scored_exemplars = []
            kept_count = 0
            filtered_count = 0
            all_scored = []
            for idx, ex in enumerate(exemplars):
                score = scores.get(str(idx), scores.get(idx, 3))  # Default to 3 if missing
                try:
                    score = float(score)
                except (ValueError, TypeError):
                    score = 3.0
                score = max(1.0, min(5.0, score))  # Clamp to [1, 5]

                ex_with_score = dict(ex)
                ex_with_score['helpfulness_score'] = score
                all_scored.append((score, idx, ex_with_score))

                if score >= threshold:
                    scored_exemplars.append(ex_with_score)
                    kept_count += 1
                else:
                    filtered_count += 1
                    self.log(f"  过滤示例 ID={idx}, score={score:.1f} < {threshold}", level="WARNING")

            # Fallback: if all exemplars were filtered, keep the top 2 by score
            if not scored_exemplars and all_scored:
                all_scored.sort(key=lambda x: x[0], reverse=True)
                fallback_count = min(2, len(all_scored))
                for score, idx, ex in all_scored[:fallback_count]:
                    scored_exemplars.append(ex)
                    kept_count += 1
                    filtered_count -= 1
                    self.log(f"  保底保留示例 ID={idx}, score={score:.1f} (threshold={threshold})", level="WARNING")

            self.log(
                f"判别完成: 保留 {kept_count}/{len(exemplars)} 个示例, 过滤 {filtered_count} 个",
                level="SUCCESS"
            )
            return scored_exemplars

        except json.JSONDecodeError as e:
            self.log(f"Discriminator JSON 解析失败: {e}, 返回全部示例", level="ERROR")
            return exemplars
        except Exception as e:
            self.log(f"Discriminator 失败: {e}, 返回全部示例", level="ERROR")
            return exemplars


    def extract_trfs(self, text: str, schema: List[str], exemplars: List[Dict] = None) -> Dict[str, List[str]]:
        """
        Extract Type-Related Features (TRFs) from the input text.
        TRFs are tokens/phrases strongly associated with specific entity types
        that help capture contextual correlations surrounding entities.
        Inspired by CMAS (WWW 25) paper TRF extractor agent.

        Args:
            text: Input sentence to extract TRFs from.
            schema: Entity types to extract TRFs for.
            exemplars: Optional exemplars to provide as ICL guidance.

        Returns:
            Dict mapping entity type to list of TRF tokens/phrases found in text.
        """
        self.log(f"开始抽取类型相关特征 (TRFs)，目标类型: {schema}")

        exemplar_context = ""
        if exemplars:
            flat = exemplars
            if isinstance(exemplars, dict):
                flat = []
                for etype, elist in exemplars.items():
                    for ex in elist:
                        flat.append({"type": etype, **ex})
            exemplar_lines = []
            for ex in flat[:5]:
                ex_text = ex.get("text", "")
                ex_entities = ex.get("entities", [])
                trf_note = ""
                if ex_entities:
                    trf_note = f"  (entities: {ex_entities})"
                exemplar_lines.append(f"Text: {ex_text}{trf_note}")
            exemplar_context = "\n".join(exemplar_lines)


        ref_exemplars_str = ("Reference Exemplars:\n" + exemplar_context) if exemplar_context else ""
        prompt = f"""\
You are a BioNER TRF (Type-Related Feature) Extraction expert.

TASK:
Given a target sentence, identify tokens or short phrases that are STRONGLY ASSOCIATED with each given entity type.
TRFs are contextual clues (not the entities themselves) that indicate the presence of a certain type of entity.

For example, in "Atessis was a member of teams which set school record...", the words "member" and "teams" are TRFs for the Person type because they signal a person entity.

RULES:
1. Extract TRFs for EACH entity type in the schema that is relevant to the target sentence.
2. TRFs should be tokens or short phrases (1-4 words) that appear in the target sentence.
3. Do NOT extract the entity mentions themselves -- extract the SURROUNDING CONTEXTUAL CLUES.
4. If an entity type has no relevant TRFs in the sentence, return an empty list for that type.
5. TRFs must be exact spans from the target sentence.

Return ONLY a JSON object in this format:
{{
    "TYPE1": ["trf1", "trf2"],
    "TYPE2": ["trf3"],
    ...
}}

---

Entity Schema:
{schema}

{ref_exemplars_str}

---

Target Sentence:
"{text}"

---

Return the JSON object of TRFs per entity type:
"""
        self.log(f"TRF prompt 长度: {len(prompt)} 字符")

        try:
            llm_output = self._call_llm(prompt)
            llm_output = self._clean_json_output(llm_output)
            self.log(f"TRF LLM 输出: {llm_output[:200]}..." if len(llm_output) > 200 else f"TRF LLM 输出: {llm_output}")

            trf_result = json.loads(llm_output)

            validated_trfs = {}
            for ent_type, trf_list in trf_result.items():
                if isinstance(trf_list, list):
                    validated_trfs[ent_type.upper()] = [str(t) for t in trf_list if isinstance(t, str) and t.strip()]
                elif isinstance(trf_list, str):
                    validated_trfs[ent_type.upper()] = [trf_list.strip()] if trf_list.strip() else []

            total_trfs = sum(len(v) for v in validated_trfs.values())
            self.log(f"TRF 抽取完成，共发现 {total_trfs} 个特征", level="SUCCESS")
            return validated_trfs

        except json.JSONDecodeError as e:
            self.log(f"TRF JSON 解析失败: {e}", level="ERROR")
            return {t: [] for t in schema}
        except Exception as e:
            self.log(f"TRF 抽取失败: {e}", level="ERROR")
            return {t: [] for t in schema}


    def plan(self, text, schema, exemplars=None, trfs=None, guidelines=None):
        """
        text: input sentence
        schema: entity types (list of strings)
        exemplars: output from Retrieval Agent (list of exemplar dicts)
        trfs: optional Dict[str, List[str]] of TRFs per entity type
        guidelines: optional Dict[str, str] of annotation rules per entity type
                   e.g. {"GENE": "gene symbol abbreviation; enzyme full name"}

        Returns:
        list: 检测到的实体列表，每个实体包含 text, type, confidence, rationale
        """
        self.log(f"开始规划实体识别")
        self.log(f"输入文本: {text[:100]}..." if len(text) > 100 else f"输入文本: {text}")
        self.log(f"实体类型: {schema}")
        if exemplars is None:
            exemplars = []
        self.log(f"参考示例数量: {len(exemplars) if exemplars else 0}")
        if trfs:
            total_trfs = sum(len(v) for v in trfs.values())
            self.log(f"TRF 特征数量: {total_trfs}")
        if guidelines:
            self.log(f"注入 Guidelines: {list(guidelines.keys())}")

        # Format guidelines for prompt injection
        guidelines_str = ""
        if guidelines:
            gl_parts = []
            for etype, rule in guidelines.items():
                gl_parts.append(f"  {etype}: {rule}")
            guidelines_str = "\n".join(gl_parts)

        # Format TRFs for prompt injection
        trfs_str = ""
        if trfs:
            trf_parts = []
            for etype, features in trfs.items():
                if features:
                    features_joined = ', '.join(features)
                    trf_parts.append(f"  {etype}: {features_joined}")
            trfs_str = "\n".join(trf_parts) if trf_parts else "(none detected)"

        # 处理exemplars格式
        if exemplars and isinstance(exemplars, dict):
            # 如果是字典格式（按类型分组），转换为列表
            if all(isinstance(v, list) for v in exemplars.values()):
                flat_exemplars = []
                for entity_type, ex_list in exemplars.items():
                    for ex in ex_list:
                        ex_copy = ex.copy() if isinstance(ex, dict) else {"text": str(ex)}
                        if 'type' not in ex_copy:
                            ex_copy['type'] = entity_type
                        flat_exemplars.append(ex_copy)
                exemplars = flat_exemplars
                self.log(f"转换exemplars格式: {len(exemplars)} 个示例")

        optional_sections = []
        if exemplars:
            exemplar_payload = exemplars[:5] if isinstance(exemplars, list) else exemplars
            optional_sections.append(f"""Retrieved Exemplars (optional reference; follow their boundary style when relevant):
{json.dumps(exemplar_payload, indent=2, ensure_ascii=False)}
""")

        if trfs_str and trfs_str != "(none detected)":
            optional_sections.append(f"""Type-Related Features (optional contextual clues):
{trfs_str}
""")

        optional_sections_str = "\n---\n\n".join(optional_sections)
        if optional_sections_str:
            optional_sections_str = f"\n---\n\n{optional_sections_str}\n"
        
        prompt = f"""\
You are a BioNER Planning Agent for candidate entity generation.

TASK:
Given the input text, entity schema, and annotation guidelines generated by the upstream guideline agent, identify all entity mentions that should be annotated.

Use the Annotation Guidelines as the authoritative source for entity type decisions. If the guideline does not fully specify boundary behavior, apply the boundary rules below to choose dataset-consistent complete spans.

Your goal is to maximize end-to-end F1: keep high recall, but avoid unsupported abbreviation-only false positives.

CRITICAL INSTRUCTIONS:
1. Use only entity types from the schema.
2. Do not invent new entity-type definitions beyond the schema and Annotation Guidelines.
3. Include plausible mentions when they are supported by the guideline or by clear local biomedical context.
4. Do not omit a mention only because it is uncommon, abbreviated, descriptive, or appears in contrastive/negative context, unless the guideline says to exclude it.
5. Do not annotate a token only because it is uppercase, alphanumeric, or abbreviation-like. It must be supported by the guideline or local context.
6. Provide confidence between 0 and 1.
7. Provide a short rationale grounded in the guideline, context, or boundary decision.

BOUNDARY SELECTION RULES:
- First identify the core entity mention, then expand it to the complete contiguous mention span when adjacent modifiers, descriptors, head words, suffixes, organism/species qualifiers, technical qualifiers, or parenthetical fragments are part of the same biomedical noun phrase.
- Prefer the complete annotated span over a shortened core span.
- If a longer span and a shorter core span refer to the same mention, output only the longer complete span.
- Do not output multiple competing boundary variants for the same mention unless the guideline explicitly allows nested or overlapping annotations.
- If coordination or parenthetical text contains multiple independent mentions, annotate each independent mention separately.
- If the boundary is ambiguous, choose the span that preserves the full biomedical mention in the input text.

FALSE POSITIVE CONTROL:
- Be cautious with isolated abbreviations that have no entity-supporting context.
- Exclude terms that are more likely to be chemicals, treatments, measurement indicators, experimental groups, organizations, diseases, or generic biological processes unless the guideline clearly includes them for the target entity type.
- Do not annotate broad contextual clues as entities unless the guideline says they are valid entity mentions.

Return ONLY JSON list, no extra text:

[
  {{
    "text": "...",
    "type": "...",
    "confidence": 0.0-1.0,
    "rationale": "..."
  }}
]

Entity Schema:
{schema}

Annotation Guidelines:
{guidelines_str if guidelines_str else "(no guidelines provided)"}
{optional_sections_str}

Input Text:
"{text}"
-
Internally perform:
1. candidate recall,
2. boundary expansion,
3. false-positive filtering,
4. deduplication.

Now return the final candidate entity list:
"""
        
        self.log(f"提示词构建完成，长度: {len(prompt)} 字符")
        
        try:
            llm_output = self._call_llm(prompt)
            
            # 清理输出
            llm_output = self._clean_json_output(llm_output)
            self.log(f"LLM原始输出: {llm_output[:200]}..." if len(llm_output) > 200 else f"LLM原始输出: {llm_output}")
            
            # 解析JSON
            entities = json.loads(llm_output)
            
            # 确保是列表
            if isinstance(entities, dict):
                entities = [entities]
            
            # 验证每个实体的格式
            valid_entities = []
            for i, entity in enumerate(entities):
                if self._validate_entity(entity, text):
                    valid_entities.append(entity)
                    self.log(f"实体 {i+1}: '{entity.get('text', 'N/A')}' -> {entity.get('type', 'N/A')} (置信度: {entity.get('confidence', 'N/A')})")
                else:
                    self.log(f"实体 {i+1} 格式无效，已跳过", level="WARNING")
            
            self.log(f"成功识别 {len(valid_entities)} 个实体", level="SUCCESS")
            return valid_entities
            
        except json.JSONDecodeError as e:
            self.log(f"JSON解析失败: {e}", level="ERROR")
            self.log(f"原始输出: {llm_output}", level="ERROR")
            
            # 尝试从输出中提取JSON
            try:
                start_idx = llm_output.find('[')
                end_idx = llm_output.rfind(']') + 1
                if start_idx != -1 and end_idx > start_idx:
                    json_str = llm_output[start_idx:end_idx]
                    entities = json.loads(json_str)
                    self.log("成功从文本中提取JSON", level="SUCCESS")
                    return entities if isinstance(entities, list) else [entities]
            except:
                pass
            
            return []
            
        except Exception as e:
            self.log(f"规划失败: {e}", level="ERROR")
            return []
    
    def _validate_entity(self, entity: Dict, text: str) -> bool:
        """验证实体格式是否正确"""
        # 检查必需字段
        required_fields = ['text', 'type', 'confidence', 'rationale']
        for field in required_fields:
            if field not in entity:
                self.log(f"缺少必需字段: {field}", level="WARNING")
                return False
        
        # 验证confidence范围
        if not 0 <= entity['confidence'] <= 1:
            self.log(f"置信度超出范围: {entity['confidence']}", level="WARNING")
            return False
        
        # 验证实体文本是否在输入文本中
        entity_text = entity['text']
        if entity_text.lower() not in text.lower():
            self.log(f"实体文本不在输入文本中: '{entity_text}'", level="WARNING")
            return False
        
        return True
    
    def plan_batch(self, texts: List[str], schema: List[str], exemplars: List[Dict]) -> List[List[Dict]]:
        """
        批量处理多个文本
        
        Args:
            texts: list of input sentences
            schema: entity types
            exemplars: output from Retrieval Agent
        
        Returns:
            list: 每个文本对应的实体列表
        """
        self.log(f"批量规划开始，共 {len(texts)} 个文本")
        
        results = []
        for idx, text in enumerate(texts, 1):
            self.log(f"\n处理第 {idx}/{len(texts)} 个文本")
            entities = self.plan(text, schema, exemplars)
            results.append(entities)
            
            # 添加延迟，避免API限流
            if idx < len(texts):
                time.sleep(0.5)
        
        self.log(f"批量规划完成，成功处理 {len(results)} 个文本")
        return results


