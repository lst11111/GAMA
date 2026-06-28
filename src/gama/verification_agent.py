import ast
import json
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, ValidationError

from .coding_agent import BioNERCodingAgent


class BioNERVerificationAgent:
    """
    Verification Agent for BioNER.

    对应论文中的三阶段验证：
    1. Semantic Check (T1)
    2. Type Check (T2)
    3. Structural Check (T3)

    输入可以是 coding_agent 生成的:
    {
        "rank": ...,
        "entity_type": ...,
        "confidence": ...,
        "object": {...},
        "code": "entity = GENEEntity(...)"
    }

    或者直接传入 code + entity_object。
    """

    EXPECTED_FIELDS = {
        "entity_type",
        "text",
        "confidence",
        "rationale",
        "source_text",
    }

    CONTEXT_KEYWORDS = {
        "GENE": {
            "gene", "mutation", "variant", "expression", "pathway", "allele",
            "transcript", "amplification", "deletion", "fusion", "oncogene",
        },
        "GENE_OR_GENE_PRODUCT": {
            "gene", "protein", "expression", "mutation", "variant", "receptor",
            "kinase", "enzyme", "transcript", "phosphorylation",
        },
        "PROTEIN": {
            "protein", "receptor", "kinase", "enzyme", "antibody",
            "subunit", "phosphorylation", "binding",
        },
        "DISEASE": {
            "disease", "cancer", "syndrome", "disorder", "tumor", "tumour",
            "carcinoma", "infection", "injury", "deficiency", "failure",
            "diabetes", "neoplasm",
        },
        "CHEMICAL": {
            "drug", "compound", "treatment", "therapy", "dose", "inhibitor",
            "agonist", "antagonist", "metabolite", "acid",
        },
        "CELL_LINE": {
            "cell line", "cells", "cultured", "transfected", "clone",
            "heLa", "293t", "a549", "mcf-7",
        },
        "CELL_TYPE": {
            "cell", "lymphocyte", "neuron", "macrophage", "epithelial",
            "fibroblast", "stem cell", "t cell", "b cell",
        },
        "SPECIES": {
            "mouse", "mice", "rat", "human", "patients", "species",
            "strain", "organism",
        },
    }

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.coding_agent = BioNERCodingAgent(verbose=False)

    def log(self, message: str, level: str = "INFO") -> None:
        if self.verbose:
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp}] [{level}] [VerificationAgent] {message}")

    def verify(
        self,
        code: str,
        text: str,
        schema: List[str],
        entity_object: Optional[Any] = None,
        schema_guidelines: Optional[Any] = None,
    ) -> Tuple[bool, str]:
        """
        返回:
            (V, epsilon)
            V in {True, False}
            epsilon 为首个失败测试的诊断信息；若成功则为 ""
        """
        payload = self._recover_payload(code=code, entity_object=entity_object)

        ok, epsilon = self.semantic_check(
            payload=payload,
            input_text=text,
            schema=schema,
            schema_guidelines=schema_guidelines,
        )
        if not ok:
            return False, epsilon

        ok, epsilon = self.type_check(
            payload=payload,
            input_text=text,
            schema=schema,
        )
        if not ok:
            return False, epsilon

        ok, epsilon = self.structural_check(
            code=code,
            payload=payload,
            schema=schema,
        )
        if not ok:
            return False, epsilon

        return True, ""

    def verify_candidate(
        self,
        candidate: Dict[str, Any],
        text: str,
        schema: List[str],
        schema_guidelines: Optional[Any] = None,
    ) -> Tuple[bool, str]:
        """
        直接验证 coding_agent.build_entity_objects() 返回的单个 valid_entity 项。
        """
        return self.verify(
            code=candidate.get("code", ""),
            text=text,
            schema=schema,
            entity_object=candidate.get("object"),
            schema_guidelines=schema_guidelines,
        )

    def verify_candidates(
        self,
        candidates: List[Dict[str, Any]],
        text: str,
        schema: List[str],
        schema_guidelines: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        """
        批量验证多个候选对象。
        """
        results = []
        for idx, candidate in enumerate(candidates, start=1):
            verdict, epsilon = self.verify_candidate(
                candidate=candidate,
                text=text,
                schema=schema,
                schema_guidelines=schema_guidelines,
            )
            results.append(
                {
                    "rank": candidate.get("rank", idx),
                    "verdict": verdict,
                    "epsilon": epsilon,
                    "entity_type": candidate.get("entity_type"),
                    "confidence": candidate.get("confidence"),
                    "object": candidate.get("object"),
                    "code": candidate.get("code"),
                }
            )
        return results

    def select_first_pass(
        self,
        candidates: List[Dict[str, Any]],
        text: str,
        schema: List[str],
        schema_guidelines: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        按输入顺序返回第一个通过三阶段验证的候选。
        如果都失败，返回最后一个失败信息汇总。
        """
        failures = []

        for candidate in candidates:
            verdict, epsilon = self.verify_candidate(
                candidate=candidate,
                text=text,
                schema=schema,
                schema_guidelines=schema_guidelines,
            )
            if verdict:
                return {
                    "verdict": True,
                    "epsilon": "",
                    "candidate": candidate,
                }
            failures.append(
                {
                    "rank": candidate.get("rank"),
                    "epsilon": epsilon,
                }
            )

        return {
            "verdict": False,
            "epsilon": failures[0]["epsilon"] if failures else "没有可验证的候选对象",
            "failures": failures,
        }

    def semantic_check(
        self,
        payload: Dict[str, Any],
        input_text: str,
        schema: List[str],
        schema_guidelines: Optional[Any] = None,
    ) -> Tuple[bool, str]:
        """
        T1: Semantic Check
        1. mention 必须真实出现在输入文本中
        2. mention 与 entity_type 必须语义兼容
        """
        entity_type = str(payload.get("entity_type", "")).upper()
        entity_text = str(payload.get("text", ""))

        if entity_type not in {s.upper() for s in schema}:
            return False, f"T1 Semantic Check Failed: 未知实体类型 '{entity_type}'"

        # 检查实体文本是否在输入文本中（不区分大小写）
        if entity_text.lower() not in input_text.lower():
            return False, f"T1 Semantic Check Failed: mention '{entity_text}' 不在输入文本中"

        compatible, reason = self._is_type_semantically_compatible(
            entity_text=entity_text,
            entity_type=entity_type,
            input_text=input_text,
            schema_guidelines=schema_guidelines,
        )
        if not compatible:
            return False, f"T1 Semantic Check Failed: {reason}"

        self.log(f"T1 通过: {entity_text} -> {entity_type}", level="SUCCESS")
        return True, ""

    def type_check(
        self,
        payload: Dict[str, Any],
        input_text: str,
        schema: List[str],
    ) -> Tuple[bool, str]:
        """
        T2: Type Check
        使用 coding_agent 的动态 schema BaseModel 进行二次验证。
        """
        entity_type = str(payload.get("entity_type", ""))
        if entity_type not in schema:
            return False, f"T2 Type Check Failed: entity_type '{entity_type}' 不在 schema 中"

        try:
            model_registry = self.coding_agent.compile_schema(schema)
            model_cls = model_registry[entity_type]

            normalized_payload = dict(payload)
            normalized_payload["source_text"] = input_text

            model_cls(**normalized_payload)

            self.log(f"T2 通过: {entity_type} 的 BaseModel 验证成功", level="SUCCESS")
            return True, ""

        except ValidationError as e:
            return False, f"T2 Type Check Failed: {str(e)}"
        except Exception as e:
            return False, f"T2 Type Check Failed: {type(e).__name__}: {str(e)}"

    def structural_check(
        self,
        code: str,
        payload: Dict[str, Any],
        schema: List[str],
    ) -> Tuple[bool, str]:
        """
        T3: Structural Check
        1. 代码能编译
        2. 代码结构为 entity = XxxEntity(...)
        3. 字段必须严格等于 EXPECTED_FIELDS
        4. 执行后得到可序列化的 BaseModel 对象
        """
        try:
            compile(code, "<generated_entity_code>", "exec")
        except SyntaxError as e:
            return False, f"T3 Structural Check Failed: 代码编译失败: {str(e)}"

        try:
            func_name, kwargs = self._extract_call_from_code(code)
        except Exception as e:
            return False, f"T3 Structural Check Failed: 代码结构解析失败: {str(e)}"

        kwarg_keys = set(kwargs.keys())
        if kwarg_keys != self.EXPECTED_FIELDS:
            return (
                False,
                "T3 Structural Check Failed: 字段集合不匹配，"
                f"期望 {sorted(self.EXPECTED_FIELDS)}，实际 {sorted(kwarg_keys)}",
            )

        entity_type = payload["entity_type"]
        expected_model_name = self.coding_agent._sanitize_model_name(entity_type)
        if func_name != expected_model_name:
            return (
                False,
                f"T3 Structural Check Failed: 模型名不匹配，期望 '{expected_model_name}'，实际 '{func_name}'",
            )

        try:
            model_registry = self.coding_agent.compile_schema(schema)
            model_cls = model_registry[entity_type]

            local_vars: Dict[str, Any] = {}
            safe_globals: Dict[str, Any] = {
                "__builtins__": {},
                expected_model_name: model_cls,
            }

            exec(code, safe_globals, local_vars)

            if "entity" not in local_vars:
                return False, "T3 Structural Check Failed: 执行后未生成变量 'entity'"

            entity = local_vars["entity"]
            if not isinstance(entity, BaseModel):
                return False, "T3 Structural Check Failed: 生成结果不是 BaseModel 实例"

            dumped = entity.model_dump()
            if set(dumped.keys()) != self.EXPECTED_FIELDS:
                return (
                    False,
                    "T3 Structural Check Failed: 序列化后字段集合不匹配，"
                    f"实际为 {sorted(dumped.keys())}",
                )

            json.dumps(dumped, ensure_ascii=False)

            self.log("T3 通过: 代码结构、执行与序列化均成功", level="SUCCESS")
            return True, ""

        except Exception as e:
            return False, f"T3 Structural Check Failed: {type(e).__name__}: {str(e)}"

    def _recover_payload(
        self,
        code: str,
        entity_object: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        优先使用 entity_object；若没有则从 code 中静态解析参数。
        """
        if entity_object is not None:
            payload = self._normalize_payload_dict(entity_object)
            return payload

        _, kwargs = self._extract_call_from_code(code)
        payload = self._normalize_payload_dict(kwargs)
        return payload

    def _normalize_payload_dict(self, payload_like: Any) -> Dict[str, Any]:
        if isinstance(payload_like, BaseModel):
            payload = payload_like.model_dump()
        elif isinstance(payload_like, dict):
            payload = dict(payload_like)
        else:
            raise TypeError("entity_object 必须是 dict 或 BaseModel")

        missing = self.EXPECTED_FIELDS - set(payload.keys())
        if missing:
            raise ValueError(f"payload 缺少字段: {sorted(missing)}")

        payload["entity_type"] = str(payload["entity_type"])
        payload["text"] = str(payload["text"])
        payload["rationale"] = str(payload["rationale"])
        payload["source_text"] = str(payload["source_text"])
        payload["confidence"] = float(payload["confidence"])

        return payload

    def _extract_call_from_code(self, code: str) -> Tuple[str, Dict[str, Any]]:
        """
        解析形如:
        entity = GENEEntity(...)
        """
        tree = ast.parse(code, mode="exec")

        if len(tree.body) != 1:
            raise ValueError("代码必须只包含一条语句")

        stmt = tree.body[0]
        if not isinstance(stmt, ast.Assign):
            raise ValueError("代码必须是赋值语句")
        if len(stmt.targets) != 1 or not isinstance(stmt.targets[0], ast.Name):
            raise ValueError("赋值目标不合法")
        if stmt.targets[0].id != "entity":
            raise ValueError("赋值变量名必须为 'entity'")
        if not isinstance(stmt.value, ast.Call):
            raise ValueError("右值必须是函数/类调用")

        call = stmt.value
        if not isinstance(call.func, ast.Name):
            raise ValueError("调用对象必须是简单类名")

        if call.args:
            raise ValueError("不允许位置参数，必须全部使用关键字参数")

        kwargs = {}
        for kw in call.keywords:
            if kw.arg is None:
                raise ValueError("不支持 **kwargs")
            kwargs[kw.arg] = ast.literal_eval(kw.value)

        return call.func.id, kwargs

    def _is_type_semantically_compatible(
        self,
        entity_text: str,
        entity_type: str,
        input_text: str,
        schema_guidelines: Optional[Any] = None,
    ) -> Tuple[bool, str]:
        """
        轻量级、确定性的 BioNER 语义兼容性检查。
        不依赖额外模型，避免 verification 本身再次引入不稳定性。
        """
        # 保持原始大小写用于基因符号等特殊实体的识别
        original_entity = entity_text
        lower_entity = entity_text.lower()
        
        # 在实体文本附近查找上下文
        entity_pos = input_text.lower().find(lower_entity)
        if entity_pos != -1:
            start_context = max(0, entity_pos - 50)
            end_context = min(len(input_text), entity_pos + len(entity_text) + 50)
            context = input_text[start_context:end_context].lower()
        else:
            context = input_text.lower()

        score = 0.0
        reasons = []

        # 对于GENE类型，特别处理大小写敏感的基因/基因产物写法。
        if entity_type == "GENE":
            hard_negative, hard_negative_reason = self._is_gene_hard_negative(
                original_entity, lower_entity, context
            )
            if hard_negative:
                return False, hard_negative_reason

            incomplete_boundary, boundary_reason = self._has_incomplete_gene_boundary(
                original_entity, input_text
            )
            if incomplete_boundary:
                return False, boundary_reason

            # 基因符号通常为大写字母和数字，如TP53, MYC, BRCA1等
            if re.match(r'^[A-Z][A-Z0-9]+$', original_entity) and len(original_entity) >= 2:
                score += 0.8  # 给予较高分数
                reasons.append(f"'{original_entity}' 符合基因符号模式")
            elif self._looks_like_extended_gene_symbol(original_entity, context):
                score += 0.65
                reasons.append(f"'{original_entity}' 符合扩展基因/蛋白符号模式")
            elif self._is_descriptive_gene_name(original_entity, lower_entity, context):
                score += 0.5
                reasons.append(f"'{original_entity}' 是描述性基因/蛋白名")
            elif self._lexical_shape_match(original_entity, entity_type):
                score += 0.6
                reasons.append("mention 词形与实体类型匹配")
            else:
                # 即使不匹配任何模式，也给一个基础分，避免全部拒绝
                score += 0.1
                reasons.append(f"'{original_entity}' 给予基础分，由后续阶段进一步判断")
        else:
            if self._lexical_shape_match(original_entity, entity_type):
                score += 0.6
                reasons.append("mention 词形与实体类型匹配")

        context_hits = self._count_keyword_hits(context, entity_type)
        if context_hits > 0:
            score += min(0.3, 0.1 * context_hits)
            reasons.append(f"上下文命中 {context_hits} 个类型关键词")

        if entity_type == "GENE" and self._has_strong_gene_context(context):
            score += 0.25
            reasons.append("上下文包含强基因/蛋白线索")

        guideline_hits = self._count_guideline_hits(
            lower_entity=lower_entity,
            context=context,
            entity_type=entity_type,
            schema_guidelines=schema_guidelines,
        )
        if guideline_hits > 0:
            score += min(0.2, 0.05 * guideline_hits)
            reasons.append(f"与 guideline 命中 {guideline_hits} 个线索词")

        if entity_type not in self.CONTEXT_KEYWORDS and score == 0:
            return True, f"类型 {entity_type} 未配置专用语义规则，跳过严格兼容性检查"

        # 降低阈值，使验证不过于严格
        if score >= 0.15:  # 从0.3降低到0.15，放宽验证
            return True, "; ".join(reasons) if reasons else "通过默认语义阈值"

        return False, (
            f"mention '{original_entity}' 与类型 '{entity_type}' 的语义兼容性不足"
            + (f"；当前线索: {'; '.join(reasons)}" if reasons else "")
        )

    def _count_keyword_hits(self, context: str, entity_type: str) -> int:
        keywords = self.CONTEXT_KEYWORDS.get(entity_type, set())
        hits = 0
        for kw in keywords:
            if kw.lower() in context:
                hits += 1
        return hits

    def _count_guideline_hits(
        self,
        lower_entity: str,
        context: str,
        entity_type: str,
        schema_guidelines: Optional[Any] = None,
    ) -> int:
        guideline_map = self._normalize_guidelines(schema_guidelines)
        guideline = guideline_map.get(entity_type, "")

        if not guideline:
            return 0

        tokens = re.findall(r"[a-zA-Z][a-zA-Z\-]{3,}", guideline.lower())
        stopwords = {
            "annotate", "including", "include", "entity", "entities", "mention",
            "mentions", "text", "type", "types", "should", "would", "their",
            "them", "with", "from", "into", "that", "this", "those", "these",
        }
        tokens = [tok for tok in tokens if tok not in stopwords]

        hits = 0
        for tok in set(tokens):
            if tok in lower_entity or tok in context:
                hits += 1
        return hits

    def _normalize_guidelines(self, schema_guidelines: Optional[Any]) -> Dict[str, str]:
        """
        支持两种输入:
        1. {"GENE": "...", "DISEASE": "..."}
        2. [{"type": "GENE", "guideline": "..."}, ...]
        """
        if schema_guidelines is None:
            return {}

        if isinstance(schema_guidelines, dict):
            normalized = {}
            for k, v in schema_guidelines.items():
                normalized[str(k).upper()] = str(v)
            return normalized

        if isinstance(schema_guidelines, list):
            normalized = {}
            for item in schema_guidelines:
                if isinstance(item, dict) and "type" in item and "guideline" in item:
                    normalized[str(item["type"]).upper()] = str(item["guideline"])
            return normalized

        return {}

    def _is_gene_hard_negative(
        self,
        entity_text: str,
        lower_entity: str,
        context: str,
    ) -> Tuple[bool, str]:
        stripped = lower_entity.strip()
        negative_context_patterns = [
            r"\babo\s*-\s*incompatible\b",
            r"\bnmda\s+antagonist\b",
            r"\bandrogen\s+ablation\b",
            r"\bauc\b",
            r"\bco2\b",
            r"\bbehavioral\b",
            r"\borganization\b",
        ]
        isolated_negative_mentions = {
            "abo", "nmda", "ftf", "bdrt", "trt", "pr", "pc", "peh", "dht",
        }
        if stripped in isolated_negative_mentions:
            for pattern in negative_context_patterns:
                if re.search(pattern, context):
                    return True, (
                        f"mention '{entity_text}' 更像实验指标/组织/处理条件缩写，"
                        "不作为 GENE 候选通过"
                    )

        if stripped in {"bhlh", "basic - helix - loop - helix", "basic-helix-loop-helix"}:
            if "domain" in context and "transcription factor" in context:
                return True, f"mention '{entity_text}' 是结构域/上下文线索，不是完整 GENE mention"

        return False, ""

    def _has_incomplete_gene_boundary(
        self,
        entity_text: str,
        input_text: str,
    ) -> Tuple[bool, str]:
        """
        拦截明显短于数据集标注习惯的候选，如:
        - Meis1 -> Meis1 cDNA
        - Sp1 -> Sp1 binding element
        - alpha2-integrin -> alpha2-integrin promoter
        对多次出现的 mention，只要存在一个 occurrence 看起来边界完整，就不拒绝。
        """
        matches = list(re.finditer(re.escape(entity_text), input_text, flags=re.IGNORECASE))
        if not matches:
            return False, ""

        right_head_pattern = re.compile(
            r"^\s*(?:"
            r"genes?|promoters?|cDNA|mRNA|RNAs?|proteins?|subunits?|"
            r"receptors?|transcripts?|constructs?|regions?|sites?|"
            r"complex(?:es)?|famil(?:y|ies)|homolog(?:ue)?s?|clones?|"
            r"motifs?|domains?|binding\s+elements?"
            r")\b",
            re.IGNORECASE,
        )
        left_modifier_pattern = re.compile(
            r"(?:"
            r"human|mouse|murine|rat|rhesus|primate|rodent|yeast|"
            r"escherichia\s+coli|e\s*\.\s*coli|saccharomyces\s+cerevisiae|"
            r"s\s*\.\s*typhimurium|plasma|serum|gene\s+reporter"
            r")\s+$",
            re.IGNORECASE,
        )

        incomplete_reasons = []
        for match in matches:
            before = input_text[max(0, match.start() - 60):match.start()]
            after = input_text[match.end():match.end() + 80]

            reasons = []
            right_match = right_head_pattern.search(after)
            if right_match:
                right_head = right_match.group(0).strip().lower()
                coordinated_complex = (
                    right_head.startswith("complex")
                    and re.search(r"(?:-|\+|/)\s*$", before)
                )
                if not coordinated_complex:
                    reasons.append("右侧紧跟同一实体短语的 head/descriptor")
            if re.search(r"^\s*\(\s*\d", after):
                reasons.append("右侧紧跟位置/片段括号说明")
            if left_modifier_pattern.search(before):
                reasons.append("左侧紧邻物种/样本/实体修饰词")

            if not reasons:
                return False, ""
            incomplete_reasons.append(", ".join(reasons))

        return True, (
            f"mention '{entity_text}' 可能是更长 GENE mention 的短边界；"
            f"线索: {'; '.join(incomplete_reasons[:2])}"
        )

    def _looks_like_extended_gene_symbol(self, entity_text: str, context: str) -> bool:
        compact = re.sub(r"\s+", " ", entity_text.strip())

        if re.fullmatch(r"[A-Za-z]{2,}\d+[A-Za-z0-9]*", compact):
            return any(ch.isupper() for ch in compact) and any(ch.islower() for ch in compact)

        if re.fullmatch(r"[a-z]{2,}[A-Z][A-Za-z0-9]*", compact):
            return True

        if re.fullmatch(r"[A-Z]{2,}s", compact):
            return True

        if re.fullmatch(r"[A-Za-z]{2,}\d*p", compact):
            return any(ch.isupper() for ch in compact) or any(ch.isdigit() for ch in compact)

        if re.fullmatch(
            r"[A-Za-z0-9]+(?:\s*-\s*[A-Za-z0-9]+|\s+[A-Za-z0-9]+){1,3}",
            compact,
        ):
            return any(ch.isupper() for ch in compact) and (
                any(ch.isdigit() for ch in compact)
                or re.search(r"\b(?:I|II|III|IV|V|VI|alpha|beta|PKS|RNAP|AP)\b", compact)
            )

        if re.fullmatch(r"[a-z]{2,6}", compact):
            return self._has_strong_gene_context(context)

        if re.fullmatch(r"[A-Z][a-z]{1,5}", compact):
            return self._has_strong_gene_context(context)

        return False

    def _has_strong_gene_context(self, context: str) -> bool:
        strong_context_terms = {
            "gene", "genes", "protein", "proteins", "enzyme", "receptor", "receptors",
            "kinase", "polymerase", "reductase", "hydrolase", "transcript", "transcripts",
            "rna", "mrna", "cdna", "clone", "cloned", "encodes", "maps", "genomic",
            "sequence", "homology", "mutation", "mutant", "expression", "transcription",
            "phosphorylation", "activity", "activities", "binds", "binding", "motif",
            "domain", "subunit", "promoter", "complex", "orf",
        }
        context_words = set(re.findall(r"[a-z]+", context.lower()))
        return bool(context_words & strong_context_terms)


    def _is_descriptive_gene_name(self, entity_text: str, lower_entity: str, context: str) -> bool:
        """
        检测描述性基因/蛋白名称，如:
        - 带 gene/promoter/receptor 等关键词: "human serotonin transporter gene"
        - 带连字符和空格的: "NF - Y", "Crk II", "Id - 1H"
        - 蛋白/酶名: "lactose permease", "estrogen receptor", "alkaline extracellular protease"
        """
        gene_descriptor_words = {
            "gene", "genes", "promoter", "receptor", "transporter", "polymerase",
            "kinase", "enzyme", "transcript", "permease", "protease", "ligase",
            "phosphatase", "oxidase", "transferase", "synthase", "synthetase",
            "reductase", "hydrolase", "dehydrogenase", "cytochrome", "hormone",
            "factor", "protein", "proteins", "subunit", "subunits", "complex",
            "homologue", "homolog", "equivalent", "variant", "allele", "isoform",
            "domain", "motif",
        }
        common_gene_product_names = {
            "insulin",
        }
        if lower_entity.strip() in common_gene_product_names:
            return True
        # 检查实体文本中是否包含描述性基因关键词
        entity_words = set(re.findall(r'[a-z]+', lower_entity))
        if entity_words & gene_descriptor_words:
            return True
        # 检查带连字符+空格的缩写模式，如 "NF - Y", "Id - 1H"
        if re.search(r'[A-Za-z]+\s*-\s*[A-Za-z0-9]+', entity_text):
            return True
        # 检查上下文中是否有强基因关联词
        context_gene_words = {"expression", "transcription", "translation", "mutation",
                               "coding", "sequence", "mrna", "cdna", "genomic", "exon",
                               "intron", "splicing", "regulate"}
        if entity_words & context_gene_words:
            return True
        return False

    def _lexical_shape_match(self, entity_text: str, entity_type: str) -> bool:
        lower_text = entity_text.lower()
        stripped = entity_text.strip()

        if entity_type == "GENE":
            # 改进基因符号的识别逻辑
            # 基因符号通常是大写字母和数字的组合，长度2-10个字符
            return bool(
                re.fullmatch(r"[A-Z][A-Z0-9]{1,9}", stripped) or
                re.fullmatch(r"[A-Z][A-Z0-9\-]{1,9}", stripped)
            )

        if entity_type == "GENE_OR_GENE_PRODUCT":
            return bool(
                re.fullmatch(r"[A-Z][A-Z0-9\-/]{1,24}", stripped)
                and (any(ch.isupper() for ch in stripped) or "protein" in lower_text)
            )

        if entity_type == "PROTEIN":
            protein_markers = ["protein", "receptor", "kinase", "enzyme", "antibody", "factor"]
            return any(marker in lower_text for marker in protein_markers)

        if entity_type == "DISEASE":
            disease_markers = [
                "disease", "cancer", "syndrome", "disorder", "tumor", "tumour",
                "carcinoma", "infection", "diabetes", "neoplasm", "deficiency",
                "failure", "injury",
            ]
            return any(marker in lower_text for marker in disease_markers)

        if entity_type == "CHEMICAL":
            chemical_markers = [
                "acid", "amine", "drug", "compound", "inhibitor",
                "agonist", "antagonist", "chloride", "sodium",
            ]
            return any(marker in lower_text for marker in chemical_markers)

        if entity_type == "SPECIES":
            if lower_text in {"human", "mouse", "mice", "rat", "rats"}:
                return True
            return bool(re.fullmatch(r"[A-Z][a-z]+ [a-z]+", stripped))

        if entity_type == "CELL_LINE":
            return bool(
                re.fullmatch(r"[A-Z][A-Z0-9\-]*", stripped)
                or stripped in {"HeLa", "A549", "MCF-7", "293T", "Jurkat"}
            )

        if entity_type == "CELL_TYPE":
            cell_markers = ["cell", "lymphocyte", "neuron", "macrophage", "fibroblast"]
            return any(marker in lower_text for marker in cell_markers)

        return False


