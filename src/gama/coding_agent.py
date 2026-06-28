import json
from datetime import datetime
from typing import Any, Dict, List, Literal, Tuple, Type

from pydantic import BaseModel, ConfigDict, Field, ValidationError, create_model, field_validator, model_validator


class BioNEREntityTemplate(BaseModel):
    """
    BioNER 实体对象模板。
    所有具体实体类型（如 GENEEntity, DISEASEEntity）都从这个模板动态继承。
    """
    model_config = ConfigDict(extra="forbid")

    entity_type: str
    text: str
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str
    source_text: str

    @field_validator("text")
    @classmethod
    def validate_text_not_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text 不能为空")
        return value.strip()


class BioNERCodingAgent:
    """
    Coding Agent for BioNER.
    思路：
    1. 将 schema 编译为 Python BaseModel
    2. 将 planning agent 输出的 hypothesis 转为可执行 Python 代码
    3. 通过实例化 BaseModel 做确定性运行时验证
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose

    def log(self, message: str, level: str = "INFO") -> None:
        if self.verbose:
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp}] [{level}] [CodingAgent] {message}")

    def _sanitize_model_name(self, entity_type: str) -> str:
        cleaned = "".join(ch if ch.isalnum() else "_" for ch in entity_type)
        return f"{cleaned}Entity"

    def compile_schema(self, schema: List[str]) -> Dict[str, Type[BaseModel]]:
        """
        将实体类型列表动态编译为 Pydantic BaseModel 子类。

        例如:
        schema = ["GENE", "DISEASE"]

        =>
        {
            "GENE": GENEEntity,
            "DISEASE": DISEASEEntity
        }
        """
        self.log(f"开始编译 schema，共 {len(schema)} 个实体类型")

        model_registry: Dict[str, Type[BaseModel]] = {}

        for entity_type in schema:
            model_name = self._sanitize_model_name(entity_type)

            # 用 Literal 强制 entity_type 只能等于当前 schema 类型
            entity_model = create_model(
                model_name,
                entity_type=(Literal[entity_type], entity_type),
                __base__=BioNEREntityTemplate,
            )

            model_registry[entity_type] = entity_model
            self.log(f"已编译实体模型: {model_name}")

        return model_registry

    def sort_hypotheses(self, hypotheses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        按 confidence 从高到低排序。
        """
        return sorted(
            hypotheses,
            key=lambda x: float(x.get("confidence", 0.0)),
            reverse=True,
        )

    def normalize_hypothesis(self, text: str, hypothesis: Dict[str, Any]) -> Dict[str, Any]:
        """
        将 Planning Agent 输出规范化为 Coding Agent 所需格式。
        """
        if "type" not in hypothesis:
            raise ValueError("hypothesis 缺少字段: type")
        if "text" not in hypothesis:
            raise ValueError("hypothesis 缺少字段: text")

        normalized = {
            "entity_type": hypothesis["type"],
            "text": hypothesis["text"],
            "confidence": float(hypothesis.get("confidence", 0.0)),
            "rationale": str(hypothesis.get("rationale", "")),
            "source_text": text,
        }
        return normalized

    def generate_code(self, entity_type: str, payload: Dict[str, Any]) -> str:
        """
        生成可执行的 Python 实例化代码。
        """
        model_name = self._sanitize_model_name(entity_type)

        lines = [
            f"entity = {model_name}(",
            f"    entity_type={payload['entity_type']!r},",
            f"    text={payload['text']!r},",
            f"    confidence={payload['confidence']!r},",
            f"    rationale={payload['rationale']!r},",
            f"    source_text={payload['source_text']!r},",
            ")",
        ]
        return "\n".join(lines)

    def execute_code(
        self,
        code: str,
        model_namespace: Dict[str, Type[BaseModel]],
    ) -> BaseModel:
        """
        在受限环境中执行生成的代码，并返回实例化结果。
        """
        local_vars: Dict[str, Any] = {}
        safe_globals: Dict[str, Any] = {"__builtins__": {}}
        safe_globals.update(model_namespace)

        exec(code, safe_globals, local_vars)

        if "entity" not in local_vars:
            raise RuntimeError("代码执行后未生成变量 'entity'")

        entity = local_vars["entity"]
        if not isinstance(entity, BaseModel):
            raise RuntimeError("生成结果不是 BaseModel 实例")

        return entity

    def build_entity_objects(
        self,
        text: str,
        hypotheses: List[Dict[str, Any]],
        schema: List[str],
        top_k: int = None,
    ) -> Dict[str, Any]:
        """
        将多个候选实体 hypothesis 转成可执行代码并进行运行时验证。

        Returns:
        {
            "valid_entities": [...],
            "failed_cases": [...],
            "generated_codes": [...]
        }
        """
        self.log("开始构建实体对象")
        self.log(f"输入文本长度: {len(text)}")
        self.log(f"候选 hypothesis 数量: {len(hypotheses)}")

        model_registry = self.compile_schema(schema)
        ranked_hypotheses = self.sort_hypotheses(hypotheses)

        if top_k is not None:
            ranked_hypotheses = ranked_hypotheses[:top_k]
            self.log(f"仅保留 top-{top_k} 个 hypothesis")

        valid_entities: List[Dict[str, Any]] = []
        failed_cases: List[Dict[str, Any]] = []
        generated_codes: List[Dict[str, Any]] = []

        seen = set()

        for idx, hypothesis in enumerate(ranked_hypotheses, start=1):
            try:
                payload = self.normalize_hypothesis(text, hypothesis)
                entity_type = payload["entity_type"]

                if entity_type not in model_registry:
                    raise ValueError(f"未知实体类型: {entity_type}")

                dedup_key = (payload["entity_type"], payload["text"])
                if dedup_key in seen:
                    self.log(f"跳过重复 hypothesis: {dedup_key}", level="WARNING")
                    continue

                code = self.generate_code(entity_type, payload)
                generated_codes.append(
                    {
                        "rank": idx,
                        "entity_type": entity_type,
                        "confidence": payload["confidence"],
                        "code": code,
                    }
                )

                model_name = self._sanitize_model_name(entity_type)
                entity_obj = self.execute_code(
                    code=code,
                    model_namespace={model_name: model_registry[entity_type]},
                )

                valid_entities.append(
                    {
                        "rank": idx,
                        "entity_type": entity_type,
                        "confidence": payload["confidence"],
                        "object": entity_obj.model_dump(),
                        "code": code,
                    }
                )

                seen.add(dedup_key)
                self.log(
                    f"实例化成功: {payload['text']} -> {entity_type} "
                    f"(confidence={payload['confidence']:.3f})",
                    level="SUCCESS",
                )

            except ValidationError as e:
                failed_cases.append(
                    {
                        "rank": idx,
                        "hypothesis": hypothesis,
                        "error_type": "ValidationError",
                        "error": str(e),
                    }
                )
                self.log(f"验证失败: {e}", level="ERROR")

            except Exception as e:
                failed_cases.append(
                    {
                        "rank": idx,
                        "hypothesis": hypothesis,
                        "error_type": type(e).__name__,
                        "error": str(e),
                    }
                )
                self.log(f"构建失败: {e}", level="ERROR")

        self.log(
            f"构建完成: valid={len(valid_entities)}, failed={len(failed_cases)}",
            level="SUCCESS",
        )

        return {
            "valid_entities": valid_entities,
            "failed_cases": failed_cases,
            "generated_codes": generated_codes,
        }

    def build_best_entity(
        self,
        text: str,
        hypotheses: List[Dict[str, Any]],
        schema: List[str],
    ) -> Dict[str, Any]:
        """
        只返回最高分且通过运行时验证的一个实体对象。
        更接近论文里"highest-scoring hypothesis -> executable Python"的描述。
        """
        result = self.build_entity_objects(
            text=text,
            hypotheses=hypotheses,
            schema=schema,
            top_k=None,
        )

        if result["valid_entities"]:
            return result["valid_entities"][0]

        return {}

    def render_schema_templates(self, schema: List[str]) -> Dict[str, str]:
        """
        返回各实体类型对应的实例化模板，便于调试或给后续 Verification Agent 使用。
        """
        templates = {}
        for entity_type in schema:
            model_name = self._sanitize_model_name(entity_type)
            templates[entity_type] = (
                f"{model_name}(\n"
                f"    entity_type={entity_type!r},\n"
                f"    text='...',\n"
                f"    confidence=0.0,\n"
                f"    rationale='...',\n"
                f"    source_text='...'\n"
                f")"
            )
        return templates

