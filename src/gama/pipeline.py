import copy
import json
import os
import argparse
from datetime import datetime
from typing import Any, Dict, List, Optional

from .coding_agent import BioNERCodingAgent
from .planning_agent import BioNERPlanningAgent
from .verification_agent import BioNERVerificationAgent
from .metrics import calculate_f1_from_data, print_f1_report


class BioNERDualLoopPipeline:
    """
    双循环 Pipeline（Guideline 注入 Planning 版本）：使用 Guideline Agent 生成全局标注指南，
    将总结的 rules 传入 Planning Agent 作为识别约束，然后 Coding → Verification 串联。
    """

    def __init__(
        self,
        api_key: str,
        planning_model: str = "qwen3.7-max-2026-06-08",
        guideline_model: str = "qwen3.7-max-2026-06-08",
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        max_retries: int = 3,
        verbose: bool = True,
    ):
        self.verbose = verbose
        self.api_key = api_key
        self.guideline_model = guideline_model
        self.base_url = base_url
        self._guideline_agent = None

        self.planning_agent = BioNERPlanningAgent(
            api_key=api_key,
            model=planning_model,
            base_url=base_url,
            max_retries=max_retries,
            verbose=verbose,
        )

        self.coding_agent = BioNERCodingAgent(verbose=verbose)
        self.verification_agent = BioNERVerificationAgent(verbose=verbose)

    def log(self, message: str, level: str = "INFO") -> None:
        if self.verbose:
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp}] [{level}] [DualLoopPipeline] {message}")

    def get_guideline_agent(self):
        if self._guideline_agent is None:
            from .guideline_summarizer import BioNERSummarizeAgent
            self._guideline_agent = BioNERSummarizeAgent(
                api_key=self.api_key,
                model=self.guideline_model,
                base_url=self.base_url,
                verbose=self.verbose,
            )
        return self._guideline_agent

    def run_on_dataset(
        self,
        dataset_path: str,
        schema: List[str],
        train_data_path: str = None,
        dataset_name: str = "CustomDataset",
        max_patch_attempts: int = 3,
        save_results: bool = True,
        output_dir: str = "./results",
        batch_size: int = 1,
        calculate_f1: bool = True,
        guideline_sample_size: int = None,
        reuse_existing_guidelines: bool = False,
        guideline_path: Optional[str] = None,
        save_incremental_results: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        在整个数据集上运行 pipeline（无 Retrieval Agent）。

        Args:
            dataset_path: 数据集文件路径
            schema: 实体类型列表，例如 ["GENE", "DISEASE"]
            train_data_path: 训练数据路径（用于生成 guideline），None 则用 dataset_path
            dataset_name: 数据集名称（用于日志和输出）
            max_patch_attempts: 每个 hypothesis 最多 patch 几次
            save_results: 是否保存结果到文件
            output_dir: 输出目录
            batch_size: 批处理大小
            calculate_f1: 是否计算F1分数
            guideline_sample_size: 生成 guideline 时采样多少条训练样本
            reuse_existing_guidelines: 是否复用已有 guidelines.json，跳过规则总结阶段
            guideline_path: 已有 guidelines.json 路径；为空时使用 guidelines_output/{dataset_name}/guidelines.json
            save_incremental_results: 是否每处理完一个样本就追加写入 JSONL 结果文件

        Returns:
            处理结果列表
        """
        self.log(f"开始处理数据集: {dataset_path}")
        run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 读取数据集
        with open(dataset_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)

        total_samples = len(dataset)
        self.log(f"数据集加载完成，共 {total_samples} 条样本")

        # 数据验证和清理
        dataset = self.validate_and_clean_dataset(dataset)

        guideline_dir = os.path.join("./guidelines_output", dataset_name.replace("/", "_"))
        if reuse_existing_guidelines:
            guideline_file = guideline_path or os.path.join(guideline_dir, "guidelines.json")
            self.log(f"复用已有 Guidelines，跳过 Summarizer: {guideline_file}")
            global_guidelines = self.load_guidelines_from_file(guideline_file, schema)
        else:
            # ========== 1. 使用 Guideline Summarizer Agent 生成全局标注指南 ==========
            self.log(f"使用 Guideline Summarizer Agent 为数据集 '{dataset_name}' 生成全局指南...")
            guideline_rules = self.get_guideline_agent().summarize(
                train_data=train_data_path or dataset_path,
                output_dir=guideline_dir,
                top_k=20,
                max_workers=8,
                sample_size=guideline_sample_size,
            )
            global_guidelines = self.normalize_guideline_rules(guideline_rules, schema)
            if not global_guidelines:
                self.log("警告：Summarizer 未生成指南，使用 fallback", level="WARNING")
                global_guidelines = [{"type": t, "guideline": f"标注所有 {t} 类型的实体"} for t in schema]
            self.log(f"全局指南生成完成，共 {len(global_guidelines)} 条")
            # ================================================================

        incremental_paths = None
        if save_results and save_incremental_results:
            incremental_paths = self.init_incremental_result_files(
                output_dir=output_dir,
                dataset_name=dataset_name,
                timestamp=run_timestamp,
            )

        # 批量处理文本
        results = []
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (len(dataset) - 1) // batch_size + 1
            self.log(f"处理批次 {batch_num}/{total_batches}")

            for j, sample in enumerate(batch):
                current_sample = i + j + 1
                self.log(f"处理样本 {current_sample}/{total_samples} - 文本: {sample.get('text', '')[:50]}...")

                # 所有样本复用同一份全局指南
                guidelines = global_guidelines

                # 2. Planning Agent：生成候选实体池（注入 guidelines 作为识别约束）
                candidate_pool = self.plan_candidates(
                    text=sample.get('text', ''),
                    schema=schema,
                    guidelines=guidelines,
                )

                # 4. 双循环 refinement（返回所有通过验证的实体）
                validated_entities = self.dual_loop_refine_all(
                    text=sample.get('text', ''),
                    schema=schema,
                    guidelines=guidelines,
                    candidate_pool=candidate_pool,
                    max_patch_attempts=max_patch_attempts,
                )

                # 去重
                unique_validated_entities = self.remove_duplicate_entities(validated_entities)

                # 组装结果
                result = {
                    "text": sample.get('text', ''),
                    "schema": schema,
                    "guidelines": guidelines,
                    "candidate_pool": candidate_pool,
                    "sample_index": i + j,
                    "original_sample": sample,
                    "validated_entities": unique_validated_entities,
                    "success": len(unique_validated_entities) > 0,
                }
                results.append(result)

                if incremental_paths:
                    self.append_incremental_result(result, incremental_paths)

                # 实时显示预测数量
                pred_count = len(unique_validated_entities)
                gold_count = len(sample.get('entities', []))
                self.log(f"  - 预测实体: {pred_count}, 真实实体: {gold_count}")

                # 可选：实时计算累积F1（代价较高，可注释）
                if calculate_f1 and len(results) > 0:
                    current_f1 = self.calculate_f1_score(results, dataset[:len(results)])
                    overall_f1 = current_f1['overall']['f1']
                    overall_precision = current_f1['overall']['precision']
                    overall_recall = current_f1['overall']['recall']
                    self.log(f"  - 累积F1: {overall_f1:.4f}, P: {overall_precision:.4f}, R: {overall_recall:.4f} (样本 {len(results)}/{total_samples})")

        # 最终统计
        successful_count = sum(1 for r in results if r.get("success", False))
        self.log(f"数据集处理完成！成功处理 {successful_count}/{total_samples} 个样本")

        # 保存结果
        if save_results:
            self.save_dataset_results(results, output_dir, dataset_name, timestamp=run_timestamp)
            self.save_simplified_results(results, output_dir, dataset_name, timestamp=run_timestamp)

        # 计算最终F1分数
        if calculate_f1:
            self.log("开始计算最终F1分数...")
            f1_metrics = self.calculate_f1_score(results, dataset)
            print_f1_report(f1_metrics)
            self.save_f1_report(f1_metrics, output_dir, dataset_name, timestamp=run_timestamp)

        return results

    # ---------- 辅助方法 ----------
    def plan_candidates(
        self,
        text: str,
        schema: List[str],
        guidelines: List[Dict[str, str]] = None,
    ) -> List[Dict[str, Any]]:
        self.log("调用 Planning Agent 生成候选实体池（注入 guidelines）")
        # Convert guidelines to dict format for Planning Agent
        guideline_map = self._guidelines_to_map(guidelines) if guidelines else {}
        hypotheses = self.planning_agent.plan(
            text=text,
            schema=schema,
            guidelines=guideline_map,
        )
        ranked = sorted(
            hypotheses,
            key=lambda x: float(x.get("confidence", 0.0)),
            reverse=True,
        )
        self.log(f"Planning 完成，共得到 {len(ranked)} 个候选 hypothesis", level="SUCCESS")
        return ranked

    def dual_loop_refine_all(
        self,
        text: str,
        schema: List[str],
        guidelines: List[Dict[str, str]],
        candidate_pool: List[Dict[str, Any]],
        max_patch_attempts: int = 3,
    ) -> List[Dict[str, Any]]:
        self.log("进入 Dual-Loop Refinement (处理所有实体)")
        remaining_pool = copy.deepcopy(candidate_pool)
        validated_entities = []
        schema_guidelines = self._guidelines_to_map(guidelines)

        while remaining_pool:
            current = remaining_pool.pop(0)
            self.log(f"选择当前 hypothesis: {current.get('text')} -> {current.get('type')} (confidence={current.get('confidence')})")

            current_trace = []
            working_hypothesis = copy.deepcopy(current)

            for attempt_idx in range(1, max_patch_attempts + 1):
                self.log(f"内层尝试 {attempt_idx}/{max_patch_attempts}")

                coding_result = self._build_code_for_single_hypothesis(
                    text=text,
                    hypothesis=working_hypothesis,
                    schema=schema,
                )

                trace_item = {
                    "selected_hypothesis_rank": current.get("rank"),
                    "attempt": attempt_idx,
                    "input_hypothesis": copy.deepcopy(working_hypothesis),
                    "coding_success": coding_result["success"],
                    "generated_code": coding_result.get("code"),
                    "entity_object": coding_result.get("entity_object"),
                    "coding_error": coding_result.get("error", ""),
                }

                if not coding_result["success"]:
                    epsilon = f"Coding Failed: {coding_result['error']}"
                    trace_item["verdict"] = False
                    trace_item["epsilon"] = epsilon
                    current_trace.append(trace_item)
                    break

                verdict, epsilon = self.verification_agent.verify(
                    code=coding_result["code"],
                    text=text,
                    schema=schema,
                    entity_object=coding_result["entity_object"],
                    schema_guidelines=schema_guidelines,
                )

                trace_item["verdict"] = verdict
                trace_item["epsilon"] = epsilon
                current_trace.append(trace_item)

                if verdict:
                    self.log(f"当前 hypothesis 通过验证: {working_hypothesis.get('text')}")
                    validated_entities.append({
                        "final_hypothesis": working_hypothesis,
                        "final_code": coding_result["code"],
                        "final_object": coding_result["entity_object"],
                        "epsilon": "",
                        "trace": current_trace,
                    })
                    break
                else:
                    patched = self.patch_hypothesis(
                        text=text,
                        hypothesis=working_hypothesis,
                        epsilon=epsilon,
                        schema=schema,
                    )
                    if patched is None or self._same_hypothesis(patched, working_hypothesis):
                        self.log("验证失败且无法修补，跳过当前实体", level="WARNING")
                        break
                    working_hypothesis = patched

        self.log(f"处理完成，共找到 {len(validated_entities)} 个通过验证的实体")
        return validated_entities

    def _build_code_for_single_hypothesis(
        self,
        text: str,
        hypothesis: Dict[str, Any],
        schema: List[str],
    ) -> Dict[str, Any]:
        try:
            payload = self.coding_agent.normalize_hypothesis(text, hypothesis)
            entity_type = payload["entity_type"]
            model_registry = self.coding_agent.compile_schema(schema)
            if entity_type not in model_registry:
                raise ValueError(f"未知实体类型: {entity_type}")
            code = self.coding_agent.generate_code(entity_type, payload)
            model_name = self.coding_agent._sanitize_model_name(entity_type)
            entity_obj = self.coding_agent.execute_code(
                code=code,
                model_namespace={model_name: model_registry[entity_type]},
            )
            return {
                "success": True,
                "payload": payload,
                "code": code,
                "entity_object": entity_obj.model_dump(),
                "error": "",
            }
        except Exception as e:
            return {
                "success": False,
                "payload": None,
                "code": "",
                "entity_object": None,
                "error": f"{type(e).__name__}: {str(e)}",
            }

    def patch_hypothesis(
        self,
        text: str,
        hypothesis: Dict[str, Any],
        epsilon: str,
        schema: List[str],
    ) -> Optional[Dict[str, Any]]:
        patched = copy.deepcopy(hypothesis)
        if "type" in patched and isinstance(patched["type"], str):
            patched["type"] = patched["type"].upper()
            if patched["type"] in ('DSIEASE', 'DSIASE'):
                patched["type"] = 'DISEASE'
        if patched.get("type") not in schema:
            if len(schema) == 1:
                patched["type"] = schema[0]
            else:
                return None
        if "confidence" in patched:
            try:
                patched["confidence"] = max(0.0, min(1.0, float(patched["confidence"])))
            except Exception:
                patched["confidence"] = 0.5
        else:
            patched["confidence"] = 0.5
        if not patched.get("rationale"):
            patched["rationale"] = "Patched by pipeline after verification failure."
        return patched

    def _same_hypothesis(self, h1: Dict[str, Any], h2: Dict[str, Any]) -> bool:
        return (
            h1.get("type") == h2.get("type")
            and h1.get("text") == h2.get("text")
            and float(h1.get("confidence", 0.0)) == float(h2.get("confidence", 0.0))
            and h1.get("rationale") == h2.get("rationale")
        )

    def _guidelines_to_map(self, guidelines: List[Dict[str, str]]) -> Dict[str, str]:
        guideline_map = {}
        for item in guidelines:
            ent_type = str(item.get("type", "")).upper()
            guideline = str(item.get("guideline", ""))
            if ent_type:
                guideline_map[ent_type] = guideline
        return guideline_map

    def normalize_guideline_rules(self, guideline_rules: Any, schema: List[str]) -> List[Dict[str, str]]:
        schema_set = {s.upper() for s in schema}
        global_guidelines = []

        if isinstance(guideline_rules, list):
            for item in guideline_rules:
                if not isinstance(item, dict):
                    continue
                ent_type = str(item.get("type", "")).upper()
                guideline = item.get("guideline", "")
                if ent_type in schema_set and guideline:
                    global_guidelines.append({
                        "type": ent_type,
                        "guideline": self._join_guideline_value(guideline),
                    })
            return global_guidelines

        if not isinstance(guideline_rules, dict):
            return []

        for etype, rules in guideline_rules.items():
            ent_type = str(etype).upper()
            if ent_type in schema_set:
                guideline = self._join_guideline_value(rules)
                if guideline:
                    global_guidelines.append({
                        "type": ent_type,
                        "guideline": guideline,
                    })
        return global_guidelines

    def load_guidelines_from_file(self, guideline_file: str, schema: List[str]) -> List[Dict[str, str]]:
        if not os.path.exists(guideline_file):
            raise FileNotFoundError(
                f"找不到已有 guideline 文件: {guideline_file}. "
                "请先跑一次规则总结，或传入正确的 guideline_path。"
            )

        with open(guideline_file, 'r', encoding='utf-8') as f:
            guideline_rules = json.load(f)

        global_guidelines = self.normalize_guideline_rules(guideline_rules, schema)
        if not global_guidelines:
            raise ValueError(
                f"已有 guideline 文件中没有匹配当前 schema={schema} 的有效规则: {guideline_file}"
            )

        self.log(f"已加载已有 Guidelines，共 {len(global_guidelines)} 条", level="SUCCESS")
        return global_guidelines

    def _join_guideline_value(self, value: Any) -> str:
        if isinstance(value, list):
            return "; ".join(str(item) for item in value if str(item).strip())
        if isinstance(value, str):
            return value.strip()
        return str(value).strip()

    # ---------- 数据处理与保存方法 ----------
    def validate_and_clean_dataset(self, dataset: List[Dict]) -> List[Dict]:
        cleaned_dataset = []
        for item in dataset:
            cleaned_item = copy.deepcopy(item)
            entities = cleaned_item.get('entities', [])
            cleaned_entities = []
            for entity in entities:
                entity_type = entity.get('type', '').upper()
                if entity_type in ('DSIEASE', 'DSIASE'):
                    entity_type = 'DISEASE'
                cleaned_entities.append({
                    'text': entity.get('text', ''),
                    'type': entity_type
                })
            cleaned_item['entities'] = cleaned_entities
            cleaned_dataset.append(cleaned_item)
        return cleaned_dataset

    def remove_duplicate_entities(self, validated_entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen = set()
        unique_entities = []
        for entity in validated_entities:
            hypothesis = entity.get("final_hypothesis", {})
            text = hypothesis.get("text", "").lower()
            entity_type = hypothesis.get("type", "").upper()
            identifier = (text, entity_type)
            if identifier not in seen:
                seen.add(identifier)
                unique_entities.append(entity)
        return unique_entities

    def init_incremental_result_files(self, output_dir: str, dataset_name: str, timestamp: str) -> Dict[str, str]:
        os.makedirs(output_dir, exist_ok=True)
        paths = {
            "full": os.path.join(
                output_dir,
                f"incremental_bio_ner_dataset_results_{dataset_name}_{timestamp}.jsonl",
            ),
            "simplified": os.path.join(
                output_dir,
                f"incremental_simplified_results_{dataset_name}_{timestamp}.jsonl",
            ),
        }
        for path in paths.values():
            with open(path, 'w', encoding='utf-8'):
                pass
        self.log(f"增量完整结果将写入: {paths['full']}", level="SUCCESS")
        self.log(f"增量简化结果将写入: {paths['simplified']}", level="SUCCESS")
        return paths

    def append_incremental_result(self, result: Dict[str, Any], paths: Dict[str, str]) -> None:
        with open(paths["full"], 'a', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False)
            f.write("\n")

        simplified = self.simplify_result(result)
        with open(paths["simplified"], 'a', encoding='utf-8') as f:
            json.dump(simplified, f, ensure_ascii=False)
            f.write("\n")

    def simplify_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        predicted_entities = []
        for ent in result.get("validated_entities", []):
            hypothesis = ent.get("final_hypothesis", {})
            try:
                confidence = round(float(hypothesis.get("confidence", 0.8)), 3)
            except (TypeError, ValueError):
                confidence = 0.8
            predicted_entities.append({
                "text": hypothesis.get("text", ""),
                "type": hypothesis.get("type", ""),
                "confidence": confidence,
            })
        return {
            "text": result.get("text", ""),
            "sample_index": result.get("sample_index"),
            "original_entities": result.get("original_sample", {}).get("entities", []),
            "predicted_entities": predicted_entities,
        }

    def save_dataset_results(
        self,
        results: List[Dict],
        output_dir: str,
        dataset_name: str,
        timestamp: Optional[str] = None,
    ) -> str:
        os.makedirs(output_dir, exist_ok=True)
        timestamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(output_dir, f"bio_ner_dataset_results_{dataset_name}_{timestamp}.json")
        save_data = {
            "dataset_name": dataset_name,
            "total_samples": len(results),
            "successful_samples": sum(1 for r in results if r.get("success", False)),
            "results": results
        }
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)
        self.log(f"完整结果已保存到: {output_path}", level="SUCCESS")
        return output_path

    def save_simplified_results(
        self,
        results: List[Dict],
        output_dir: str,
        dataset_name: str,
        timestamp: Optional[str] = None,
    ) -> str:
        os.makedirs(output_dir, exist_ok=True)
        timestamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(output_dir, f"simplified_results_{dataset_name}_{timestamp}.json")
        simplified_data = [self.simplify_result(result) for result in results]
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(simplified_data, f, ensure_ascii=False, indent=2)
        self.log(f"简化版结果已保存到: {output_path}", level="SUCCESS")
        return output_path

    # ---------- F1 计算相关 ----------
    def calculate_f1_score(self, pred_results: List[Dict], gold_dataset: List[Dict]) -> Dict[str, Any]:
        # 使用 metrics.py 中的统一函数
        return calculate_f1_from_data(pred_results, gold_dataset)

    def save_f1_report(
        self,
        f1_metrics: Dict,
        output_dir: str,
        dataset_name: str,
        timestamp: Optional[str] = None,
    ) -> None:
        os.makedirs(output_dir, exist_ok=True)
        timestamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(output_dir, f"f1_report_{dataset_name}_{timestamp}.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(f1_metrics, f, ensure_ascii=False, indent=2)
        self.log(f"F1报告已保存到: {output_path}", level="SUCCESS")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the GAMA BioNER dual-loop pipeline.")
    parser.add_argument("--dataset", default="examples/data/sample_bioner.json", help="JSON dataset path.")
    parser.add_argument("--train-data", default=None, help="Training JSON/JSONL path for guideline summarization.")
    parser.add_argument("--schema", nargs="+", default=["GENE"], help="Entity types, for example: GENE DISEASE.")
    parser.add_argument("--dataset-name", default="sample_bioner", help="Name used in output files.")
    parser.add_argument("--output-dir", default="results", help="Directory for run outputs.")
    parser.add_argument("--api-key", default=os.getenv("BIO_NER_API_KEY"), help="API key. Defaults to BIO_NER_API_KEY.")
    parser.add_argument(
        "--base-url",
        default=os.getenv("BIO_NER_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        help="OpenAI-compatible chat completions base URL.",
    )
    parser.add_argument(
        "--planning-model",
        default=os.getenv("BIO_NER_PLANNING_MODEL", "qwen3.7-max-2026-06-08"),
        help="Planning model name.",
    )
    parser.add_argument(
        "--guideline-model",
        default=os.getenv("BIO_NER_GUIDELINE_MODEL", "qwen3.7-max-2026-06-08"),
        help="Guideline summarizer model name.",
    )
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--max-patch-attempts", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--guideline-sample-size", type=int, default=None)
    parser.add_argument("--reuse-existing-guidelines", action="store_true")
    parser.add_argument("--guideline-path", default=None)
    parser.add_argument("--no-f1", action="store_true", help="Disable F1 calculation.")
    parser.add_argument("--no-incremental-results", action="store_true", help="Disable JSONL incremental outputs.")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if not args.api_key:
        raise SystemExit("Missing API key. Set BIO_NER_API_KEY or pass --api-key.")

    pipeline = BioNERDualLoopPipeline(
        api_key=args.api_key,
        planning_model=args.planning_model,
        guideline_model=args.guideline_model,
        base_url=args.base_url,
        max_retries=args.max_retries,
        verbose=not args.quiet,
    )

    results = pipeline.run_on_dataset(
        dataset_path=args.dataset,
        train_data_path=args.train_data,
        schema=args.schema,
        dataset_name=args.dataset_name,
        max_patch_attempts=args.max_patch_attempts,
        save_results=True,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        calculate_f1=not args.no_f1,
        guideline_sample_size=args.guideline_sample_size,
        reuse_existing_guidelines=args.reuse_existing_guidelines,
        guideline_path=args.guideline_path,
        save_incremental_results=not args.no_incremental_results,
    )

    success_cnt = sum(1 for r in results if r.get("success", False))
    print(f"Dataset processing finished: {len(results)} samples, {success_cnt} successful.")


if __name__ == "__main__":
    main()
