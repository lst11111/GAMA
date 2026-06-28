import json
from typing import List, Dict, Tuple, Set
from collections import defaultdict


def calculate_f1_from_files(pred_file: str, gold_file: str) -> Dict[str, float]:
    """
    从两个文件中读取预测结果和黄金标签，计算F1分数
    
    Args:
        pred_file: 预测结果文件路径
        gold_file: 黄金标签文件路径
    
    Returns:
        包含总体和各类别F1分数的字典
    """
    with open(pred_file, 'r', encoding='utf-8') as f:
        pred_data = json.load(f)
    
    with open(gold_file, 'r', encoding='utf-8') as f:
        gold_data = json.load(f)
    
    return calculate_f1_from_data(pred_data, gold_data)


def calculate_f1_from_data(pred_data: List[Dict], gold_data: List[Dict]) -> Dict[str, float]:
    """
    从数据中计算F1分数
    
    Args:
        pred_data: 预测结果数据
        gold_data: 黄金标签数据
    
    Returns:
        包含总体和各类别F1分数的字典
    """
    # 提取预测和黄金标签的实体
    pred_entities = extract_entities_from_results(pred_data)
    gold_entities = extract_entities_from_gold(gold_data)
    
    # 计算总体指标
    overall_metrics = calculate_metrics_for_entities(pred_entities, gold_entities)
    
    # 获取所有实体类型
    all_types = set()
    for entities in pred_entities + gold_entities:
        all_types.add(entities['type'])
    
    # 计算各类别的指标
    type_metrics = {}
    for entity_type in all_types:
        pred_filtered = [e for e in pred_entities if e['type'] == entity_type]
        gold_filtered = [e for e in gold_entities if e['type'] == entity_type]
        type_metrics[entity_type] = calculate_metrics_for_entities(pred_filtered, gold_filtered)
    
    # 合并结果
    result = {
        'overall': overall_metrics,
        'by_type': type_metrics
    }
    
    return result


def extract_entities_from_results(results: List[Dict]) -> List[Dict]:
    """
    从pipeline结果中提取预测的实体（按句子级别去重）
    
    Args:
        results: pipeline的输出结果
    
    Returns:
        实体列表，每个实体包含text, type字段
    """
    entities = []
    
    # 如果是单个结果
    if isinstance(results, dict) and 'results' in results:
        results = results['results']
    
    for result in results:
        if isinstance(result, dict):
            # 获取当前句子的所有实体
            sentence_entities = []
            
            # 检查是否有final_hypothesis
            if 'final_hypothesis' in result and result['final_hypothesis']:
                hyp = result['final_hypothesis']
                sentence_entities.append({
                    'text': hyp.get('text', '').lower(),  # 转换为小写以进行不区分大小写的比较
                    'type': hyp.get('type', '').upper(),
                })
            # 检查是否有final_entities
            elif 'final_entities' in result:
                for ent in result['final_entities']:
                    sentence_entities.append({
                        'text': ent.get('text', '').lower(),
                        'type': ent.get('label', '').upper(),
                    })
            # 检查是否有validated_entities（这是主pipeline中使用的字段）
            elif 'validated_entities' in result:
                for ent in result['validated_entities']:
                    hyp = ent.get('final_hypothesis', {})
                    sentence_entities.append({
                        'text': hyp.get('text', '').lower(),
                        'type': hyp.get('type', '').upper(),
                    })
            # 检查原始样本中的实体
            elif 'original_sample' in result:
                orig = result['original_sample']
                if 'entities' in orig:
                    for ent in orig['entities']:
                        sentence_entities.append({
                            'text': ent.get('text', '').lower(),
                            'type': ent['type'].upper(),
                        })
            
            # 对当前句子内的实体进行去重
            seen_in_sentence = set()
            unique_sentence_entities = []
            for ent in sentence_entities:
                identifier = (ent['text'], ent['type'])
                if identifier not in seen_in_sentence:
                    seen_in_sentence.add(identifier)
                    unique_sentence_entities.append(ent)
            
            # 将去重后的实体添加到总列表
            entities.extend(unique_sentence_entities)
    
    return entities


def extract_entities_from_gold(data: List[Dict]) -> List[Dict]:
    """
    从黄金标签数据中提取实体（按句子级别去重）
    
    Args:
        data: 黄金标签数据
    
    Returns:
        实体列表，每个实体包含text, type字段
    """
    entities = []
    
    for item in data:
        if 'text' in item and 'entities' in item:
            text = item['text']
            # 获取当前句子的所有实体
            sentence_entities = []
            
            for ent in item['entities']:
                # 如果实体有text字段，直接使用；否则从原文提取
                if 'text' in ent:
                    entity_text = ent['text']
                else:
                    entity_text = text[ent['span'][0]:ent['span'][1]]
                
                entity = {
                    'text': entity_text.lower(),  # 转换为小写以进行不区分大小写的比较
                    'type': ent['type'].upper(),
                }
                sentence_entities.append(entity)
            
            # 对当前句子内的实体进行去重
            seen_in_sentence = set()
            unique_sentence_entities = []
            for ent in sentence_entities:
                identifier = (ent['text'], ent['type'])
                if identifier not in seen_in_sentence:
                    seen_in_sentence.add(identifier)
                    unique_sentence_entities.append(ent)
            
            # 将去重后的实体添加到总列表
            entities.extend(unique_sentence_entities)
    
    return entities

def calculate_metrics_for_entities(pred_entities: List[Dict], gold_entities: List[Dict]) -> Dict[str, float]:
    """
    计算实体级别的指标（只考虑内容和类型，不考虑span）
    注意：这里保持每个实体实例，不去除跨句子的重复
    
    Args:
        pred_entities: 预测实体列表
        gold_entities: 黄金实体列表
    
    Returns:
        包含precision, recall, f1的字典
    """
    # 将实体列表转换为可比较的形式，但保留所有实例
    pred_list = [(ent['text'], ent['type']) for ent in pred_entities]
    gold_list = [(ent['text'], ent['type']) for ent in gold_entities]
    
    # 计算TP, FP, FN
    # 为了正确计算，我们需要考虑匹配过程
    # 每个预测实体最多匹配一个黄金实体，每个黄金实体最多被一个预测实体匹配
    matched_gold_indices = set()
    tp = 0  # 真正例
    
    for pred_idx, pred_entity in enumerate(pred_list):
        # 查找未匹配的黄金实体中是否有匹配项
        for gold_idx, gold_entity in enumerate(gold_list):
            if gold_idx not in matched_gold_indices and pred_entity == gold_entity:
                tp += 1
                matched_gold_indices.add(gold_idx)
                break
    
    fp = len(pred_list) - tp  # 假正例
    fn = len(gold_list) - tp  # 假负例
    
    # 计算精确率、召回率和F1分数
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'total_pred': len(pred_list),
        'total_gold': len(gold_list)
    }


def print_f1_report(metrics: Dict[str, float]):
    """
    打印F1报告
    
    Args:
        metrics: 包含F1指标的字典
    """
    print("=" * 60)
    print("F1 Score Report (Content and Type Only)")
    print("=" * 60)
    
    # 打印总体指标
    overall = metrics['overall']
    print(f"Overall Metrics:")
    print(f"  Precision: {overall['precision']:.4f}")
    print(f"  Recall:    {overall['recall']:.4f}")
    print(f"  F1-Score:  {overall['f1']:.4f}")
    print(f"  TP: {overall['tp']}, FP: {overall['fp']}, FN: {overall['fn']}")
    print(f"  Total Predicted: {overall['total_pred']}, Total Gold: {overall['total_gold']}")
    print()
    
    # 打印各类别指标
    print("Metrics by Entity Type:")
    for entity_type, type_metric in metrics['by_type'].items():
        print(f"  {entity_type}:")
        print(f"    Precision: {type_metric['precision']:.4f}")
        print(f"    Recall:    {type_metric['recall']:.4f}")
        print(f"    F1-Score:  {type_metric['f1']:.4f}")
        print(f"    TP: {type_metric['tp']}, FP: {type_metric['fp']}, FN: {type_metric['fn']}")
        print()


def calculate_f1_from_predictions_and_gold(predictions: List[Dict], gold_labels: List[Dict]) -> Dict[str, float]:
    """
    直接从预测结果和黄金标签计算F1分数
    
    Args:
        predictions: 预测结果列表
        gold_labels: 黄金标签列表
    
    Returns:
        包含F1指标的字典
    """
    return calculate_f1_from_data(predictions, gold_labels)