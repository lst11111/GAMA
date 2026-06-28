import argparse
import json
from typing import List, Dict, Any


def parse_tsv_file(file_path: str, entity_type: str = "GENE") -> List[Dict[str, Any]]:
    """
    解析TSV文件并转换为所需格式
    IOB格式: 每行一个token \t label，空行分隔句子
    标签格式: B-{entity_type}, I-{entity_type}, O
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    documents = []
    current_sentence_tokens = []
    current_sentence_labels = []
    
    for line in lines:
        line = line.rstrip('\n\r')  # 移除换行符
        
        if not line.strip():  # 空行表示句子结束
            if current_sentence_tokens:
                doc = process_sentence(current_sentence_tokens, current_sentence_labels, entity_type)
                documents.append(doc)

                # 重置
                current_sentence_tokens = []
                current_sentence_labels = []
        else:
            parts = line.split('\t')
            if len(parts) >= 2:
                token = parts[0]
                label = parts[1].strip()
                current_sentence_tokens.append(token)
                current_sentence_labels.append(label)

    # 处理最后一个句子（如果有）
    if current_sentence_tokens:
        doc = process_sentence(current_sentence_tokens, current_sentence_labels, entity_type)
        documents.append(doc)
    
    return documents


def process_sentence(tokens: List[str], labels: List[str], entity_type: str = "GENE") -> Dict[str, Any]:
    """
    处理一个句子，提取指定类型的实体
    """
    # 构建句子文本
    sentence_text = ' '.join(tokens)

    # 提取实体
    entities = []
    i = 0
    b_prefix = f'B-{entity_type}'
    i_prefix = f'I-{entity_type}'
    while i < len(labels):
        if labels[i].startswith(b_prefix):  # 开始一个新实体
            # 找到实体的开始和结束位置
            start_idx = i
            entity_tokens = [tokens[i]]

            # 查找连续的I标签
            j = i + 1
            while j < len(labels) and labels[j] == i_prefix:
                entity_tokens.append(tokens[j])
                j += 1

            # 构建实体文本
            entity_text = ' '.join(entity_tokens)

            # 在句子中找到实体的确切位置
            start_pos = find_entity_position(sentence_text, entity_text, start_idx, tokens)
            if start_pos != -1:
                end_pos = start_pos + len(entity_text)
                entities.append({
                    "text": entity_text,
                    "type": entity_type
                })

            i = j  # 跳到下一个非I的位置
        else:
            i += 1

    return {
        "text": sentence_text,
        "entities": entities
    }


def find_entity_position(sentence: str, entity: str, token_idx: int, tokens: List[str]) -> int:
    """
    在句子中找到实体的精确位置
    """
    # 首先尝试直接搜索
    pos = sentence.find(entity)
    if pos != -1:
        return pos
    
    # 如果直接搜索失败，尝试更精确的方法
    # 重建从token_idx开始的部分句子
    partial_tokens = tokens[token_idx:]
    partial_sentence = ' '.join(partial_tokens)
    
    # 在部分句子中查找
    pos = partial_sentence.find(entity)
    if pos != -1:
        # 计算在完整句子中的位置
        prefix_length = len(sentence) - len(partial_sentence)
        return prefix_length + pos
    
    # 如果仍然找不到，尝试去掉多余的空格
    entity_clean = ' '.join(entity.split())
    pos = sentence.find(entity_clean)
    if pos != -1:
        return pos
    
    return -1


def convert_bc2gm_iob_to_json(input_folder: str, output_file: str, entity_type: str = "GENE"):
    """
    将IOB格式数据集转换为JSON格式
    """
    all_documents = []

    # 处理训练集
    train_file = f"{input_folder}/train.tsv"
    print(f"正在处理训练集: {train_file}")
    train_docs = parse_tsv_file(train_file, entity_type)
    all_documents.extend(train_docs)
    print(f"训练集处理完成，共 {len(train_docs)} 个文档")

    # 处理开发集
    devel_file = f"{input_folder}/devel.tsv"
    print(f"正在处理开发集: {devel_file}")
    devel_docs = parse_tsv_file(devel_file, entity_type)
    all_documents.extend(devel_docs)
    print(f"开发集处理完成，共 {len(devel_docs)} 个文档")

    # 处理测试集
    test_file = f"{input_folder}/test.tsv"
    print(f"正在处理测试集: {test_file}")
    test_docs = parse_tsv_file(test_file, entity_type)
    all_documents.extend(test_docs)
    print(f"测试集处理完成，共 {len(test_docs)} 个文档")

    # 保存为JSON格式
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_documents, f, ensure_ascii=False, indent=2)

    print(f"转换完成！总共 {len(all_documents)} 个文档已保存到 {output_file}")

    # 显示一些统计信息
    total_entities = sum(len(doc['entities']) for doc in all_documents)
    print(f"总共找到 {total_entities} 个 {entity_type} 实体")


def convert_single_file(input_file: str, output_file: str, entity_type: str = "GENE"):
    """
    转换单个TSV文件
    """
    print(f"正在处理文件: {input_file}")
    documents = parse_tsv_file(input_file, entity_type)

    # 保存为JSON格式
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(documents, f, ensure_ascii=False, indent=2)

    print(f"转换完成！共 {len(documents)} 个文档已保存到 {output_file}")

    # 显示一些统计信息
    total_entities = sum(len(doc['entities']) for doc in documents)
    print(f"总共找到 {total_entities} 个 {entity_type} 实体")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert IOB/TSV BioNER data to JSON.")
    parser.add_argument("--input", required=True, help="Input TSV file or folder containing train/devel/test.tsv.")
    parser.add_argument("--output", required=True, help="Output JSON file.")
    parser.add_argument("--entity-type", default="GENE", help="Entity type label, for example GENE or DISEASE.")
    parser.add_argument(
        "--folder",
        action="store_true",
        help="Treat --input as a folder and merge train.tsv, devel.tsv, and test.tsv.",
    )
    args = parser.parse_args()

    if args.folder:
        convert_bc2gm_iob_to_json(args.input, args.output, args.entity_type)
    else:
        convert_single_file(args.input, args.output, args.entity_type)


if __name__ == "__main__":
    main()
