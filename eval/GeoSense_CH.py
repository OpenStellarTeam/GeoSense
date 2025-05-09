# -*- coding: UTF-8 -*-
"""
文件名: MetaGeo.py
日期: 2025.03.05
描述: 多模态模型评估框架
作者: @hongyuan
"""
import argparse
import json
import logging
import re
import time
from typing import Dict, List

from openai import OpenAI
from tqdm import tqdm

TOOL_NAME = "gpt-4o-0513"
# TOOL_NAME = "deepseek-v3"
# TOOL_NAME = "qwen-turbo"
# 配置全局日志
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)

class MetaGeoEvaluator:
    """几何元认知能力评估器"""
    
    # API配置
    API_CONFIG = {
    "api_key": "replace with your api key",
    "base_url": "replace with your base url"
    }
    
    # 评估参数
    NUM_TRIALS = 5
    THRESHOLD = 0.6
    MAX_RETRIES = 10
    RETRY_DELAY = 3  # 秒
    temperature = 0.0002

    def __init__(self, args):
        # 文件路径
        self.generate_texts_path = args.generate_texts_path
        self.ground_truth_path = args.ground_truth_path
        self.output_dir = args.output_dir
        self.model_name = args.model_name

        self.output_path = f"{self.output_dir}/{self.model_name}.json"
        
        # 初始化组件
        self.client = OpenAI(**self.API_CONFIG)
        self.evaluation_data = self._prepare_evaluation_data()
        self._evaluate_final_answers()
        
        # 输出最终结果
        answer_acc, knowledge_acc, knowledge_alignment_acc, knowledge_alignment_precision, knowledge_alignment_recall = self._calculate_accuracy()
        print(f"评估完成，答案准确率: {answer_acc:.2%} | 知识点应用率: {knowledge_acc:.2%} | 知识点对应率: {knowledge_alignment_acc:.2%} | 知识点对应Precision: {knowledge_alignment_precision:.2%} | 知识点对应Recall: {knowledge_alignment_recall:.2%}")

        # 保存最终结果
        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(self.evaluation_data, f, ensure_ascii=False, indent=4)
        saticts = {
            "answer_acc": answer_acc,
            "knowledge_acc": knowledge_acc,
            "knowledge_alignment_acc": knowledge_alignment_acc,
            "knowledge_alignment_precision": knowledge_alignment_precision,
            "knowledge_alignment_recall": knowledge_alignment_recall
        }
        with open(self.output_path.replace(".json", "_satistics.json"), "w", encoding="utf-8") as f:
            json.dump(saticts, f, ensure_ascii=False, indent=4)


    def _prepare_evaluation_data(self) -> List[Dict]:
        """准备评估数据集"""
        # 读取原始文件（保留原始路径）
        with open(self.generate_texts_path, 'r', encoding='utf-8') as f:
            reasoning_data = [json.loads(line) for line in f if line.strip()]
        
        with open(self.ground_truth_path, 'r', encoding='utf-8') as f:
            ground_truth_data = [json.loads(line) for line in f]

        # 构建数据映射
        truth_map = {item["img_path"]: item for item in ground_truth_data}
        return [
            {**truth_map[item["image"][0]], "predict": item["predict"]}
            for item in reasoning_data
            if item["image"][0] in truth_map
        ]

    def _evaluate_final_answers(self) -> None:
        """执行答案评估流程"""
        for item in tqdm(self.evaluation_data, desc="评估进度"):
            item["answer_score"] = self._evaluate_single_item(item)
            # 新增知识点识别评估
            try:
                correct, item["knowledge_score"] = self._evaluate_knowledge_points(item)
            except Exception as e:
                logging.error(f"在这里报错 {str(e)}")
                print(item)
            if item['knowledge_score'] is None:
                print(item)
            # 新增知识点对应评估
            knowledge_alignment_score, item_precision, item_recall = self._evaluate_knowledge_alignment(item)
            item["knowledge_alignment_score"] = knowledge_alignment_score / correct if correct > 0 else 0.0
            item["knowledge_alignment_precision"] = item_precision / correct if correct > 0 else 0.0
            item["knowledge_alignment_recall"] = item_recall / correct if correct > 0 else 0.0
    
    def _evaluate_knowledge_points(self, item: Dict) -> float:
        """评估知识点应用情况"""
        if not item.get("knowledge_points") or item["knowledge_points"] == []:
            logging.warning(f"{item['img_path']}中的知识点为空")
            return 0.0, 0.0
            
        total = len(item["knowledge_points"])
        correct = 0
        
        for kp in item["knowledge_points"]:
            prompt = self._build_knowledge_prompt(
                item["predict"], 
                kp["name"],
                kp["content"]
            )
            try:
                positive = self._get_positive_count(prompt)
                if positive / self.NUM_TRIALS >= self.THRESHOLD:
                    correct += 1
                    kp["kp_recog_score"] = 1
                else:
                    kp["kp_recog_score"] = 0
            except Exception as e:
                logging.error(f"知识点评估失败: {str(e)}")
                continue
                
        return correct, correct / total if total > 0 else 0.0
    
    def _evaluate_knowledge_alignment(self, item: Dict) -> float:
        """第三步：知识点与几何图的对应评估"""
        if not item.get("knowledge_points"):
            logging.warning(f"{item['img_path']}中的知识点为空")
            return 0.0
        item_alignment_score = 0
        item_precision = 0
        item_recall = 0
        for kp in item["knowledge_points"]:
            # 首先判断该 kp 是否被识别
            if kp.get("kp_recog_score") != 1:
                kp["kp_alignment_score"] = 0
                kp["extracted"] = ""
                kp["num_acc"] = 0
                kp["total_notes"] = 0.0
                continue
            # 第一小步：提取相关文本
            extracted_content = self._extract_related_content(
                item["predict"],
                kp["name"],
                kp["content"]
            )
            
            # 第二小步：评估对应准确性
            alignment_metrics = self._evaluate_alignment_accuracy(
                extracted_content,
                kp.get("this", ""),  # 从标注数据获取知识点的this字段
            )
            
            if alignment_metrics["precision"] + alignment_metrics["recall"] == 0.:
                kp_alignment_score = 0.
            else:
                kp_alignment_score = 2 * (alignment_metrics["precision"] * alignment_metrics["recall"]) / (alignment_metrics["precision"] + alignment_metrics["recall"])
            # 保存评估结果
            kp.update({
                "extracted": extracted_content,
                "num_exist": alignment_metrics["num_exist"],
                "num_acc": alignment_metrics["num_acc"],
                "total_notes": alignment_metrics["total_notes"],
                "precision": alignment_metrics["precision"],
                "recall": alignment_metrics["recall"],
                "kp_alignment_score" : kp_alignment_score
            })
            item_alignment_score += kp["kp_alignment_score"]
            item_precision += kp["precision"]
            item_recall += kp["recall"]
            # "kp_alignment_score": 0.7 * alignment_metrics["precision"] + 0.3 * alignment_metrics["recall"]

        return item_alignment_score, item_precision, item_recall  # / len(item["knowledge_points"])
    
    def _extract_related_content(self, predict: str, name: str, content: str) -> str:
        """调用GPT-4o提取响应中相关的内容"""
        prompt = (
            f"你是一个几何题判题专家。根据以下知识点从模型响应中提取所有相关的内容(只要有相关内容就提取，可能不是连续出现，你也要正确识别并输出)：\n"
            f"知识点名称：'{name}'\n知识点内容：'{content}'\n\n"
            f"模型响应：'{predict}'\n\n"
            "请直接输出模型响应中与所有该知识点相关的内容"
        )
        # ，包括知识点描述及对应几何元素。无需包含其他无关内容。
        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                response = self.client.chat.completions.create(
                    model=TOOL_NAME,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=500,
                    temperature=self.temperature
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                logging.error(f"内容提取失败: {str(e)}, retrying {attempt}/{self.MAX_RETRIES}")

                # return ""

    def _evaluate_alignment_accuracy(self, response: str, ground_truth: str) -> Dict:
        """评估对应准确性"""
        prompt = (
            "你是一个几何题判题专家，你需要根据标准答案对于某知识点的描述，来对模型生成的描述进行评估。"
            "具体来说，在标准答案中，用<note></note>包裹的内容是重点，你要判断模型响应中关于知识点的"
            "描述是否有这些重点，如果有，是否对应正确，尤其是符号表示（但角ABC和角CBA是同一个角，"
            "类似的问题不应判错）。你最后请直接输出[ans]num_exist, num_acc, ground_total[/ans]，"
            "其中num_exist是标准答案中的重点(即<note></note>包裹的内容想要表示的元素)在模型响应中存在的个数（不管对错），num_acc是存在且对应正确的个数，ground_total是标准答案中<note></note>包裹的重点的总个数\n\n"
            f"标准的描述是：'{ground_truth}'\n"
            f"题目响应中的相关步骤是：{response}"
        )
        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                result = self.client.chat.completions.create(
                    model=TOOL_NAME,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=100,
                    temperature=self.temperature
                )
                data = self._parse_alignment_result(result.choices[0].message.content)
                if data is not None:
                    return data
                else:
                    raise Exception("解析评估结果失败")
            except Exception as e:
                logging.error(f"对应性评估失败: {str(e)} 重试中 {attempt}/{self.MAX_RETRIES}")
                time.sleep(self.RETRY_DELAY)
        print("最大重试次数耗尽") 
        return {"num_exist": 0, "num_acc": 0, "total_notes": 0.0, "precision": 0.0, "recall": 0.0}
    
    def _parse_alignment_result(self, text: str) -> Dict:
        """解析评估结果"""
        match = re.search(r'\[ans\](.+?)\[/ans\]', text)
        if not match:
            logging.error("解析评估结果失败")
            return None
        
        try:
            num_exist, num_acc, ground_total = map(int, match.group(1).split(','))
            recall = num_acc / ground_total if ground_total > 0 else 0.0
            precision = num_acc / num_exist if num_exist > 0 else 0.0
            return {
                "num_exist": num_exist,
                "num_acc": num_acc,
                "total_notes": ground_total,
                "precision": precision,
                "recall": recall
            }
        except Exception as e:
            logging.error(f"解析评估结果失败: {str(e)}")
            return None


    def _evaluate_single_item(self, item: Dict) -> int:
        """评估单个数据项"""
        prompt = self._build_prompt(item["predict"], item["answer"])
        
        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                positive = self._get_positive_count(prompt)
                return int(positive / self.NUM_TRIALS >= self.THRESHOLD)
            except Exception as e:
                logging.warning("评估重试中(%d/%d): %s", attempt, self.MAX_RETRIES, str(e))
                time.sleep(self.RETRY_DELAY)
        
        logging.error("最大重试次数耗尽: %s", item["img_path"])
        return 0

    @staticmethod
    def _build_prompt(predict: str, answer: str) -> str:
        """构建评估提示模板"""
        return (
                f"给定模型的预测: '{predict}' 和正确答案: '{answer}', "
                f"判断模型的预测答案是否正确. 请直接给出明确的回答，用[[ANSWER]]是[[/ANSWER]]或[[ANSWER]]否[[/ANSWER]]包裹。"
            )

    @staticmethod
    def _build_knowledge_prompt(predict: str, name: str, content: str) -> str:
        """构建知识点评测提示模板"""
        return (
            f"请严格判断模型的预测回答是否应用了以下几何知识点：\n"
            f"知识点名称：{name}\n知识点内容：{content}\n\n"
            f"模型回答：'{predict}'\n\n"
            "注意：请用[[ANSWER]]是[[/ANSWER]]或[[ANSWER]]否[[/ANSWER]]回答。"
        )

    def _get_positive_count(self, prompt: str) -> int:
        """获取API响应中的正向判断数量"""
        response = self.client.chat.completions.create(
            model=TOOL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            n=self.NUM_TRIALS,
            temperature=self.temperature
        )
        return sum(
            parse_response(choice.message.content.strip().lower())
            for choice in response.choices
        )

    def _calculate_accuracy(self) -> float:
        """计算准确率"""
        correct = sum(item["answer_score"] for item in self.evaluation_data)
        answer_acc = correct / len(self.evaluation_data)
        # 新增知识点准确率计算
        knowledge_total = sum(item["knowledge_score"] for item in self.evaluation_data)
        knowledge_acc = knowledge_total / len(self.evaluation_data) if self.evaluation_data else 0.0
        # 新增知识点对应准确率计算
        knowledge_alignment_total = sum(item["knowledge_alignment_score"] for item in self.evaluation_data)
        knowledge_alignment_acc = knowledge_alignment_total / len(self.evaluation_data) if self.evaluation_data else 0.0
        # 新增知识点对应精确率
        knowledge_alignment_precision_total = sum(item["knowledge_alignment_precision"] for item in self.evaluation_data)
        knowledge_alignment_precision_acc = knowledge_alignment_precision_total / len(self.evaluation_data) if self.evaluation_data else 0.0
        # 新增知识点对应召回率
        knowledge_alignment_recall_total = sum(item["knowledge_alignment_recall"] for item in self.evaluation_data)
        knowledge_alignment_recall_acc = knowledge_alignment_recall_total / len(self.evaluation_data) if self.evaluation_data else 0.0
        return answer_acc, knowledge_acc, knowledge_alignment_acc, knowledge_alignment_precision_acc, knowledge_alignment_recall_acc

def parse_response(response_text: str) -> bool:
    """解析模型响应"""
    match = re.search(r"\[\[answer\]\](是|否)\[\[/answer\]\]", response_text)
    match = re.search(r'\[\[answer\]\](是|否)\[\[/answer\]\]', response_text)
    return bool(match and match.group(1) == "是")

if __name__ == "__main__":
    # 命令行参数配置（保留原始默认路径）
    # tool_name = "deepseek-v3"
    # tool_name = "gpt-4o-0513"
    parser = argparse.ArgumentParser(description="几何元认知能力评估框架")
    parser.add_argument(
        "--generate_texts_path",
        type=str,
        default="../resoning_results_ch/Qwen2.5-VL-3B-Instruct_output.jsonl",
        help="模型生成结果文件路径"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen2.5-VL-3B-Instruct",
        help="评测的模型名称"
    )
    parser.add_argument(
        "--ground_truth_path",
        type=str,
        default="../data/geosense_ch.jsonl",
        help="标注数据文件路径"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="../eval_results",
        help="评估结果文件路径"
    )

    # 执行评估
    evaluator = MetaGeoEvaluator(parser.parse_args())
