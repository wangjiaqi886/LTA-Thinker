import json
import re


def convert_aqua_to_gsm8k_format(input_file, output_file):
    """
    将 dev.tok 文件中的数据转换为 dev_socratic 的数据结构

    Args:
        input_file: 输入文件路径 (dev.tok)
        output_file: 输出文件路径 (dev_socratic)
    """

    def process_rationale(rationale, correct_answer, options):
        """
        处理 rationale，将其转换为 GSM8K 格式的答案
        """
        # 清理 rationale 文本
        rationale = rationale.strip()

        # 将 rationale 按行分割，形成步骤
        lines = [line.strip() for line in rationale.split("\n") if line.strip()]

        # 构建答案字符串
        answer_parts = []

        for i, line in enumerate(lines):
            if line.startswith("Correct answer"):
                continue

            # 清理每一行，移除多余的空格和符号
            clean_line = re.sub(r"\s+", " ", line)
            clean_line = clean_line.replace("( ", "(").replace(" )", ")")

            answer_parts.append(clean_line)

        # 组合最终答案，正确答案改为选项形式
        final_answer = "\n".join(answer_parts)
        final_answer += f"\n#### {correct_answer}"

        return final_answer

    def format_question_with_options(question, options):
        """
        将选项添加到问题末尾
        """
        formatted_question = question.strip()
        if not formatted_question.endswith(".") and not formatted_question.endswith(
            "?"
        ):
            formatted_question += "."

        formatted_question += (
            "\n\nChoose the correct answer from the following options:"
        )
        for option in options:
            formatted_question += f"\n{option}"

        return formatted_question

    # 读取输入文件
    converted_data = []

    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)

                # 转换数据结构
                converted_item = {
                    "question": format_question_with_options(
                        data["question"], data["options"]
                    ),
                    "answer": process_rationale(
                        data["rationale"], data["correct"], data["options"]
                    ),
                }

                converted_data.append(converted_item)

            except json.JSONDecodeError as e:
                print(f"跳过无效的JSON行: {line[:50]}... 错误: {e}")
                continue

    # 写入输出文件
    with open(output_file, "w", encoding="utf-8") as f:
        for item in converted_data:
            json.dump(item, f, ensure_ascii=False)
            f.write("\n")

    print(f"转换完成！共处理 {len(converted_data)} 条数据")
    print(f"输出文件: {output_file}")


def convert_with_detailed_steps(input_file, output_file):
    """
    更详细的转换函数，尝试保持原有的推理步骤结构
    """

    def format_answer_with_steps(rationale, correct_answer, options):
        """
        将 rationale 格式化为带有明确步骤的答案
        """
        lines = [line.strip() for line in rationale.split("\n") if line.strip()]

        # 构建步骤化的答案
        answer_steps = []
        step_counter = 1

        for line in lines:
            if line.startswith("Correct answer"):
                continue

            # 清理和格式化每一步
            clean_line = re.sub(r"\s+", " ", line.strip())

            # 如果行包含计算，尝试提取并格式化
            if "=" in clean_line and any(char.isdigit() for char in clean_line):
                # 尝试找到计算表达式
                calc_match = re.search(r"(.+?)\s*=\s*(.+)", clean_line)
                if calc_match:
                    expression = calc_match.group(1).strip()
                    result = calc_match.group(2).strip()

                    # 格式化为 GSM8K 风格
                    if "<<" not in clean_line:
                        formatted_line = (
                            f"{expression} = <<{expression}={result}>>{result}"
                        )
                        answer_steps.append(formatted_line)
                    else:
                        answer_steps.append(clean_line)
                else:
                    answer_steps.append(clean_line)
            else:
                answer_steps.append(clean_line)

            step_counter += 1

        # 组合最终答案，正确答案改为选项形式
        final_answer = "\n".join(answer_steps)
        final_answer += f"\n#### {correct_answer}"

        return final_answer

    def format_question_with_options(question, options):
        """
        将选项添加到问题末尾
        """
        formatted_question = question.strip()
        if not formatted_question.endswith(".") and not formatted_question.endswith(
            "?"
        ):
            formatted_question += "."

        formatted_question += (
            "\n\nChoose the correct answer from the following options:"
        )
        for option in options:
            formatted_question += f"\n{option}"

        return formatted_question

    converted_data = []

    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)

                converted_item = {
                    "question": format_question_with_options(
                        data["question"], data["options"]
                    ),
                    "answer": format_answer_with_steps(
                        data["rationale"], data["correct"], data["options"]
                    ),
                }

                converted_data.append(converted_item)

            except json.JSONDecodeError as e:
                print(f"跳过无效的JSON行: {line[:50]}... 错误: {e}")
                continue

    with open(output_file, "w", encoding="utf-8") as f:
        for item in converted_data:
            json.dump(item, f, ensure_ascii=False)
            f.write("\n")

    print(f"详细转换完成！共处理 {len(converted_data)} 条数据")


# 使用示例
if __name__ == "__main__":
    input_file = "/home/wanghy/code/wjq/SoftCoT-main/data/AQuA-master/dev.tok.json"  # 输入文件路径
    output_file = "/home/wanghy/code/wjq/SoftCoT-main/data/AQuA-master/gsm_style_dev.jsonl"  # 输出文件路径

    # 基础转换
    convert_aqua_to_gsm8k_format(input_file, output_file)

    input_file = "/home/wanghy/code/wjq/SoftCoT-main/data/AQuA-master/test.tok.json"  # 输入文件路径
    output_file = "/home/wanghy/code/wjq/SoftCoT-main/data/AQuA-master/gsm_style_test.jsonl"  # 输出文件路径

    # 基础转换
    convert_aqua_to_gsm8k_format(input_file, output_file)

    input_file = "/home/wanghy/code/wjq/SoftCoT-main/data/AQuA-master/train.tok.json"  # 输入文件路径
    output_file = "/home/wanghy/code/wjq/SoftCoT-main/data/AQuA-master/gsm_style_train.jsonl"  # 输出文件路径

    # 基础转换
    convert_aqua_to_gsm8k_format(input_file, output_file)

    # 或者使用更详细的转换
    # convert_with_detailed_steps(input_file, output_file + "_detailed")
