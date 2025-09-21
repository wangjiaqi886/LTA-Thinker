import json
import os


def convert_date_understanding_data(input_file, output_file):
    """
    读取日期理解数据集JSON文件，并转换为指定格式

    Args:
        input_file (str): 输入JSON文件路径
        output_file (str): 输出JSON文件路径
    """
    # 检查输入文件是否存在
    if not os.path.exists(input_file):
        print(f"错误: 文件 {input_file} 不存在")
        return []

    try:
        # 读取原始JSON文件
        with open(input_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        print(f"成功读取文件: {input_file}")
        print(f"数据类型: {type(data)}")

        # 转换数据格式
        converted_data = []

        # 检查数据结构
        if isinstance(data, dict):
            if "examples" in data:
                examples = data["examples"]
                print(f"找到 examples 字段，包含 {len(examples)} 条数据")
            else:
                print("未找到 examples 字段，尝试处理整个字典")
                examples = [data] if "input" in data and "target_scores" in data else []
        elif isinstance(data, list):
            examples = data
            print(f"数据为列表格式，包含 {len(examples)} 条数据")
        else:
            print(f"未知的数据格式: {type(data)}")
            return []

        # 创建输出目录
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        # 打开输出文件，逐行写入
        with open(output_file, "w", encoding="utf-8") as f:
            # 处理每个示例
            for i, example in enumerate(examples):
                try:
                    # 获取问题
                    # question = example.get("input", "")

                    # # 从target_scores中找到值为1的答案
                    # target_scores = example.get("target_scores", {})
                    # answer = None

                    # for key, value in target_scores.items():
                    #     if value == 1:
                    #         answer = key
                    #         break

                    # 如果找到答案，写入文件
                    # converted_item = {"question": question, "answer": answer}
                    converted_item = process_single_example(example)
                    # 将每个JSON对象写入一行
                    f.write(json.dumps(converted_item, ensure_ascii=False) + "\n")
                    converted_data.append(converted_item)

                except Exception as e:
                    print(f"处理第 {i+1} 条数据时出错: {e}")
                    continue

        print(f"成功转换 {len(converted_data)} 条数据")
        print(f"数据已保存到: {output_file}")

        return converted_data

    except json.JSONDecodeError as e:
        print(f"JSON解析错误: {e}")
        return []
    except Exception as e:
        print(f"处理文件时出错: {e}")
        return []


def process_single_example(example_dict):
    """
    处理单个示例数据

    Args:
        example_dict (dict): 包含input和target_scores的字典

    Returns:
        dict: 转换后的格式 {"question": ..., "answer": ...}
    """
    question = example_dict.get("input", "")
    target_scores = example_dict.get("target_scores", {})

    # 处理问题文本：删除 " in MM/DD/YYYY" 并添加格式要求
    if " in MM/DD/YYYY" in question:
        question = question.replace(" in MM/DD/YYYY", "")

    # 确保问题以 "?" 结尾
    if not question.endswith("?"):
        question += "?"

    # 添加格式要求
    question += "\n\nChoose the correct answer from the following options:"

    # 添加选项并找到正确答案的选项字母
    options = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
    option_index = 0
    answer = None
    correct_option = None

    for key in target_scores.keys():
        if option_index < len(options):
            question += f"\n{options[option_index]}) {key}"
            # 如果这个选项的值为1，记录答案和对应的选项字母
            if target_scores[key] == 1:
                answer = key
                correct_option = options[option_index]
            option_index += 1

    if answer is not None and correct_option is not None:
        answer = (
            "Based on the reference date and the time offset provided in the question, the target date is calculated by adjusting accordingly.\n"
            f"{answer}\n"
            f"{correct_option}"
        )
        return {"question": question, "answer": answer}
    else:
        return None


if __name__ == "__main__":
    # 指定文件路径
    input_file = "/home/wanghy/code/wjq/SoftCoT-main/data/du/task.json"
    output_file = (
        "/home/wanghy/code/wjq/SoftCoT-main/data/du/date_understanding_gsm_style.jsonl"
    )

    # 转换数据
    print("\n开始转换数据...")
    converted_data = convert_date_understanding_data(input_file, output_file)

    print(f"\n处理完成！共转换 {len(converted_data)} 条数据")
