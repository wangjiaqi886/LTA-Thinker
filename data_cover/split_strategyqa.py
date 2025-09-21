import json


def normalize_evidence_field(item):
    """删除不需要的字段"""
    fields_to_remove = ["evidence", "term", "description", "decomposition"]

    for field in fields_to_remove:
        if field in item:
            del item[field]

    if "facts" in item and isinstance(item["facts"], list):
        item["facts"] = "\n".join(item["facts"])

    return item


def split_strategyqa_data():
    # 读取原始JSON文件
    input_file = "/home/wanghy/code/wjq/SoftCoT-main/data/strategyqa/strategyqa_train_origin.json"

    try:
        with open(input_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        print(f"成功读取数据，总共有 {len(data)} 条记录")

        # 标准化数据结构
        normalized_data = []
        for item in data:
            normalized_item = normalize_evidence_field(item.copy())
            normalized_data.append(normalized_item)

        # 计算分割点（8:2比例）
        total_count = len(normalized_data)
        train_count = int(total_count * 0.8)

        # 分割数据
        train_data = normalized_data[:train_count]  # 前80%
        dev_data = normalized_data[train_count:]  # 后20%

        print(f"训练集数据量: {len(train_data)}")
        print(f"验证集数据量: {len(dev_data)}")

        # 保存训练集数据（每行一个JSON对象）
        train_output_file = (
            "/home/wanghy/code/wjq/SoftCoT-main/data/strategyqa/strategyqa_train.jsonl"
        )
        with open(train_output_file, "w", encoding="utf-8") as f:
            for item in train_data:
                json.dump(item, f, ensure_ascii=False, separators=(",", ":"))
                f.write("\n")
        print(f"训练集已保存到: {train_output_file}")

        # 保存验证集数据（每行一个JSON对象）
        dev_output_file = (
            "/home/wanghy/code/wjq/SoftCoT-main/data/strategyqa/strategyqa_dev.jsonl"
        )
        with open(dev_output_file, "w", encoding="utf-8") as f:
            for item in dev_data:
                json.dump(item, f, ensure_ascii=False, separators=(",", ":"))
                f.write("\n")
        print(f"验证集已保存到: {dev_output_file}")

        print("数据分割完成！")

    # except FileNotFoundError:
    #     print(f"错误：找不到文件 {input_file}")
    except json.JSONDecodeError:
        print(f"错误：JSON文件格式不正确")
    except Exception as e:
        print(f"发生错误：{str(e)}")


if __name__ == "__main__":
    split_strategyqa_data()
