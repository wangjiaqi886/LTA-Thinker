import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties
from matplotlib import font_manager
import matplotlib

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
# download the font files and save in this fold
font_path = "../tsfont"

font_files = font_manager.findSystemFonts(fontpaths=font_path)

for file in font_files:
    font_manager.fontManager.addfont(file)
# 数据
epochs = list(range(1, 11))

# Liner-Assistant的loss数据
liner_assistant_loss = [
    0.002515746,
    0.001731327,
    0.000869192,
    0.00050563,
    0.001073996,
    0.000735236,
    0.001298478,
    0.000884472,
    0.000921036,
    0.000860909,
]

# Transformer Block的loss数据
transformer_block_loss = [
    0.001856923,
    0.000713944,
    0.000565388,
    0.000206103,
    0.000132024,
    9.27e-05,
    0.000109873,
    7.83e-05,
    0.00010265,
    6.66e-05,
]

# 设置中文字体
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.unicode_minus"] = False  # 用来正常显示负号

# 创建图形
plt.figure(figsize=(10, 4))

# 绘制折线图
plt.plot(
    epochs,
    liner_assistant_loss,
    marker="o",
    linewidth=2,
    label="Liner-Assistant",
    color="blue",
)
plt.plot(
    epochs,
    transformer_block_loss,
    marker="s",
    linewidth=2,
    label="Transformer Block",
    color="red",
)

# 设置图形属性
# plt.title(
#     "Comparison of Eval-Loss during model training", fontsize=26, fontweight="bold"
# )
plt.xlabel("Epoch", fontsize=26, fontfamily="Times New Roman")
plt.ylabel("Eval-Loss", fontsize=26, fontfamily="Times New Roman")
plt.grid(True, alpha=0.3)
plt.legend(fontsize=26, prop={"family": "Times New Roman", "size": 26})

# 设置坐标轴
plt.xlim(0.5, 10.5)
plt.xticks(epochs)
plt.xticks(fontsize=24, fontfamily="Times New Roman")
plt.yticks(fontsize=24, fontfamily="Times New Roman")
# 使用科学记数法显示y轴
plt.ticklabel_format(style="scientific", axis="y", scilimits=(0, 0))
ax = plt.gca()  # 获取当前坐标轴
ax.yaxis.offsetText.set_fontsize(24)
ax.yaxis.offsetText.set_fontfamily("Times New Roman")
# 调整布局
plt.tight_layout()

# 保存图片到当前目录
plt.savefig("./data_cover/eval_loss_comparison.png", dpi=300, bbox_inches="tight")
plt.savefig("./data_cover/eval_loss_comparison.pdf", bbox_inches="tight")

# 显示图形
plt.show()

print("折线图已保存到当前目录：")
print("- eval_loss_comparison.png")
print("- eval_loss_comparison.pdf")
