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
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.unicode_minus"] = False

# 数据
latent_tokens = ["L-1", "L-2", "L-3", "L-4", "L-6", "L-8"]
accuracy = [90.90, 93.25, 92.65, 92.42, 91.36, 88.10]

# 创建图形
plt.figure(figsize=(10, 4))
plt.plot(
    latent_tokens, accuracy, marker="o", linewidth=2, markersize=8, color="#2E86AB"
)

# 设置图表样式
# plt.title(
#     "The impact of different numbers of latent thought tokens",
#     fontsize=26,
#     fontweight="bold",
#     pad=18,
# )
plt.xlabel(
    "Numbers of Latent Thought Tokens", fontsize=26, fontfamily="Times New Roman"
)
plt.ylabel("Acc (%)", fontsize=26, fontfamily="Times New Roman")

# 设置y轴范围，突出显示数据差异
plt.ylim(87, 95)
plt.xlim(-0.5, len(latent_tokens) - 0.25)
plt.xticks(fontsize=24, fontfamily="Times New Roman")
plt.yticks(fontsize=24, fontfamily="Times New Roman")
ax = plt.gca()  # 获取当前坐标轴
ax.yaxis.offsetText.set_fontsize(24)
ax.yaxis.offsetText.set_fontfamily("Times New Roman")
# 添加网格
plt.grid(True, alpha=0.3, linestyle="--")

# 在每个数据点上标注数值
for i, v in enumerate(accuracy):
    plt.annotate(
        f"{v:.2f}%",
        (i, v),
        textcoords="offset points",
        xytext=(0, 10),
        ha="center",
        fontsize=26,
        fontfamily="Times New Roman",
    )

# 调整布局
plt.tight_layout()

# 保存图片到当前目录
plt.savefig("./data_cover/latent_token.png", dpi=300, bbox_inches="tight")
plt.savefig("./data_cover/latent_token.pdf", bbox_inches="tight")
# 显示图表
plt.show()

print("图片已保存到当前目录: latent_token.png")
