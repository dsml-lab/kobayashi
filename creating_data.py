import numpy as np
import matplotlib.pyplot as plt
import os

# パラメータ設定
height, width, channels = 32, 32, 3
dim = height * width * channels  # 3072次元
n_classes = 10
n_samples_per_class = 100  # 各クラス100枚
total_samples = n_classes * n_samples_per_class

# 共分散行列パラメータ
half_dim = dim // 2
sigma1 = 1.0  # 上位半分の標準偏差
sigma2 = 0.5  # 下位半分の標準偏差

Sigma = np.block([
    [(sigma1**2)*np.eye(half_dim), np.zeros((half_dim, half_dim))],
    [np.zeros((half_dim, half_dim)), (sigma2**2)*np.eye(half_dim)]
])

# 正単体頂点の構築
# N頂点正単体はN点が相互に等距離になる点集合
# 手順:
# 1. N個の標準基底ベクトル e_1, ..., e_N in R^N を用意
# 2. これらの重心を原点に移動する：v_i = e_i - (1/N)*sum(e_j)
# 3. このv_iはN次元にあるが、次元はN-1次元有効（合計は0になるため）
# 4. N=10なら9次元空間に等距離な10点を配置できる

N = n_classes
E = np.eye(N)  # N×N単位行列
mean_all = np.ones((N, N)) * (1/N)
V = E - mean_all  # 各行がv_i
# VはN次元空間内にあるが、行和が0になるため実効的にはN-1次元。
# ここで、Vの行はすべて同じノルムになるよう正規化し、距離を調整する
# 任意の等距離スケーリングを行うため、まず内積を確認
# 各頂点ペア間距離を揃えるために、単純にノルム調整する。

# 現状VはN点(行)をN次元ベクトルとして持っている
# 次元削減: VはN点あるが、N-1次元に埋め込み直す必要がある
# v_iの合計が0ベクトルであることを用いて、N次元 -> N-1次元に射影
# 最後の成分を削るなどしてN-1次元を抽出
# ここでは、Vは既に和が0ベクトルになっているので、
# N個の点はN-1次元部分空間に存在するとみなせる。
# 例えば、特異値分解を用いて次元削減する。
U, S, Wt = np.linalg.svd(V, full_matrices=False)  
# U: N×N, S: N, W: N×N
# 最も小さい特異値は0(Sum to zeroによる制約)
# 上位N-1成分を取り出す
V_reduced = U[:, :N-1] * S[:N-1]  # N×(N-1)
# これでN点がN-1次元空間上にある座標がV_reducedの行として得られた
# このN-1次元で全ペアの距離が等しくなる性質を持つ(正単体の頂点)

# ペア間距離を確認して、スケール調整
# （ここでは特に調整せず、そのまま使用しても等距離性は保たれている。
#  要望があれば距離を特定の値に合わせるため、全点間距離を計算しスケーリングできる。）
# とりあえずそのまま使う
# N-1=9次元空間上の座標
mu_simplex = V_reduced  # shape: (10, 9)

# 高次元(3072)空間への埋め込み
# 単純に上位9軸を使い、それ以外0とする
mu_all = np.zeros((N, dim))
mu_all[:, :9] = mu_simplex  # 前9次元に正単体座標を埋め込む

# データ生成
X = np.zeros((total_samples, dim), dtype=np.float32)
y = np.zeros(total_samples, dtype=np.int32)

idx = 0
for k in range(n_classes):
    mu_k = mu_all[k]
    samples_k = np.random.multivariate_normal(mean=mu_k, cov=Sigma, size=n_samples_per_class)
    X[idx:idx+n_samples_per_class, :] = samples_k
    y[idx:idx+n_samples_per_class] = k
    idx += n_samples_per_class

X_images = X.reshape(total_samples, height, width, channels)

# 0-255へのスケーリング
min_val = X_images.min()
max_val = X_images.max()
X_images_scaled = (X_images - min_val) / (max_val - min_val) * 255.0
X_images_scaled = X_images_scaled.astype(np.uint8)

# 保存用ディレクトリ作成
save_dir = "sample_images_simplex"
os.makedirs(save_dir, exist_ok=True)

# 各クラス1枚プロット・保存
for k in range(n_classes):
    idx_k = np.where(y == k)[0][0]
    img_k = X_images_scaled[idx_k]
    plt.imsave(os.path.join(save_dir, f"class_{k}.png"), img_k)

print("Generated dataset with 10 classes placed as vertices of a regular simplex.")
print("Each class image saved in:", save_dir)
