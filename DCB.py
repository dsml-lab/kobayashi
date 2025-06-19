# calculate digit and color bias
import torch

# デバイスの設定 (GPU が利用可能な場合は GPU を使用)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# モデルの定義とインスタンス化
model = ResNet18(num_classes=10)  # クラス数を10に設定

# 学習済みパラメータの読み込み
model.load_state_dict(torch.load("model_weights.pth"))  # pthファイルの名前を指定
model.to(device)
model.eval()  # 評価モードに設定

# 入力xの定義 (例: ランダムな入力)
x = torch.randn(1, 3, 224, 224).to(device)  # バッチサイズ1, チャンネル数3, 画像サイズ224x224

# 隠れ層の出力の取得
class Identity(nn.Module):
    def forward(self, x):
        return x

# 最終畳み込み層の直後にIdentity層を追加
model.avgpool = Identity()

# 隠れ層表現zの計算
with torch.no_grad():
    z = model(x)

print(z.shape)  # 隠れ層表現の形状を出力
