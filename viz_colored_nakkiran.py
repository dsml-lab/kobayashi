import os
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
import matplotlib.colors as mcolors
import numpy as np
from matplotlib.colors import LogNorm, Normalize
from matplotlib.ticker import LogLocator, MultipleLocator, FixedLocator
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
import matplotlib.cm as cm

# ヒートマップ作成
def generate_cmap(colors, cmap_name = 'custom_cmap'):
    values = range(len(colors))
    vmax = np.ceil(np.max(values))
    color_list = []
    for vi, ci in zip(values, colors):
        color_list.append( ( vi/ vmax, ci) )

    return mcolors.LinearSegmentedColormap.from_list(cmap_name, color_list, 256)

def plot_learning_curves(base_dir, n_values, m_values):
    """
    df: 列名が ["student epoch", "teacher epoch", "test_error"] のDataFrameを想定。
        - "teacher epoch": 横軸
        - "test_error"   : 縦軸
        - "student epoch": カラーバー(各ラインの色分け)
    """
    tsv_file_path1 = './save_data_sdb/teacher/preact-resnet50_3/train.log.tsv'
    tsv_file_path2 = './save_data_sdb/teacher/noisy-preact-resnet50-t/train.log.tsv'
    tsv_file_path3 = './save_data_sdb/teacher/noisy40%-preact-resnet50-t/train.log.tsv'
    tsv_file_path4 = './save_data_sdb/teacher/noisy60%-preact-resnet50-t/train.log.tsv'
    tsv_file_path5 = './save_data_sdb/teacher/noisy80%-preact-resnet50-t/train.log.tsv'
    tsv_file_path6 = './save_data_sdb/teacher/noisy100%-preact-resnet50-t/train.log.tsv'

    data = pd.read_csv(tsv_file_path1, sep='\t')
    train_data = data[data['split'] == 'train']
    test_data = data[data['split'] == 'test']
    train_data['epoch'] = train_data['epoch'].astype(int)
    test_data['epoch'] = test_data['epoch'].astype(int)

    data2 = pd.read_csv(tsv_file_path2, sep='\t')
    train_data2 = data2[data2['split'] == 'train']
    test_data2 = data2[data2['split'] == 'test']
    train_data2['epoch'] = train_data2['epoch'].astype(int)
    test_data2['epoch'] = test_data2['epoch'].astype(int)

    data3 = pd.read_csv(tsv_file_path3, sep='\t')
    train_data3 = data3[data3['split'] == 'train']
    test_data3 = data3[data3['split'] == 'test']
    train_data3['epoch'] = train_data3['epoch'].astype(int)
    test_data3['epoch'] = test_data3['epoch'].astype(int)

    data4 = pd.read_csv(tsv_file_path4, sep='\t')
    train_data4 = data4[data4['split'] == 'train']
    test_data4 = data4[data4['split'] == 'test']
    train_data4['epoch'] = train_data4['epoch'].astype(int)
    test_data4['epoch'] = test_data4['epoch'].astype(int)

    data5 = pd.read_csv(tsv_file_path5, sep='\t')
    train_data5 = data5[data5['split'] == 'train']
    test_data5 = data5[data5['split'] == 'test']
    train_data5['epoch'] = train_data5['epoch'].astype(int)
    test_data5['epoch'] = test_data5['epoch'].astype(int)

    data6 = pd.read_csv(tsv_file_path6, sep='\t')
    train_data6 = data6[data6['split'] == 'train']
    test_data6 = data6[data6['split'] == 'test']
    train_data6['epoch'] = train_data6['epoch'].astype(int)
    test_data6['epoch'] = test_data6['epoch'].astype(int)

    # "Loss"のデータフレームを作成（任意：6つのモデルのLossをまとめたテーブル）
    # loss_df = pd.DataFrame({
    #     'epoch': test_data['epoch'],
    #     'LN 0%': test_data['loss'],
    #     'LN 20%': test_data2['loss'],
    #     'LN 40%': test_data3['loss'],
    #     'LN 60%': test_data4['loss'],
    #     'LN 80%': test_data5['loss'],
    #     'LN 100%': test_data6['loss']
    # })
    loss_df = pd.DataFrame({
        'epoch': test_data['epoch'],
        'LN 0%': test_data['accuracy'],
        'LN 20%': test_data2['accuracy'],
        'LN 40%': test_data3['accuracy'],
        'LN 60%': test_data4['accuracy'],
        'LN 80%': test_data5['accuracy'],
        'LN 100%': test_data6['accuracy']
    })

    plt.figure(figsize=(16, 10.4))
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams["font.size"] = 24
    norm = plt.Normalize(min(m_values), max(m_values)/4)
    colormap = cm.viridis
    heatmap_data = []
    num = 0
    
    # データ読み込み・ラインプロット部分
    for n, m in zip(n_values, m_values):
        num += 40
        dir_name = os.path.join(
            base_dir, 
            f"{n}%_teacher_{m}", 
            "50000-64"
        )
        csv_file = os.path.join(dir_name, "train.50000.log.tsv")
        
        if not os.path.exists(csv_file):
            print(f"CSVファイルが見つかりません: {csv_file}")
            continue
        
        data = pd.read_csv(csv_file, sep='\t')
        # テストデータのみ抽出
        data = data[data['split'] == 'test']
        data = data[data['epoch'] <= 200]
        # 精度を0～1表記に
        data['accuracy'] =  data['accuracy']/100
        
        if 'epoch' not in data.columns or 'loss' not in data.columns or 'accuracy' not in data.columns:
            print(f"CSVのフォーマットが正しくありません: {csv_file}")
            continue
        
        # ヒートマップ用データに追加(行: student epoch, 列: teacher epoch, 値: accuracy)
        for epoch_val, accuracy_val in zip(data['epoch'], data['loss']):
            heatmap_data.append([epoch_val, m, accuracy_val])
        
        color = colormap(num)
        
        # ラインプロット（x軸ログスケールはしない）
        plt.plot(data['epoch'], data['accuracy'], label=f'teacher_epoch={m}', color=color)
        # plt.plot(data['epoch'], data['accuracy'], label=f'teacher_epoch={m}', lw = 2.5)
    
    plt.xlabel("epoch", fontsize=26, labelpad=15)
    plt.ylabel("test error", fontsize=26, labelpad=15)
    # plt.legend()
    plt.grid(True)
    # plt.xscale('log')
    plt.tight_layout()
    plt.xlim(0,200)
    plt.savefig("test_lineplot.png")
    plt.close()
    cmthermal = generate_cmap(['#1c3f75', '#068fb9','#f1e235', '#d64e8b', '#730e22'], 'cmthermal')

    if heatmap_data:
        plt.figure(figsize=(16, 10.4))
        
        heatmap_df = pd.DataFrame(heatmap_data, columns=["student epoch", "teacher epoch", "accuracy"])
        
        # pivot: 行＝student epoch (y軸), 列＝teacher epoch (x軸)
        heatmap_pivot = heatmap_df.pivot(index="student epoch", columns="teacher epoch", values="accuracy")
        
        # 昇順ソート
        heatmap_pivot = heatmap_pivot.sort_index(axis=0)
        heatmap_pivot = heatmap_pivot[sorted(heatmap_pivot.columns)]
        
        # 軸ラベルとして使う値を取得
        print(heatmap_df.head())
        print(heatmap_pivot)
        x_vals = heatmap_pivot.columns.values  # 教師モデルエポック
        y_vals = heatmap_pivot.index.values    # 生徒モデルエポック
        
        # pcolormesh用にセル境界座標を作成
        # 行数, 列数
        ny, nx = heatmap_pivot.shape
        
        # x方向は 0～nx の整数、y方向は 0～ny の整数をセル境界とする
        X_edges, Y_edges = np.meshgrid(np.arange(nx), np.arange(ny))
        
        vmin = heatmap_pivot.values.min()
        vmax = heatmap_pivot.values.max()
        print(vmin,vmax)
        
        fig, ax = plt.subplots(figsize=(16, 10.4))
        pcm = ax.pcolormesh(
            X_edges, Y_edges, heatmap_pivot.values,
            # cmap=cmthermal,x
            cmap="magma_r",
            # vmax=0.9
            # norm=LogNorm(vmin=vmin, vmax=vmax),
            shading='auto'
        )

        cbar = fig.colorbar(pcm, ax=ax)
        cbar.set_label("test error")
        allticks = cbar.ax.yaxis.get_majorticklocs()
        cbar.ax.yaxis.set_minor_locator(FixedLocator(allticks))
        cbar.update_ticks()

        # plt.title("Accuracy Heatmap", fontsize=26)
        plt.xlabel("teacher model checkpoint", fontsize=24)
        plt.ylabel("student model epoch", fontsize=24)
        plt.tight_layout()

        # Y軸をlogスケール（必要であれば）
        ax.set_yscale('log')
        ax.set_yticks([1, 10, 100, 200])
        ax.set_yticklabels([1, 10, 100, 200])

        # x軸とy軸のラベルを実際の値に対応させる
        # セルはインデックス0から数えているため、Tickはセル中央を指すように0.5オフセットを与える
        ax.set_xticks(np.arange(nx))
        ax.set_xticklabels(x_vals)  # ここで教師エポック(m_values)をそのまま表示
        # ax.set_yticks(np.arange(ny) + 0.5)
        # ax.set_yticklabels(y_vals)
        plt.ylim(1,200)

        plt.savefig("heatmap.png")
        plt.close()
    # ======================================================
    # (C) contourf を使ったヒートマップ相当
    # ======================================================
    # contourf の場合は 「実座標 X, Y, Z」 をそのまま使う形がおすすめ
    
    # まずは 1次元の x, y を meshgrid 化
    #  - x_unique = shape (nx,)
    #  - y_unique = shape (ny,)
    #  => X, Y = shape (ny, nx)
    Z = heatmap_pivot.values  # shape = (ny, nx)
    y_unique = heatmap_pivot.index.values    # 学習(epoch)のユニーク値(行)
    x_unique = heatmap_pivot.columns.values  # 教師(epoch)のユニーク値(列)
    ny, nx = Z.shape
    X, Y = np.meshgrid(x_unique, y_unique)  # X: teacher epoch, Y: student epoch

    plt.close()

    """
    df は列に ["student_epoch", "teacher_epoch", "test_error"] を含むDataFrameを想定。
    
    - ヒートマップの描画方法にならい、
      pivot(index="student_epoch", columns="teacher_epoch")
      →  行=student epoch, 列=teacher epoch (shape=(ny,nx)) にする
    
    - 横軸は [0..nx-1] という離散インデックスでプロットし、
      軸ラベルだけを 実際の「教師エポック」(teacher_epoch) に設定する。

    - 各 row(=ある1つの student_epoch) について、列方向にラインを引く
      (x=[0..nx-1], y=pivot_dfのrow)

    - ラインの色は「student_epoch」に応じたカラーマップを用いる
    """

    # pivot: 行=student_epoch, 列=teacher_epoch, 値=test_error
    pivot_df = heatmap_df.pivot(index="student epoch", columns="teacher epoch", values="accuracy")

    # 行・列の昇順ソート（ヒートマップと同じ処理）
    pivot_df = pivot_df.sort_index(axis=0)         # student_epoch 昇順
    pivot_df = pivot_df[sorted(pivot_df.columns)] # teacher_epoch 昇順

    # 行数、列数
    ny, nx = pivot_df.shape

    # X軸に対応する教師エポックのリスト（列のユニーク値）
    x_vals = pivot_df.columns.values  # 例: array([1,2,3,5,10,15,...])
    # Y軸(行)に対応する生徒エポックのリスト
    y_vals = pivot_df.index.values    # 例: array([1,5,10,20,100,...])

    # pivot_df は (ny, nx)
    Z = pivot_df.values  # テスト誤差

    # Figure
    fig, ax1= plt.subplots(figsize=(16, 10.4))
    ax2 = ax1.twinx()

    # カラーマップを設定 (例：生徒エポックに対してログスケールで割り当て)
    norm = Normalize(vmin=y_vals.min(), vmax=y_vals.max())
    cmap = cm.viridis_r

    # 各行(= 1つの student_epoch) ごとにラインを引く
    #  x は 0～(nx-1) の離散インデックス
    #  y は Z[row,:]
    #  カラーは row(生徒エポックの実値)を用いて決める
    for i, s_epoch in enumerate(y_vals):
        # i番目の行が Z[i,:]
        # x軸は離散 [0..nx-1]
        x_plot = np.arange(nx)
        y_plot = Z[i, :]

        color = cmap(norm(s_epoch))
        # color = cmap(s_epoch)
        ax1.plot(x_plot, y_plot, color=color, lw=0.8, alpha=1.0, zorder=1)

    # test_data をプロット（X軸を np.arange(nx) にマッピング）
    if 'epoch' in test_data3.columns and 'accuracy' in test_data3.columns:
        # test_data の教師エポック値を x_vals にマッピング
        print(x_vals)
        test_data3 = test_data3[test_data3['epoch'].isin(x_vals)].copy()
        test_data3['x_index'] = test_data3['epoch'].apply(lambda e: np.where(x_vals == e)[0][0])  # x_vals のインデックスに変換

        # ax2.plot(test_data3['x_index'], 1 - test_data3['accuracy'] / 100,
        #         color="red", marker=".", lw=3.0, ms=20,  label='teacher test error', zorder=3)

    # 軸ラベルを設定
    # ax.set_yscale('log')
    ax1.set_xlabel("teacher model checkpoint", fontsize=26, labelpad=10)
    ax1.set_ylabel("student test agreement", fontsize=26,labelpad=10)
    # ax2.set_ylabel("teacher test error", fontsize=26 ,labelpad=10)
    # ax.set_xlabel("teacher model epoch (discrete)", fontsize=26)
    # ax.set_ylabel("test error", fontsize=26)
    # ax.set_title("Test Error vs. Teacher Epoch (Color: Student Epoch)", fontsize=16)

    # x 軸を離散表示: 0～nx-1 に対して、ラベルに x_vals(教師エポック) を付与
    ax1.set_xticks(np.arange(nx))
    ax1.set_yscale('log')
    ax1.set_xticklabels(x_vals, rotation=45)
    # ax2.legend()

    # カラーバーを設定（student_epoch の範囲）
        # カラーバー作成
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # ダミーアレイ
    print(sm)
    cbar = fig.colorbar(sm, ax=ax2, pad=0.1, aspect=30, location="right") 
    cbar.set_label("student model epoch", labelpad=10, fontsize=26)

    # ======= ここで colorbar の tick を明示的に設定 =======
    # 例: 生徒エポックが1～200のログスケールなら、主要な目盛をこう置く
    # tick_positions = [1, 2, 5, 10, 20, 50, 100, 200]
    # cbar.set_ticks(tick_positions)
    # ラベル文字列も必要...