from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os
import numpy as np


from data_ml import IndictorDataset, MLDataset


def tsne_plot_indicators(indicators, labels):
    tsne = TSNE(n_components=2, random_state=0)
    X_tsne = tsne.fit_transform(indicators)
    plt.figure(figsize=(10, 10))
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels.astype(int), cmap='jet')
    plt.colorbar()
    plt.show()

def demo_of_tsne_plot_indicators():
    csv_root = 'mini_data'
    csv_files = [os.path.join(csv_root, item) for item in os.listdir(csv_root)]
    for i, file in enumerate(csv_files):
        if not file.endswith(".csv"):
            continue
        single_csv_files = [file]
        indictor_dataset = IndictorDataset(single_csv_files)
        indicators = []
        labels = []
        for i in range(len(indictor_dataset)):
            data, label = indictor_dataset[i]
            indicators.append(data)
            labels.append(label)
        indicators = np.array(indicators)
        labels = np.array(labels)
        labels = np.where(labels > 0.03, 1, labels)
        labels = np.where(labels < -0.03, -1, labels)
        labels = np.where(abs(labels) <= 0.3, 0, labels)
        labels = labels.astype(int)
        print(indicators.shape)
        print(labels.shape)
        print(indicators[:10])
        print(labels[:10])

        tsne_plot_indicators(indicators, labels)

def demo_of_tsne_predays_OHLCV():
    csv_root = 'mini_data'
    predays = 15
    future_days = 10
    csv_files = [os.path.join(csv_root, item) for item in os.listdir(csv_root)]
    for i, file in enumerate(csv_files):
        if not file.endswith(".csv"):
            continue
        single_csv_files = [file]
        datas = []
        labels = []
        ml_dataset = MLDataset(single_csv_files, predays, future_days)
        for i in range(len(ml_dataset)):
            data, label = ml_dataset[i]
            datas.append(data)
            labels.append(label)
            print(data.shape)
            print(label.shape)

        datas = np.array(datas)
        labels = np.array(labels)
        print(datas.shape)
        print(labels.shape)
        print(datas[:10])
        print(labels[:10])
        tsne_plot_indicators(datas, labels)
            
            
        

if __name__ == '__main__':
    # demo_of_tsne_plot_indicators()
    demo_of_tsne_predays_OHLCV()