"""clip filter is a tool to use a knn index and a image/text collection to extract interesting subsets"""
import numpy


def average_embedding(csv_path = "../data/artistic_visual_storytelling.csv", feature="artist_name",
                      embedding_path="../embeddings_small/img_emb/img_emb_0.npy", indice_folder="../index_small/", n=20, method='umap', aggregation='avg',
                      write_json=True):
    """Entry point of clip filter"""

    import faiss  # pylint: disable=import-outside-toplevel
    import torch  # pylint: disable=import-outside-toplevel
    from pathlib import Path  # pylint: disable=import-outside-toplevel
    import pandas as pd  # pylint: disable=import-outside-toplevel
    import clip  # pylint: disable=import-outside-toplevel
    import numpy as np

    import rasterfairy
    import umap
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    import json

    features = pd.read_csv(csv_path)
    if feature=='artist_nationality':
        features[feature] = features[feature].str.split(',').str[0]
        unique = features[feature].unique()
        coordinates = []
        dict={}
        for f in unique:
            locations = pd.read_csv('../countries.csv')
            longitude = locations.loc[locations['Demonym'] == f, 'longitude'].iloc[0]
            latitude = locations.loc[locations['Demonym'] == f, 'latitude'].iloc[0]
            coordinates.append([longitude, latitude])

        coordinates = np.asarray(coordinates)
        reduced_data = rasterfairy.transformPointCloud2D(coordinates, proportionThreshold=0.4)[0]
        # center_of_mass = np.mean(coordinates, axis=0)+5
        # coordinates = coordinates - center_of_mass
        # reduced_data = coordinates / np.sqrt(np.absolute(coordinates))**1.2

        if write_json:
            dict={}
            for i, country in enumerate(unique):
                dict[country] = {'x':reduced_data[i][0], 'y':reduced_data[i][1]}

            filename = '../average_embeddings/%s_grid.json' % feature
            with open(filename, 'w') as f:
                json.dump(dict, f, indent=2)
            print('Written to json')

        plt.scatter(reduced_data[:, 0], reduced_data[:, 1])

        for i, txt in enumerate(unique):
            plt.annotate(txt, (reduced_data[i][0], reduced_data[i][1]))
        plt.show()

    else:
        embeddings = np.load(embedding_path)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, _ = clip.load("ViT-B/32", device=device, jit=False)

        if feature=='date':
            features[feature] = features[feature].apply(str).str.slice(stop=3)
        unique = features[feature].unique()
        dict = {}
        for f in unique[:n]:
            ids = features[features[feature]==f].id.tolist()
            aggregate_emb = np.zeros(512)
            if aggregation=='avg':
                for id in ids:
                    aggregate_emb += embeddings[id]
                aggregate_emb = aggregate_emb / len(ids)
            else:
                for id in ids:
                    aggregate_emb = np.maximum(aggregate_emb, embeddings[id])

            if write_json:
                aggregate_emb = str(list(aggregate_emb))
            dict[f] = aggregate_emb


        filename = '../average_embeddings/%s.json' % feature
        if write_json:
            with open(filename, 'w') as f:
                json.dump(dict, f, indent=2)
            print('Written to json')

        else:
            values = list(dict.values())
            keys = list(dict.keys())
            if method=='umap':
                reduced_data = umap.UMAP(n_neighbors=5,
                                      min_dist=0.3,
                                      metric='correlation').fit_transform(values)
            elif method=='pca':
                pca = PCA(n_components=2)
                pca.fit(values)
                reduced_data = pca.transform(values)
            else:
                values = np.stack(values, axis=0)
                reduced_data = TSNE(n_components=2, learning_rate='auto',
                                  init = 'random').fit_transform(values)
            plt.scatter(reduced_data[:, 0], reduced_data[:, 1])
            for i, txt in enumerate(keys):
                plt.annotate(txt, (reduced_data[:,0][i], reduced_data[:,1][i]))

            plt.show()

    return 0


if __name__ == "__main__":
    average_embedding(feature='artist_nationality', method='umap', n=20000, aggregation='max', write_json=True)
