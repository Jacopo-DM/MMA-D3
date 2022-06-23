"""clip filter is a tool to use a knn index and a image/text collection to extract interesting subsets"""


def clip_query(query, indice_folder="../index/", num_results=10, threshold=None):
    """Entry point of clip filter"""
    import matplotlib.pyplot as plt

    import faiss  # pylint: disable=import-outside-toplevel
    import torch  # pylint: disable=import-outside-toplevel
    from pathlib import Path  # pylint: disable=import-outside-toplevel
    import pandas as pd  # pylint: disable=import-outside-toplevel
    import clip  # pylint: disable=import-outside-toplevel
    from sklearn.decomposition import PCA

    import umap

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = clip.load("ViT-B/32", device=device, jit=False)

    data_dir = Path(indice_folder + "/metadata")
    df = pd.concat(pd.read_parquet(parquet_file) for parquet_file in sorted(data_dir.glob("*.parquet")))

    url_list = None
    if "url" in df:
        url_list = df["url"].tolist()

    image_list = df["image_path"].tolist()
    image_index = faiss.read_index(indice_folder + "/image.index")
    indices_loaded = {
        "image_list": image_list,
        "image_index": image_index,
    }

    text_input = query
    image_index = indices_loaded["image_index"]
    image_list = indices_loaded["image_list"]
    #if not os.path.exists(output_folder):
    #    os.mkdir(output_folder)

    text = clip.tokenize([text_input]).to(device)
    text_features = model.encode_text(text)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    query = text_features.cpu().detach().numpy().astype("float32")

    index = image_index

    if threshold is not None:
        _, d, i = index.range_search(query, threshold)
        print(f"Found {i.shape} items with query '{text_input}' and threshold {threshold}")
    else:
        d, i = index.search(query, num_results)
        print(f"Found {num_results} items with query '{text_input}'")
        i = i[0]
        d = d[0]

    result=[]
    paths=[]
    scores=[]
    vectors=[]

    for score, ei in zip(d, i):
        path = image_list[ei]
        vectors.append(index.reconstruct(int(ei)))
        paths.append(path)
        scores.append(score)


    umap_result = umap.UMAP(n_neighbors=5,
                          min_dist=0.3,
                          metric='correlation').fit_transform(vectors)

    #pca = PCA(n_components=2)
    #pca.fit(vectors)
    #pca_result = pca.transform(vectors)

    plt.scatter(umap_result[:,0], umap_result[:,1])
    plt.show()

    result = list(zip(paths, scores, umap_result))

    return result


if __name__ == "__main__":
    print(clip_query("animals", num_results=30))
