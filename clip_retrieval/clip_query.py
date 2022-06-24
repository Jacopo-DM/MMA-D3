"""clip filter is a tool to use a knn index and a image/text collection to extract interesting subsets"""


def query(query_text, indice_folder="../index_small/", num_results=10, threshold=None, plot=False):
    """Entry point of clip filter"""
    import matplotlib.pyplot as plt

    import faiss  # pylint: disable=import-outside-toplevel
    import torch  # pylint: disable=import-outside-toplevel
    from pathlib import Path  # pylint: disable=import-outside-toplevel
    import pandas as pd  # pylint: disable=import-outside-toplevel
    import clip  # pylint: disable=import-outside-toplevel

    import umap
    import uuid
    import re

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

    image_index = indices_loaded["image_index"]
    image_list = indices_loaded["image_list"]
    #if not os.path.exists(output_folder):
    #    os.mkdir(output_folder)

    text = clip.tokenize([query_text]).to(device)
    text_features = model.encode_text(text)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    query = text_features.cpu().detach().numpy().astype("float32")

    index = image_index

    if threshold is not None:
        _, d, i = index.range_search(query, threshold)
        print(f"Found {i.shape} items with query '{query_text}' and threshold {threshold}")
    else:
        d, i = index.search(query, num_results)
        print(f"Found {num_results} items with query '{query_text}'")
        i = i[0]
        d = d[0]
    results = {}
    results['id']= str(uuid.uuid4())
    results['query'] = query_text

    paths=[]
    scores=[]
    vectors=[]


    for score, ei in zip(d, i):
        path = image_list[ei]
        vectors.append(index.reconstruct(int(ei)))
        paths.append(path)
        scores.append(str(score))


    umap_result = umap.UMAP(n_neighbors=5,
                          min_dist=0.3,
                          metric='correlation').fit_transform(vectors)

    for i in range(num_results):
        result={}
        result['img_id'] = re.findall(r'\d+', paths[i])[-1]
        result['score'] = str(scores[i])
        result['x'] = str(umap_result[i][0])
        result['y'] = str(umap_result[i][0])

        results[str(i)] = result

    if plot:
        plt.scatter(umap_result[:,0], umap_result[:,1])
        plt.show()


    return results


if __name__ == "__main__":
    print(query("animals", num_results=5, plot=True))
