import { UMAP } from 'umap-js';
import * as json_data from './artist_name.json' assert {type: "json"};
import * as request from './request.json' assert {type: "json"};

const embeddings = json_data['default'];

const features = request['default']["features_to_cluster"].split(', ');

var keys = Object.keys(embeddings);

var arr = Array.from(Array(features.length), () => new Array(512))

for (let i = 0; i < features.length; i++) {
    console.log(features[i])
    console.log(embeddings[features[i]])
    arr[i]=JSON.parse(embeddings[features[i]]);
  }

const umap = new UMAP({
  nComponents: 2,
  nEpochs: 400,
  nNeighbors: 5,
});
console.log('umap start')

const result = umap.fit(arr);

console.log('umap finish')
console.log(result)
