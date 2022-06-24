import asyncio
import json
import uuid
from watchfiles import arun_process
from clip_retrieval import clip_query

def process_query(a, b, c):
    with open('jsons/requests/request.json', 'r') as j:
        args = json.loads(j.read())
    print('Your query is: %s' %args['query'])
    result = clip_query.query(args['query'], num_results = args['n_results'], indice_folder="index_small")
    with open('jsons/result.json', 'w') as f:
        json.dump(result, f, indent=2)
    print('Query processed!')


async def main():
    await arun_process('jsons/requests', target=process_query, args=(1, 2, 3))

if __name__ == '__main__':
    asyncio.run(main())