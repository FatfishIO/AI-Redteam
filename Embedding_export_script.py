import requests
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

WEAVIATE_URL = "http://localhost:8080/v1/graphql"
COLLECTION = "DocChunk"
PAGE_SIZE = 200
OUT_DIR = Path("./export")
OUT_DIR.mkdir(exist_ok=True)

def gql(query: str):
    resp = requests.post(WEAVIATE_URL, json={"query": query})
    resp.raise_for_status()
    data = resp.json()
    if "errors" in data:
        raise RuntimeError(data["errors"])
    return data["data"]

print(f"Connected to Weaviate via raw HTTP. Collection: {COLLECTION}")

count_query = """
{
  Aggregate {
    DocChunk {
      meta {
        count
      }
    }
  }
}
"""

count_res = gql(count_query)
total = count_res["Aggregate"]["DocChunk"][0]["meta"]["count"]
print(f"Total objects: {total}")

all_ids = []
all_chunk_ids = []
all_vectors = []

cursor = None
fetched = 0

pbar = tqdm(total=total, desc="Exporting embeddings")

while True:
    after_clause = f'after: "{cursor}"' if cursor else ""

    query = f"""
    {{
      Get {{
        {COLLECTION}(
          limit: {PAGE_SIZE}
          {after_clause}
        ) {{
          chunk_id
          _additional {{
            id
            vector
          }}
        }}
      }}
    }}
    """

    res = gql(query)
    objs = res["Get"][COLLECTION]

    if not objs:
        break

    for obj in objs:
        all_ids.append(obj["_additional"]["id"])
        all_chunk_ids.append(obj.get("chunk_id"))
        all_vectors.append(obj["_additional"]["vector"])

    cursor = objs[-1]["_additional"]["id"]
    fetched += len(objs)
    pbar.update(len(objs))

    if fetched >= total:
        break

pbar.close()

vectors_np = np.array(all_vectors, dtype=np.float32)
np.save(OUT_DIR / "embeddings.npy", vectors_np)
np.save(OUT_DIR / "chunk_ids.npy", np.array(all_chunk_ids))
np.save(OUT_DIR / "uuids.npy", np.array(all_ids, dtype=object))

df_meta = pd.DataFrame({"uuid": all_ids, "chunk_id": all_chunk_ids})
df_vec = pd.DataFrame(
    vectors_np,
    columns=[f"dim_{i}" for i in range(vectors_np.shape[1])]
)
df_full = pd.concat([df_meta, df_vec], axis=1)

df_full.to_csv(OUT_DIR / "embeddings.csv", index=False)
df_full.to_parquet(OUT_DIR / "embeddings.parquet", index=False)

print("Vectors successfully exported!")
