import itertools
import sys

import numpy as np
import sentence_transformers


match len(sys.argv):
    case 1:
        model_name = "PORTULAN/serafim-900m-portuguese-pt-sentence-encoder"

    case 2:
        model_name = sys.argv[1]

    case _:
        print(f"usage: {sys.argv[0]} [MODEL_NAME]", file=sys.stderr)
        sys.exit(2)

emb_model = sentence_transformers.SentenceTransformer(
    model_name_or_path=model_name,
)

# read questions from stdin, replace tabs with spaces, remove duplicates and sort (for reproducibility):
questions = sorted(
    {line.strip().replace("\t", " ") for line in sys.stdin if line.strip()}
)

# compute embeddings for all questions
embeddings = emb_model.encode(questions, convert_to_numpy=True)

# normalize embeddings row-wise
normed_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

# full cosine similarity matrix
sim_matrix = np.dot(normed_embeddings, normed_embeddings.T)

# collect only upper-triangle values (excluding diagonal)
scored_pairs = [
    (sim_matrix[i, j], questions[i], questions[j])
    for i, j in itertools.combinations(range(len(questions)), 2)
]

# sort from highest to lowest score
scored_pairs.sort(reverse=True)

for score, question1, question2 in scored_pairs:
    print(score, question1, question2, sep="\t")
