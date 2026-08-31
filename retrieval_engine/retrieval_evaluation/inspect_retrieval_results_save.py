import json

save_path = None  # Set to the path of the retrieval results JSONL file to inspect
if save_path is None:
    raise ValueError("save_path must be set before running this script.")

results = []

with open(save_path, "r", encoding="UTF-8") as file:
    lines = file.readlines()
    for line in lines:
        results.append(json.loads(line))



print("question: ",results[0]["question"])
print("citations: ", results[0]["citations"])
print("relevant_chunks:\n\n",results[0]["relevant_chunks"])
print()

citation = results[0]["citations"]

for x in results[0]["relevant_chunks"]:
    print(citation in x[0]["text"])
        # print(True)