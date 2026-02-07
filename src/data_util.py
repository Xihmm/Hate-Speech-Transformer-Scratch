import json

# Load dataset.json and post_id_divisions.json
with open('raw_data/dataset.json', 'r') as f:
    dataset = json.load(f)
with open('Raw_data/post_id_divisions.json', 'r') as f2:
    post_id_divisions = json.load(f2)

def load_datasets(type):
    if type not in post_id_divisions:
        raise ValueError(f"Invalid type: {type}. Must be one of {list(post_id_divisions.keys())}.")
    
    extracted_data = []
    for key in post_id_divisions[type]:
        data = dataset[key]
        post_id = data['post_id']
        post_tokens = data['post_tokens']
        annotators = data['annotators']

        json_format = {
            "Post ID": post_id,
            "Post": ' '.join(post_tokens),
            "Annotators": annotators[0]
        }

        extracted_data.append(json_format)

    # Save the dataset to a file
    output_file = f'processed_data/{type}_dataset.json'
    with open(output_file, 'w') as f:
        json.dump(extracted_data, f, indent=4)
    
    print(f"Saved {type} dataset to {output_file}")
    return extracted_data

if __name__ == '__main__':
    # Example usage:
    load_datasets('train')  # Replace 'test' with 'train' or 'val' as needed
