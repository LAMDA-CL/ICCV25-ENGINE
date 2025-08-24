import json
from tqdm import tqdm
from openai import OpenAI
import re
def Gene_des(dataname):
    client = OpenAI(
        api_key="API key here", 
        base_url="url here",
    )

    js = {}
    label_path = "./labels.json"
    labels = json.load(open(label_path, 'r'))
    classes = labels[dataname]
    
    for target_class in tqdm(classes):
        js[target_class] = {}
        for other_class in classes:
            if target_class == other_class:
                continue
            js[target_class][other_class] = None
    
    for target_class_idx in tqdm(range(len(classes))):
        for other_class_idx in range(target_class_idx+1, len(classes)):
            target_class = classes[target_class_idx]
            other_class = classes[other_class_idx]
            prompt = "Q: What are different visual features between the Dog and the Cat in a photo?\n\
                        Dog:\n\
                        1. Generally larger and more muscular body, with various sizes depending on the breed.\n\
                        2. Often has a longer snout and wider nose.\n\
                        Cat:\n\
                        1. Smaller, more agile body with a slender frame.\n\
                        2. Shorter snout with a small, triangular nose.\n\
                        Q: What are different visual features between the {} and the {} in a photo? You should decide the number of attributes to output based on the actual situation, not just 2. You should follow the format of the examples given above. You should just output like example.\n\
                        {}:\n\
                        1.\
                        ...\
                        {}:\n\
                        1.\
                        ...".format(target_class, other_class, target_class, other_class)
            completion = client.chat.completions.create(
                model="qwen2.5-72b-instruct",
                messages=[
                    {'role': 'system', 'content': 'You are a helpful assistant.'},
                    {'role': 'user', 'content': prompt},],
                )
            des = completion.choices[0].message.content
            result = extract_classes_and_descriptions(des)
            key1 = list(result.keys())[0]
            key2 = list(result.keys())[1]
            js[target_class][other_class] = result[key1]
            js[other_class][target_class] = result[key2]
            
            with open("./{}_des.json".format(dataname), 'w') as f:
                json.dump(js, f, indent=4) 

def extract_classes_and_descriptions(text):
    lines = text.strip().split('\n')
    
    parsed_data = {}
    current_class = None
    current_description = []
    
    for line in lines:
        line = line.strip()
        if line.endswith(':'):
            if current_class is not None:
                parsed_data[current_class] = current_description
            
            current_class = line[:-1]
            current_description = []
        elif re.match(r'^\d+\.\s', line):
            feature = line.split('. ', 1)[1]
            current_description.append(feature)
    if current_class is not None:
        parsed_data[current_class] = current_description
    
    return parsed_data
Gene_des('sun')