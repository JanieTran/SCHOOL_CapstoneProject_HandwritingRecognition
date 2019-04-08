import pandas as pd
import xml.etree.ElementTree as ET
import os

folder = os.path.join('iamdataset', 'xml')

xml_list = os.listdir(folder)
print('Total number of XML files:', len(xml_list))

# Get file names for training
train_files = open(os.path.join('iamdataset', 'subject', 'trainset.txt')).read()
train_files = train_files.split('\n')
del train_files[-1]

# Get file names for validation
val_files = open(os.path.join('iamdataset', 'subject', 'validationset1.txt')).read()
val_files = val_files.split('\n')
del val_files[-1]

print('Train set size:', len(train_files))
print('Val set size:', len(val_files))

# Initialise dataframe
df = pd.DataFrame(columns=['set', 'id', 'path', 'text'])

for xml_file in xml_list:
    tree = ET.parse(os.path.join(folder, xml_file))
    root = tree.getroot()
    handwritten_part = root[1]

    for line in root.iter('line'):
        # Get line id
        line_id = line.attrib['id']

        # If in train set
        if line_id in train_files:
            dataset = 'train'
        # If in val set
        elif line_id in val_files:
            dataset = 'val'
        # Ignore if in neither
        else:
            continue

        # Get line text
        line_text = line.attrib['text']
        line_text = line_text.replace('&quot;', '"')

        # Get file path
        line_name = line_id.split('-')
        file_path = os.path.join('dataset', 'iamdataset', 'line', line_name[0], line_name[0] + '-' + line_name[1], line_id + '.png')

        # Append to dataframe
        df = df.append({'set': dataset, 'id': line_id, 'path': file_path, 'text': line_text}, ignore_index=True)

print(df.shape)
df.to_csv('labels.csv', index=False)