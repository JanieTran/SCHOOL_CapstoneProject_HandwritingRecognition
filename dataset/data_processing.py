import pandas as pd
import xml.etree.ElementTree as ET
import os


def extract_files(txt_file):
    file_list = open(os.path.join('iamdataset', 'subject', txt_file)).read()
    file_list = file_list.split('\n')
    del file_list[-1]
    return file_list


folder = os.path.join('iamdataset', 'xml')

xml_list = os.listdir(folder)
print('Total number of XML files:', len(xml_list))

# Get file names
train_files = extract_files('trainset.txt')
val1_files = extract_files('validationset1.txt')
val2_files = extract_files('validationset2.txt')
test_files = extract_files('testset.txt')

print('Train set size:', len(train_files))
print('Val1 set size:', len(val1_files))
print('Val2 set size:', len(val2_files))
print('Test set size:', len(test_files))

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
        if line_id in train_files or line_id in val1_files:
            dataset = 'train'
        # If in val set
        elif line_id in val2_files:
            dataset = 'val'
        elif line_id in test_files:
            dataset = 'test'
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