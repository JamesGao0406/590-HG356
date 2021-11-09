import pandas as pd
import string
import re
import os 
novels="/Users/jamesgao/590-HG356/HW5.0/novels"
labels = []
content = []


def read_in_chunks(file_object, chunk_size=1000):
    """Lazy function (generator) to read a file piece by piece.
    Default chunk size: 1k."""
    while True:
        data = file_object.read(chunk_size)
        if not data:
            break
        yield data

for fname in os.listdir(novels):

    if fname[-4:] == '.txt':
        print(len(content))
        with open(os.path.join(novels, fname)) as f:
           
            for chunk in read_in_chunks(f):
                content.append(chunk)
                labels.append(fname[:-4])
        print(len(content))
# function to clean texts including lower cases, removing special characters, numbers and punctuations.
def text_clean(text):
    text=text.lower()
    pattern1 = r'[^a-zA-z0-9.,!?/:;\"\'\s]' 
    text1=re.sub(pattern1, '', text)
    pattern2 = r'[^a-zA-z.,!?/:;\"\'\s]' 
    text2= re.sub(pattern2, '', text1)
    text3 = ''.join([words for words in text2 if words not in string.punctuation])
    pattern3 = r'^\s*|\s\s*'
    text4=re.sub(pattern3, ' ', text3).strip()
    return text4

texts_new=[]
for text in content:
    texts_new.append(text_clean(text))

print(len(texts_new))

novel_names={"Friars and Filipinos":0, "Sister Carrie":1,"Monday and Tuesday":2, "The Letters of Jane Austen":3}

for i in range(len(labels)):
	labels[i]=novel_names[labels[i]]

df=pd.DataFrame({"text":texts_new,"label":labels})

df.to_csv('texts_new.csv',index=False)