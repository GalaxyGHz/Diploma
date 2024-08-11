import pandas as pd
from tqdm import tqdm
from transformers import T5ForConditionalGeneration, T5Tokenizer

model_name = 'google/madlad400-10b-mt'
model = T5ForConditionalGeneration.from_pretrained(model_name, device_map="auto")
tokenizer = T5Tokenizer.from_pretrained(model_name)
text_translation_token = "<2sl> "

def translate_dataframe(dataframe, field_name='tweet', train_test='train'):
    tqdm.pandas()
    def translate(row):
        input_ids = tokenizer(text_translation_token + row[field_name], return_tensors="pt").input_ids.to(model.device)
        outputs = model.generate(input_ids=input_ids, max_new_tokens=256)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)

    df.insert(1, f'translated_{field_name}', df.progress_apply(translate, axis=1))
    df.to_csv(f'datasets/translatediSarcasmEval_T5/{train_test}.csv', encoding='utf-8', index=False, header=True)
    print(df.head())

# Translate train
df = pd.read_csv('datasets/iSarcasmEval/train/train.En.csv')
del df[df.columns[0]] # index
df = df.drop(columns=['rephrase'])
translate_dataframe(df, 'tweet', 'train')

# Translate test A
df = pd.read_csv('datasets/iSarcasmEval/test/task_A_En_test.csv')
translate_dataframe(df, 'text', 'test_A')

# Translate test B
df = pd.read_csv('datasets/iSarcasmEval/test/task_B_En_test.csv')
translate_dataframe(df, 'text', 'test_B')