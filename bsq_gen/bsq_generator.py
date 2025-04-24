import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.llm.llm_call import llm_structed_output
from bsq_gen.bsq_gen_prompter import zero_shot_bsq_prompt
import pandas as pd
from pydantic import BaseModel

class BSQ(BaseModel):
    bsq: str

def zs_generate_bsq(topic, n_bsqs, label_list, base_on_feature):
    prompt = zero_shot_bsq_prompt(topic=topic, n_bsqs=n_bsqs, label_list=label_list, base_on_feature=base_on_feature)
    return llm_structed_output(prompt=prompt, res_schema=list[BSQ])

if __name__ == '__main__':
    labels_df = pd.read_csv('../data/medical_tc_labels.csv')
    labels_list = list(labels_df['condition_name'])
    bsq_list = zs_generate_bsq(topic='medical', n_bsqs=15, label_list=labels_list, base_on_feature='medical_abstract')
    output_data = [b.bsq for b in bsq_list]
    bsq_df = pd.DataFrame(output_data, columns=['bsq'])
    bsq_df.to_csv('output/bsq_list.csv', index=False)

