import numpy as np
import re

class FeatureExtractor:
    def __init__(self):

        self.medical_patterns = {
            'neoplasm_related': r'\b(cancer|tumor|neoplasm|malignancy|metastasis|oncology)\b',
            'digestive_related': r'\b(digestive|gastrointestinal|stomach|colon|intestine|liver|hepatitis|ulcer)\b',
            'nervous_related': r'\b(brain|nerve|spinal cord|neurological|parkinson|alzheimer|epilepsy)\b',
            'cardiovascular_related': r'\b(heart|cardiac|vascular|artery|stroke|blood vessel|hypertension|arrhythmia)\b',
            'general_pathology_related': r'\b(inflammation|infection|autoimmune|chronic disease|degenerative|biopsy|pathology)\b'
        }

    def extract_medical_features(self, text):
        text = str(text).lower()
        return [int(bool(re.search(pattern, text))) for pattern in self.medical_patterns.values()]

    def transform(self, df):
        bsq_cols = [col for col in df.columns if 'BSQ' in col]
        X_nllf = df[bsq_cols].values
        X_medical = np.array([self.extract_medical_features(text) for text in df['abstract']])
        X_combined = np.concatenate([X_nllf, X_medical], axis=1)
        feature_names = bsq_cols + list(self.medical_patterns.keys())
        return X_combined, feature_names

#phần này cũng tăng sương sương độ chính xác ấy, nên mình thấy nó cũng hữu ích, dù với chỉ là tìm keywworks thôi