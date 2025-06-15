import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, f1_score
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
import pickle
import textstat
import warnings
warnings.filterwarnings('ignore')

class JobFraudModelTrainer:
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.best_params = None

    def extract_features(self, df):
        features = pd.DataFrame()
        features['title_length'] = df['title'].str.len().fillna(0)
        features['description_length'] = df['description'].str.len().fillna(0)
        features['word_count'] = df['description'].str.split().str.len().fillna(0)
        features['avg_word_length'] = df['description'].apply(
            lambda x: np.mean([len(w) for w in str(x).split()]) if pd.notna(x) and len(str(x).split()) > 0 else 0
        )
        features['readability'] = df['description'].apply(
            lambda x: textstat.flesch_reading_ease(str(x)) if pd.notna(x) else 0
        )
        features['caps_ratio'] = df['description'].apply(
            lambda x: sum(c.isupper() for c in str(x)) / max(len(str(x)), 1)
        )
        features['digit_ratio'] = df['description'].apply(
            lambda x: sum(c.isdigit() for c in str(x)) / max(len(str(x)), 1)
        )
        features['punct_ratio'] = df['description'].apply(
            lambda x: sum(1 for c in str(x) if c in '.,!?') / max(len(str(x)), 1)
        )
        features['stopword_ratio'] = df['description'].apply(
            lambda x: len([w for w in str(x).lower().split() if w in ENGLISH_STOP_WORDS]) / max(len(str(x).split()), 1)
        )
        # Pattern flags
        patterns = {
            'urgent_kw': r'urgent|asap|immediate|quick|fast|now|hurry|rush',
            'money_kw': r'\$|money|payment|earn|income|profit|cash|dollar|pay|salary|wage',
            'contact_kw': r'email|phone|whatsapp|telegram|contact|call|text|message',
            'guarantee_kw': r'guarantee|promised|assured|certain|sure|definitely',
            'easy_kw': r'easy|simple|no experience|entry level|beginner|basic',
            'remote_kw': r'remote|work from home|anywhere|global|worldwide|online'
        }
        for key, pat in patterns.items():
            features[key] = df['description'].str.contains(pat, case=False, na=False)
        features['title_urgent'] = df['title'].str.contains(patterns['urgent_kw'], case=False, na=False)
        features['title_money'] = df['title'].str.contains(patterns['money_kw'], case=False, na=False)
        features['title_easy'] = df['title'].str.contains(patterns['easy_kw'], case=False, na=False)
        features['company_length'] = df.get('company', pd.Series([''] * len(df))).apply(lambda x: len(str(x)) if pd.notna(x) else 0)
        features['has_company'] = df.get('company', pd.Series([''] * len(df))).apply(lambda x: int(pd.notna(x) and str(x).strip() != ''))
        features['remote_work'] = df.get('location', pd.Series([''] * len(df))).str.contains(patterns['remote_kw'], case=False, na=False).fillna(False).astype(int)
        return features

    def preprocess_text(self, df):
        text = (df['title'].fillna('') + ' ' + df['description'].fillna('')).str.lower()
        text = text.str.replace(r'http\S+', ' ', regex=True)
        text = text.str.replace(r'[^\w\s]', ' ', regex=True)
        text = text.str.replace(r'\s+', ' ', regex=True).str.strip()
        return text

    def train_model(self, df, target_col='fraudulent', model_type='xgboost', balance_method='smotetomek'):
        structural = self.extract_features(df)
        text_data = self.preprocess_text(df)
        self.vectorizer = TfidfVectorizer(max_features=3000, stop_words='english', ngram_range=(1, 2))
        text_features = self.vectorizer.fit_transform(text_data).toarray()
        text_df = pd.DataFrame(text_features)

        # Combine features
        X = pd.concat([structural.reset_index(drop=True), text_df.reset_index(drop=True)], axis=1)

        # ✅ Save feature names to JSON here
        import json
        with open("feature_names.json", "w") as f:
            json.dump(list(X.columns), f)

        # Ensure column names are string
        X.columns = X.columns.map(str)

        # Fill NaNs
        X = X.fillna(0)

        # Target
        y = df[target_col].astype(int)

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42, test_size=0.2)

        # Apply balancing
        if balance_method == 'smotetomek':
            sampler = SMOTETomek(random_state=42)
        elif balance_method == 'smote':
            sampler = SMOTE(random_state=42)
        else:
            sampler = None

        if sampler:
            X_train, y_train = sampler.fit_resample(X_train, y_train)

        # Train model
        if model_type == 'xgboost':
            model = XGBClassifier(random_state=42, eval_metric='logloss', n_jobs=-1)
            param_grid = {'max_depth': [4, 6], 'learning_rate': [0.1], 'n_estimators': [100]}
        else:
            model = RandomForestClassifier(random_state=42, n_jobs=-1)
            param_grid = {'n_estimators': [100, 200], 'max_depth': [10, 20, None]}
            
        grid = GridSearchCV(model, param_grid, scoring='f1', cv=StratifiedKFold(3, shuffle=True, random_state=42), n_jobs=1, verbose=1)
        grid.fit(X_train, y_train)
        self.model = grid.best_estimator_
        self.best_params = grid.best_params_

        y_pred = self.model.predict(X_test)
        print("📊 Test Set Report:")
        print(classification_report(y_test, y_pred))


    def save_model(self, model_path='model.pkl', vectorizer_path='vectorizer.pkl'):
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        with open(vectorizer_path, 'wb') as f:
            pickle.dump(self.vectorizer, f)

    def load_model(self, model_path='model.pkl', vectorizer_path='vectorizer.pkl'):
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        with open(vectorizer_path, 'rb') as f:
            self.vectorizer = pickle.load(f)

    def predict(self, df):
        structural = self.extract_features(df)
        text_data = self.preprocess_text(df)
        text_features = self.vectorizer.transform(text_data).toarray()
        text_df = pd.DataFrame(text_features)
        X = pd.concat([structural.reset_index(drop=True), text_df.reset_index(drop=True)], axis=1)
        X = X.fillna(0)
        preds = self.model.predict(X)
        probs = self.model.predict_proba(X)[:, 1]
        return preds, probs
