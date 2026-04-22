
from sklearn.base import BaseEstimator, TransformerMixin
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from collections import Counter
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


class DynamicTopNAndPCA(BaseEstimator, TransformerMixin):

    def __init__(self, top_n_max=50, pca_max=32, model_name="all-MiniLM-L6-v2"):
        self.top_n_max = top_n_max
        self.pca_max = pca_max
        self.model_name = model_name
        
        # Will be set during fit
        self.top_values_ = None
        self.model = None
        self.pca = None
        
    def _tokenise(self, x):
        """Normalize row to list of tokens"""
        # If x is array-like (list/Series/ndarray)
        if isinstance(x, (list, tuple, set, np.ndarray, pd.Series)):
            if len(x) == 0:
                return ["__EMPTY__"]
            return [str(v) if not pd.isna(v) and v != "" else "__EMPTY__" for v in x]
        
        # Otherwise scalar
        if pd.isna(x) or x == "":
            return ["__EMPTY__"]
        
        return [str(x)]
    
    def fit(self, X, y=None):
        X = np.asarray(X).ravel()
        flat = []
        for row in X:
            flat.extend(self._tokenise(row))
        
        # Count unique values
        counts = pd.Series(flat).value_counts()
        n_top = min(self.top_n_max, len(counts))
        self.top_values_ = counts.head(n_top).index.tolist()
        
        # Replace rare tokens with __OTHER__ for embedding
        texts = []
        for row in X:
            tokens = self._tokenise(row)
            tokens = [t if t in self.top_values_ else "__OTHER__" for t in tokens]
            texts.append(", ".join(tokens))
        
        # Embedding + PCA
        self.model = SentenceTransformer(self.model_name)
        embeddings = self.model.encode(texts, show_progress_bar=True)
        
        n_components = min(self.pca_max, embeddings.shape[0], embeddings.shape[1])
        self.pca = PCA(n_components=n_components, random_state=42)
        self.pca.fit(embeddings)
        
        return self
    
    def transform(self, X):
        X = np.asarray(X).ravel()
        texts = []
        for row in X:
            tokens = self._tokenise(row)
            tokens = [t if t in self.top_values_ else "__OTHER__" for t in tokens]
            texts.append(", ".join(tokens))
        
        embeddings = self.model.encode(texts, show_progress_bar=False)
        return self.pca.transform(embeddings)


def auto_top_n(df, columns, min_n=10, max_n=400, pct=0.1):
    top_n_map = {}

    for col in columns:
        n_unique = (
            df[col]
            .explode()
            .nunique()
        )

        top_n = int(max(min_n, min(max_n, n_unique * pct)))
        top_n_map[col] = top_n

    return top_n_map

class TopNTokenMapperSingleValue:


    def __init__(self, column, top_n=100, other_token="__OTHER__", empty_token="__EMPTY__"):
        self.column = column
        self.top_n = top_n
        self.other_token = other_token
        self.empty_token = empty_token
        self.vocab_ = None

    def fit(self, df):
        # Count frequency of each value
        counter = Counter(df[self.column].fillna(self.empty_token))
        
        # Determine top-N
        top_tokens = [t for t, _ in counter.most_common(self.top_n)]
        self.vocab_ = set(top_tokens) | {self.other_token, self.empty_token}
        return self

    def transform(self, df):
        if self.vocab_ is None:
            raise RuntimeError("Call fit() first")

        def map_value(x):
            val = self.empty_token if pd.isna(x) else str(x)
            return val if val in self.vocab_ else self.other_token

        df = df.copy()
        df[f"{self.column}_mapped"] = df[self.column].apply(map_value)
        return df

    def fit_transform(self, df):
        return self.fit(df).transform(df)

    def label_encode(self, df):
        """Return integer indices for embedding"""
        df = df.copy()
        le = LabelEncoder()
        df[f"{self.column}_idx"] = le.fit_transform(df[f"{self.column}_mapped"])
        return df, le
    
    
class TopNTokenMapper:
    def __init__(
        self,
        column,
        top_n=300,
        sep=",",
        other_token="__OTHER__",
        empty_token="__EMPTY__",
        max_tokens_per_row=None,
        lowercase=True,
        strip=True
    ):
        self.column = column
        self.top_n = top_n
        self.sep = sep
        self.other_token = other_token
        self.empty_token = empty_token
        self.max_tokens_per_row = max_tokens_per_row
        self.lowercase = lowercase
        self.strip = strip

        self.vocab_ = None
        self.counts_ = None

    def _tokenise(self, x):
        # Handle NaN / None
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return [self.empty_token]

        # Handle list / tuple / array
        if isinstance(x, (list, tuple, np.ndarray)):
            tokens = list(x)

        # Handle string
        elif isinstance(x, str):
            if x.strip() == "":
                return [self.empty_token]
            tokens = x.split(self.sep)

        else:
            # Fallback: coerce to string
            tokens = [str(x)]

        # Normalisation
        if self.strip:
            tokens = [t.strip() for t in tokens]
        if self.lowercase:
            tokens = [t.lower() for t in tokens]

        # Drop empty tokens
        tokens = [t for t in tokens if t]

        if not tokens:
            return [self.empty_token]

        if self.max_tokens_per_row:
            tokens = tokens[:self.max_tokens_per_row]

        return tokens

    def fit(self, df):
        counter = Counter()

        for x in df[self.column]:
            tokens = self._tokenise(x)
            counter.update(tokens)

        # Remove EMPTY before top-N selection
        counter.pop(self.empty_token, None)

        self.counts_ = counter
        top_tokens = [t for t, _ in counter.most_common(self.top_n)]

        self.vocab_ = set(top_tokens) | {self.other_token, self.empty_token}

        return self

    def transform(self, df):
        if self.vocab_ is None:
            raise RuntimeError("Call fit() before transform()")

        def map_tokens(x):
            tokens = self._tokenise(x)
            return [
                t if t in self.vocab_ else self.other_token
                for t in tokens
            ]

        df = df.copy()
        df[f"{self.column}_mapped"] = df[self.column].apply(map_tokens)
        return df

    def fit_transform(self, df):
        return self.fit(df).transform(df)


class ArrayToString(BaseEstimator, TransformerMixin):
    """Convert array-like column to single string for embeddings / TF-IDF"""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.asarray(X).ravel()  # flatten (n_samples,1) -> (n_samples,)
        out = []
        for val in X:
            if isinstance(val, (list, tuple, set, np.ndarray)):
                out.append(", ".join(map(str, val)) if len(val) > 0 else "unknown")
            else:
                out.append(str(val) if val is not None else "unknown")
        return np.array(out)

class TopNMultiHotWithOther(BaseEstimator, TransformerMixin):
    """
    Multi-hot encoder keeping only top N values, with an optional 'OTHER' column.
    """
    def __init__(self, top_n=20, include_other=True):
        self.top_n = top_n
        self.include_other = include_other
        self.top_values_ = []

    def fit(self, X, y=None):
        # Flatten all arrays to count frequency
        flat = []
        X = np.asarray(X).ravel()
        for arr in X:
            if isinstance(arr, (list, tuple, set, np.ndarray)):
                flat.extend(arr)
        counts = pd.Series(flat).value_counts()
        # Keep top N most frequent values
        self.top_values_ = counts.head(self.top_n).index.tolist()
        return self

    def transform(self, X):
        X = np.asarray(X).ravel()
        out = []
        for arr in X:
            if not isinstance(arr, (list, tuple, set, np.ndarray)):
                arr = []
            row = [1 if val in arr else 0 for val in self.top_values_]
            # Add OTHER column if any value is outside top N
            if self.include_other:
                row.append(int(any(val not in self.top_values_ for val in arr)))
            out.append(row)
        return np.array(out)
    
    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            prefix = "feature"
        else:
            prefix = input_features[0]
    
        names = [f"{prefix}_{val}" for val in self.top_values_]
    
        if self.include_other:
            names.append(f"{prefix}_OTHER")
    
        return names

class EmbeddingPCA(BaseEstimator, TransformerMixin):
    """Convert array/list column -> string -> embeddings -> PCA"""
    def __init__(self, model_name="all-MiniLM-L6-v2", pca_components=32):
        self.model_name = model_name
        self.pca_components = pca_components
        self.model = None
        self.pca = None

    def fit(self, X, y=None):
        X = np.asarray(X).ravel()
        texts = [", ".join(x) if isinstance(x, (list, np.ndarray, set, tuple)) and len(x) > 0 else "unknown"
                 for x in X]

        self.model = SentenceTransformer(self.model_name)
        embeddings = self.model.encode(texts, show_progress_bar=True)

        n_samples, n_features = embeddings.shape
        n_components = min(self.pca_components, n_samples, n_features)
        self.pca = PCA(n_components=n_components, random_state=42)
        self.pca.fit(embeddings)
        return self

    def transform(self, X):
        X = np.asarray(X).ravel()
        texts = [", ".join(x) if isinstance(x, (list, np.ndarray, set, tuple)) and len(x) > 0 else "unknown"
                 for x in X]
        embeddings = self.model.encode(texts, show_progress_bar=False)
        return self.pca.transform(embeddings)

def return_feature_names(feature_encoder, X_final):
    feature_names = []
    
    for name, transformer, cols in feature_encoder.transformers_:
        # passthrough
        if transformer == "passthrough":
            feature_names.extend(cols if isinstance(cols, list) else [cols])
        else:
            try:
                # scikit-learn transformers
                names = transformer.get_feature_names_out(cols if isinstance(cols, list) else [cols])
                feature_names.extend(names)
            except:
                # fallback: use the actual output shape
                try:
                    n_out = transformer.transform(np.zeros((1, len(cols) if isinstance(cols, list) else 1))).shape[1]
                except:
                    n_out = 1
                feature_names.extend([f"{cols}_{i}" for i in range(n_out)])
    
    return feature_names


def process_single_val_columns(df, data_dict ,single_label_list_cols = ["time_periods", "language_cues"]):
    for col in single_label_list_cols:
        new_col = list()
        for row in df[col].items():
            if row[1].size > 0:
                new_col.append(row[1][0])
            else:
                new_col.append("unknown")
        
        data_dict[col] = new_col
    return data_dict

class FeatureDiagnostics:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    # ---------- helpers ----------
    @staticmethod
    def _is_collection(x):
        return isinstance(x, (list, tuple, set, np.ndarray))

    # ---------- unique counting ----------
    def _count_unique_values(self, series):
        values = []
        for x in series:
            if isinstance(x, list):
                values.extend(x)
            elif isinstance(x, np.ndarray):
                values.extend(x.tolist())
            else:
                values.append(x)
    
        return len(set(values))

    def unique_value_table(self) -> pd.DataFrame:
        data = {
            col: self._count_unique_values(self.df[col])
            for col in self.df.columns
        }

        return (
            pd.DataFrame.from_dict(data, orient="index", columns=["n_unique"])
              .sort_values("n_unique", ascending=False)
        )

    # ---------- length diagnostics ----------
    def length_diagnostics(self) -> pd.DataFrame:
        rows = []

        for col in self.df.columns:
            lengths = []
            empty_count = 0

            for x in self.df[col].dropna():
                if self._is_collection(x):
                    l = len(x)
                    lengths.append(l)
                    if l == 0:
                        empty_count += 1
                else:
                    lengths.append(1)

            rows.append({
                "column": col,
                "avg_length": np.mean(lengths) if lengths else 0,
                "max_length": np.max(lengths) if lengths else 0,
                "pct_empty": empty_count / len(self.df),
                "is_multivalued": np.max(lengths) > 1 if lengths else False
            })

        return (
            pd.DataFrame(rows)
              .set_index("column")
              .sort_values("max_length", ascending=False)
        )

    # ---------- combined view ----------
    def summary(self) -> pd.DataFrame:
        return (
            self.unique_value_table()
            .join(self.length_diagnostics(), how="left")
            .sort_values(["is_multivalued", "n_unique"], ascending=[False, False])
        )