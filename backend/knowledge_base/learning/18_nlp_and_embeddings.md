# NLP & Embeddings — Interview Preparation Guide

**Candidate:** Rahul Sharma | **Experience:** 4+ years | **Education:** MS Data Science, UMD
**Focus:** Data Scientist / ML Engineer — NLP, Transformers, LLM Systems
**Hands-on:** Legal document classification, call transcript tagging, embedding-based retrieval, BERT/RoBERTa fine-tuning

---

# Table of Contents

1. [NLP Fundamentals](#1-nlp-fundamentals)
2. [Word Embeddings (Static)](#2-word-embeddings-static)
3. [Contextual Embeddings](#3-contextual-embeddings)
4. [Sentence & Document Embeddings](#4-sentence--document-embeddings)
5. [Embedding Similarity Metrics](#5-embedding-similarity-metrics)
6. [Embedding Training Techniques](#6-embedding-training-techniques)
7. [Language Models: From N-grams to LLMs](#7-language-models-from-n-grams-to-llms)
8. [Text Classification](#8-text-classification)
9. [Information Retrieval](#9-information-retrieval)
10. [Common Interview Questions](#10-common-interview-questions-with-strong-answers)
11. [Key Takeaways](#11-key-takeaways)

---

# **1. NLP Fundamentals**

---

## **1.1 Tokenization**

Tokenization is the first step in any NLP pipeline — splitting raw text into discrete units (tokens) that a model can process. The choice of tokenizer fundamentally affects vocabulary size, ability to handle unseen words, and model performance.

```
 ┌───────────────────────────────────────────────────────────────────┐
 │                     Tokenization Strategies                       │
 │                                                                   │
 │  "unbelievably"                                                   │
 │                                                                   │
 │  Word-level:    ["unbelievably"]         → 1 token (OOV risk)    │
 │  BPE:           ["un", "believ", "ably"] → 3 subword tokens      │
 │  WordPiece:     ["un", "##believ", "##ably"] → 3 with ## prefix  │
 │  Unigram:       ["un", "believe", "ably"]    → 3 (probabilistic) │
 │  Character:     ["u","n","b","e","l",...]    → 13 tokens         │
 │  Byte-level:    [raw UTF-8 bytes]            → byte tokens       │
 └───────────────────────────────────────────────────────────────────┘
```

### **Word-Level Tokenization**

The simplest approach — split on whitespace and punctuation.

```python
text = "The cat sat on the mat."
tokens = text.lower().split()
# ['the', 'cat', 'sat', 'on', 'the', 'mat.']
```

**Problems:**
- Huge vocabulary (English has 170K+ words)
- Cannot handle OOV (out-of-vocabulary) words
- No morphological awareness: "running", "runs", "ran" are all different tokens
- Fails on agglutinative languages (Turkish, Finnish)

### **Subword Tokenization (The Modern Standard)**

Subword methods find a middle ground between word-level and character-level by learning frequent character sequences from a corpus.

#### **BPE (Byte-Pair Encoding)**

Used by: **GPT-2, GPT-3, GPT-4, LLaMA, Mistral**

Algorithm:
1. Start with a vocabulary of individual characters
2. Count all adjacent pairs in the corpus
3. Merge the most frequent pair into a new token
4. Repeat until desired vocabulary size is reached

```
 ┌──────────────────────────────────────────────────────────────┐
 │                    BPE Training Example                      │
 │                                                              │
 │  Corpus: "low lower newest widest"                           │
 │                                                              │
 │  Initial vocab: {l, o, w, e, r, n, s, t, i, d, _}           │
 │                                                              │
 │  Iteration 1: Most frequent pair = (e, s) → merge to "es"   │
 │  Iteration 2: Most frequent pair = (es, t) → merge to "est" │
 │  Iteration 3: Most frequent pair = (l, o) → merge to "lo"   │
 │  Iteration 4: Most frequent pair = (lo, w) → merge to "low" │
 │  ...                                                         │
 │                                                              │
 │  Final vocab: {l, o, w, e, r, n, s, t, i, d, _, es, est,    │
 │                lo, low, low_, er, ne, new, newest, ...}      │
 └──────────────────────────────────────────────────────────────┘
```

```python
# Hugging Face BPE tokenizer (GPT-2)
from transformers import GPT2Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokens = tokenizer.tokenize("unbelievably")
# ['un', 'believ', 'ably']  — 3 subword tokens
```

#### **WordPiece**

Used by: **BERT, DistilBERT, Electra**

Similar to BPE but uses a **likelihood-based** merging criterion instead of raw frequency. Merges the pair that maximizes the language model likelihood of the training data.

Key difference from BPE: uses `##` prefix for continuation tokens.

```python
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
tokens = tokenizer.tokenize("unbelievably")
# ['un', '##bel', '##ie', '##va', '##bly']
```

#### **SentencePiece**

Used by: **T5, ALBERT, LLaMA, XLNet, mBART**

A **language-independent** tokenizer that operates on raw Unicode text (no pre-tokenization needed). Can implement either BPE or Unigram underneath.

Key advantage: treats the input as a raw byte stream — no need for language-specific pre-processing (spaces, punctuation rules). Uses `▁` (U+2581) to denote whitespace.

```python
import sentencepiece as spm
sp = spm.SentencePieceProcessor()
sp.Load("model.model")
tokens = sp.EncodeAsPieces("New York is great")
# ['▁New', '▁York', '▁is', '▁great']
```

#### **Unigram LM**

Used by: **SentencePiece (option), ALBERT, T5**

Starts with a **large** candidate vocabulary and iteratively **removes** tokens that least affect the total corpus likelihood (opposite direction from BPE).

At inference, it finds the **most probable segmentation** using the Viterbi algorithm:

```
P(x₁, x₂, ..., xₙ) = ∏ P(xᵢ)

Best segmentation = argmax ∑ log P(xᵢ)
```

### **Byte-Level Tokenization**

Used by: **GPT-2 (byte-level BPE), ByT5**

Operates directly on UTF-8 bytes (vocabulary = 256 byte values). Guarantees zero OOV — any text in any language can be encoded.

```
 ┌────────────────────────────────────────────────────────────────┐
 │  "café" in byte-level:                                         │
 │                                                                │
 │  UTF-8 bytes: [99, 97, 102, 195, 169]                         │
 │               c    a    f    é (2 bytes)                       │
 │                                                                │
 │  Trade-off: Longer sequences (more bytes) but zero OOV         │
 └────────────────────────────────────────────────────────────────┘
```

### **Tokenization Comparison Table**

| Method | Used By | Vocab Size | OOV Handling | Speed | Key Feature |
|--------|---------|-----------|--------------|-------|-------------|
| **Word-level** | Traditional NLP | 50K–200K | Poor (UNK token) | Fast | Simplest |
| **BPE** | GPT-2/3/4, LLaMA | 30K–50K | Good (decomposes) | Fast | Frequency-based merging |
| **WordPiece** | BERT, DistilBERT | 30K | Good | Fast | Likelihood-based merging |
| **Unigram** | T5, ALBERT | 30K | Good | Fast | Probabilistic, pruning-based |
| **SentencePiece** | T5, LLaMA, mBART | 30K–50K | Excellent | Fast | Language-independent |
| **Byte-level** | ByT5, GPT-2 | 256 base | Perfect (no OOV) | Slower | Raw bytes, any language |

---

## **1.2 Text Preprocessing**

```
 ┌──────────────────────────────────────────────────────────────────┐
 │              Classic NLP Preprocessing Pipeline                   │
 │                                                                   │
 │  Raw Text                                                         │
 │    │                                                              │
 │    ▼                                                              │
 │  Lowercasing ──→ "The QUICK Brown FOX" → "the quick brown fox"   │
 │    │                                                              │
 │    ▼                                                              │
 │  Stopword Removal ──→ "the quick brown fox" → "quick brown fox"  │
 │    │                                                              │
 │    ▼                                                              │
 │  Stemming / Lemmatization                                         │
 │    │  Stemming:      "running" → "run"  (rule-based, aggressive) │
 │    │  Lemmatization: "better"  → "good" (dictionary-based)       │
 │    │                                                              │
 │    ▼                                                              │
 │  Tokenization → Feature Extraction → Model                       │
 └──────────────────────────────────────────────────────────────────┘
```

### **Lowercasing**

```python
text = "The QUICK Brown Fox"
text_lower = text.lower()
# "the quick brown fox"
```

**When NOT to lowercase:** NER (capitalization is a feature), sentiment (ALL CAPS = emphasis), acronyms (NASA vs nasa).

### **Stopword Removal**

Removes high-frequency, low-information words ("the", "is", "at", "which").

```python
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
tokens = [w for w in tokens if w not in stop_words]
```

**When NOT to remove:** Sentiment analysis ("not good" → "good" loses negation), question answering ("What is NLP?" → "NLP?" loses intent).

### **Stemming vs Lemmatization**

| Aspect | Stemming | Lemmatization |
|--------|----------|---------------|
| **Method** | Rule-based suffix stripping | Dictionary + POS lookup |
| **Speed** | Fast | Slower |
| **Output** | May not be real word ("studies" → "studi") | Always real word ("studies" → "study") |
| **Example** | Porter: "running" → "run", "better" → "better" | WordNet: "running" → "run", "better" → "good" |
| **Use case** | Information retrieval, search | Text classification, chatbots |

```python
# Stemming (Porter)
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()
stemmer.stem("unbelievably")  # "unbeliev"

# Lemmatization (spaCy)
import spacy
nlp = spacy.load("en_core_web_sm")
doc = nlp("The striped bats are hanging on their feet")
[token.lemma_ for token in doc]
# ['the', 'striped', 'bat', 'be', 'hang', 'on', 'their', 'foot']
```

**Important for interviews:** Modern transformer models (BERT, GPT) do NOT require stemming or stopword removal — they learn these patterns from data. These techniques are mainly relevant for traditional ML pipelines (TF-IDF + logistic regression).

---

## **1.3 Bag of Words (BoW) & TF-IDF**

### **Bag of Words**

Represents text as a vector of word counts, ignoring word order entirely.

```
 Corpus:
   Doc1: "the cat sat"
   Doc2: "the dog sat"
   Doc3: "the cat and the dog"

 Vocabulary: [the, cat, sat, dog, and]

 BoW Matrix:
          the  cat  sat  dog  and
   Doc1: [ 1    1    1    0    0 ]
   Doc2: [ 1    0    1    1    0 ]
   Doc3: [ 2    1    0    1    1 ]
```

```python
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(["the cat sat", "the dog sat", "the cat and the dog"])
print(vectorizer.get_feature_names_out())
# ['and', 'cat', 'dog', 'sat', 'the']
```

**Limitations:** No word order, no semantics, very sparse, "the" dominates.

### **TF-IDF (Term Frequency — Inverse Document Frequency)**

Addresses BoW's problem of common words dominating. Upweights rare, informative terms and downweights frequent, uninformative ones.

```
TF(t, d) = count(t in d) / |d|

IDF(t) = log(N / df(t))            where N = total docs, df(t) = docs containing t

TF-IDF(t, d) = TF(t, d) × IDF(t)
```

```
 ┌─────────────────────────────────────────────────────────────┐
 │                    TF-IDF Intuition                          │
 │                                                              │
 │  Word        TF (in doc)    IDF (across corpus)   TF-IDF   │
 │  ─────────   ───────────    ──────────────────    ───────   │
 │  "the"       High           Very Low (all docs)    Low ✗   │
 │  "quantum"   Low            Very High (rare)       High ✓  │
 │  "machine"   Medium         Medium                 Medium   │
 │                                                              │
 │  TF-IDF highlights discriminative terms per document         │
 └─────────────────────────────────────────────────────────────┘
```

```python
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X = tfidf.fit_transform(corpus)
# Sparse matrix: (n_docs, 5000)
```

**Interview tip:** TF-IDF is still used as a baseline and in hybrid search (BM25 is an evolved TF-IDF). Mention it as your go-to baseline before embeddings.

---

## **1.4 N-grams**

An n-gram is a contiguous sequence of n tokens. Captures local word order and context.

```
 Text: "the cat sat on the mat"

 Unigrams (n=1): ["the", "cat", "sat", "on", "the", "mat"]
 Bigrams  (n=2): ["the cat", "cat sat", "sat on", "on the", "the mat"]
 Trigrams (n=3): ["the cat sat", "cat sat on", "sat on the", "on the mat"]
```

**Why n-grams matter:**
- Bigram "not good" captures negation that unigrams miss
- Trigram "New York City" is a single concept
- Used in language models: P(w₃ | w₁, w₂) estimated from trigram counts

**Trade-off:** Higher n → more context but exponentially more features (curse of dimensionality) and data sparsity.

```python
from sklearn.feature_extraction.text import CountVectorizer
bigram_vectorizer = CountVectorizer(ngram_range=(1, 2))
X = bigram_vectorizer.fit_transform(corpus)
```

---

## **1.5 Named Entity Recognition (NER)**

NER identifies and classifies named entities in text into predefined categories.

```
 ┌──────────────────────────────────────────────────────────────┐
 │                      NER Example                             │
 │                                                              │
 │  "Rahul Sharma works at Google in Mountain View."            │
 │   ^^^^^^^^^^^^        ^^^^^^    ^^^^^^^^^^^^^                │
 │   PERSON               ORG       LOCATION                    │
 │                                                              │
 │  Standard entity types (OntoNotes):                          │
 │  PERSON, ORG, GPE, DATE, MONEY, PERCENT, PRODUCT, EVENT     │
 └──────────────────────────────────────────────────────────────┘
```

**Approaches:**

| Approach | Method | Example |
|----------|--------|---------|
| **Rule-based** | Regex, gazetteers | Dates: `\d{2}/\d{2}/\d{4}` |
| **Statistical** | CRF, HMM | spaCy v2 |
| **Neural** | BiLSTM-CRF | Lample et al. (2016) |
| **Transformer** | BERT + token classification head | spaCy v3, Hugging Face |
| **LLM-based** | Prompt GPT-4 with entity extraction instructions | Zero-shot NER |

```python
# spaCy NER
import spacy
nlp = spacy.load("en_core_web_trf")  # Transformer-based
doc = nlp("Apple is buying a startup in London for $1 billion.")
for ent in doc.ents:
    print(f"{ent.text:20s} {ent.label_:10s}")
# Apple                ORG
# London               GPE
# $1 billion           MONEY
```

**Your experience angle:** "In my legal document classification project, I used NER to extract party names, dates, and monetary amounts as features alongside the document embeddings. This hybrid approach improved classification accuracy by 3%."

---

## **1.6 Part-of-Speech (POS) Tagging**

Assigns grammatical categories (noun, verb, adjective, etc.) to each word.

```
 "The quick brown fox jumps over the lazy dog"
  DET  ADJ   ADJ  NOUN VERB  ADP  DET  ADJ  NOUN
```

**Use in NLP pipelines:**
- Feature engineering for traditional ML models
- Input to dependency parsing
- Chunking (extracting noun phrases: "the quick brown fox")
- Disambiguation: "bank" as NOUN (river bank) vs VERB (bank on something)

```python
import spacy
nlp = spacy.load("en_core_web_sm")
doc = nlp("The quick brown fox jumps over the lazy dog")
for token in doc:
    print(f"{token.text:10s} {token.pos_:6s} {token.tag_}")
# The        DET    DT
# quick      ADJ    JJ
# brown      ADJ    JJ
# fox        NOUN   NN
# jumps      VERB   VBZ
# ...
```

---

## **1.7 Dependency Parsing**

Identifies grammatical relationships between words in a sentence, producing a tree structure.

```
 ┌──────────────────────────────────────────────────────────────┐
 │  "The quick brown fox jumps over the lazy dog"               │
 │                                                              │
 │               jumps (ROOT)                                   │
 │              /     \                                         │
 │           fox       over                                     │
 │          / | \        \                                      │
 │       The quick brown  dog                                   │
 │                       / | \                                  │
 │                    the lazy                                  │
 │                                                              │
 │  Relations:                                                  │
 │  fox  ──nsubj──→  jumps    (nominal subject)                 │
 │  The  ──det────→  fox      (determiner)                      │
 │  quick ──amod──→  fox      (adjective modifier)              │
 │  over ──prep───→  jumps    (prepositional modifier)          │
 │  dog  ──pobj───→  over     (object of preposition)           │
 └──────────────────────────────────────────────────────────────┘
```

**Use cases:**
- Relation extraction (subject–verb–object triples)
- Question answering
- Coreference resolution
- Legal NLP: extracting obligations ("Company A shall pay Company B...")

```python
import spacy
nlp = spacy.load("en_core_web_sm")
doc = nlp("The company shall pay damages to the plaintiff")
for token in doc:
    print(f"{token.text:15s} {token.dep_:10s} {token.head.text}")
# The             det        company
# company         nsubj      pay
# shall           aux        pay
# pay             ROOT       pay
# damages         dobj       pay
# to              prep       pay
# the             det        plaintiff
# plaintiff       pobj       to
```

---

## **1.8 Sentiment Analysis**

Determining the emotional tone of text — positive, negative, neutral, or fine-grained.

```
 ┌─────────────────────────────────────────────────────────────────┐
 │                   Sentiment Analysis Approaches                  │
 │                                                                  │
 │  Level 1: Document-level  → "This movie is great" → Positive    │
 │  Level 2: Sentence-level  → Sentence-by-sentence polarity       │
 │  Level 3: Aspect-based    → "Food was great, service was slow"  │
 │                              Food: Positive, Service: Negative   │
 └─────────────────────────────────────────────────────────────────┘
```

**Approaches:**

| Method | Technique | Pros | Cons |
|--------|-----------|------|------|
| **Lexicon-based** | VADER, SentiWordNet | No training data needed | Can't handle sarcasm, context |
| **Traditional ML** | TF-IDF + SVM/NB | Fast, interpretable | Needs labeled data, limited context |
| **Deep Learning** | LSTM, CNN | Captures sequence patterns | Needs large labeled data |
| **Transformers** | Fine-tuned BERT/RoBERTa | SOTA accuracy | Compute-heavy |
| **LLM Zero-shot** | Prompt GPT-4 | No training data needed | Cost, latency |

```python
# Hugging Face sentiment pipeline
from transformers import pipeline
sentiment = pipeline("sentiment-analysis")
result = sentiment("I absolutely loved working on the legal NLP project!")
# [{'label': 'POSITIVE', 'score': 0.9998}]

# Aspect-based (custom)
aspects = {"food": "The food was delicious", "service": "but the service was terrible"}
for aspect, text in aspects.items():
    result = sentiment(text)
    print(f"{aspect}: {result[0]['label']} ({result[0]['score']:.3f})")
```

---

# **2. Word Embeddings (Static)**

---

## **2.1 Why Embeddings? The Distributional Hypothesis**

> *"You shall know a word by the company it keeps."* — J.R. Firth (1957)

Words that appear in similar contexts have similar meanings. Embeddings capture this by mapping each word to a dense vector where geometric proximity reflects semantic similarity.

```
 ┌──────────────────────────────────────────────────────────────────┐
 │                  One-Hot vs Dense Embeddings                     │
 │                                                                  │
 │  One-Hot (|V| = 50,000):                                        │
 │    "king"  = [0, 0, ..., 1, ..., 0, 0]   ← 50,000-dim, sparse  │
 │    "queen" = [0, 0, ..., 0, ..., 1, 0]   ← orthogonal to king! │
 │    cos(king, queen) = 0  ← no similarity!                       │
 │                                                                  │
 │  Dense Embedding (d = 300):                                      │
 │    "king"  = [0.23, -0.11, 0.89, ..., 0.04]   ← 300-dim, dense │
 │    "queen" = [0.21, -0.09, 0.87, ..., 0.06]   ← similar vector │
 │    cos(king, queen) ≈ 0.92  ← high similarity!                  │
 │                                                                  │
 │  Famous property (Word2Vec):                                     │
 │    king - man + woman ≈ queen                                    │
 └──────────────────────────────────────────────────────────────────┘
```

---

## **2.2 Word2Vec (Mikolov et al., 2013)**

The paper that started the embedding revolution. Two architectures:

### **CBOW (Continuous Bag of Words)**

Predicts the **center word** from surrounding context words.

```
 ┌──────────────────────────────────────────────────────────────┐
 │                          CBOW                                │
 │                                                              │
 │  Context: "the cat ___ on the mat"    Target: "sat"          │
 │                                                              │
 │  Input (one-hot):                                            │
 │    "the" ──→ [embedding] ──┐                                 │
 │    "cat" ──→ [embedding] ──┤                                 │
 │    "on"  ──→ [embedding] ──┼──→ Average ──→ Hidden ──→ P(sat)│
 │    "the" ──→ [embedding] ──┤              (W')    softmax    │
 │    "mat" ──→ [embedding] ──┘                                 │
 │                                                              │
 │  Loss: Cross-entropy between predicted and true center word  │
 │                                                              │
 │  Properties:                                                 │
 │  - Faster to train (averages context)                        │
 │  - Better for frequent words                                 │
 │  - Smooths over context → less sensitive to word order       │
 └──────────────────────────────────────────────────────────────┘
```

### **Skip-gram**

Predicts **context words** from the center word. Opposite direction.

```
 ┌──────────────────────────────────────────────────────────────┐
 │                        Skip-gram                             │
 │                                                              │
 │  Center word: "sat"   Window size: 2                         │
 │                                                              │
 │                        ┌──→ P("the")                         │
 │                        ├──→ P("cat")                         │
 │  "sat" ──→ [embedding] ┼──→ P("on")                         │
 │                        ├──→ P("the")                         │
 │                        └──→ P("mat")                         │
 │                                                              │
 │  Objective: Maximize P(context | center)                     │
 │                                                              │
 │  P(wₒ | wᵢ) = exp(v'ₒ · vᵢ) / Σⱼ exp(v'ⱼ · vᵢ)           │
 │                                                              │
 │  Properties:                                                 │
 │  - Slower (predicts multiple words per center word)          │
 │  - Better for rare words (each context is a training signal) │
 │  - Captures finer semantic relationships                     │
 └──────────────────────────────────────────────────────────────┘
```

### **Training Tricks**

**Negative Sampling** — Instead of computing softmax over entire vocabulary (expensive), sample k random "negative" words and do binary classification:

```
Maximize:  log σ(v'ₒ · vᵢ)  +  Σₖ E[log σ(-v'ₙ · vᵢ)]
           ─────────────────    ────────────────────────
           positive pair         k negative pairs

σ = sigmoid function
k = 5-20 negative samples (typical)
```

**Subsampling of frequent words** — Randomly discard common words ("the", "a") with probability proportional to their frequency:

```
P(discard wᵢ) = 1 - √(t / f(wᵢ))    where t ≈ 10⁻⁵
```

```python
# Training Word2Vec with Gensim
from gensim.models import Word2Vec
sentences = [["the", "cat", "sat"], ["the", "dog", "ran"]]
model = Word2Vec(sentences, vector_size=300, window=5, min_count=1,
                 sg=1,          # 1 = Skip-gram, 0 = CBOW
                 negative=10,   # negative samples
                 epochs=15)

# Analogy: king - man + woman ≈ queen
model.wv.most_similar(positive=['king', 'woman'], negative=['man'], topn=3)
# [('queen', 0.71), ('monarch', 0.62), ('princess', 0.59)]
```

---

## **2.3 GloVe (Pennington et al., 2014)**

**Glo**bal **Ve**ctors — combines the advantages of global matrix factorization (like LSA) with local context windows (like Word2Vec).

### **Core Idea**

Build a global word-word **co-occurrence matrix** X, where Xᵢⱼ = how often word i appears in the context of word j. Then factorize it:

```
 ┌──────────────────────────────────────────────────────────────────┐
 │                     GloVe Objective                              │
 │                                                                  │
 │  Minimize:                                                       │
 │                                                                  │
 │    J = Σᵢ,ⱼ f(Xᵢⱼ) · (wᵢᵀ · w̃ⱼ + bᵢ + b̃ⱼ - log Xᵢⱼ)²       │
 │                                                                  │
 │  where:                                                          │
 │    wᵢ, w̃ⱼ   = word and context word vectors                     │
 │    bᵢ, b̃ⱼ   = bias terms                                        │
 │    f(Xᵢⱼ)   = weighting function (caps influence of very        │
 │                frequent pairs)                                    │
 │                                                                  │
 │           ┌ (x/x_max)^α   if x < x_max                          │
 │    f(x) = ┤                                                      │
 │           └  1             otherwise                              │
 │                                                                  │
 │    (α = 0.75, x_max = 100 typically)                             │
 └──────────────────────────────────────────────────────────────────┘
```

```
 ┌──────────────────────────────────────────────────────────────┐
 │              GloVe vs Word2Vec                               │
 │                                                              │
 │  Word2Vec:                                                   │
 │    - Learns from local context windows (sliding window)      │
 │    - Online/incremental training possible                    │
 │    - Predictive model (neural network)                       │
 │                                                              │
 │  GloVe:                                                      │
 │    - Learns from global co-occurrence statistics             │
 │    - Requires full corpus co-occurrence matrix first         │
 │    - Count-based model (matrix factorization)                │
 │    - Often better on analogy tasks                           │
 └──────────────────────────────────────────────────────────────┘
```

---

## **2.4 FastText (Bojanowski et al., 2017)**

Extends Word2Vec by representing each word as a **bag of character n-grams** plus the word itself.

```
 ┌──────────────────────────────────────────────────────────────────┐
 │                    FastText Subword Example                      │
 │                                                                  │
 │  Word: "where"   (n-gram range: 3 to 6)                         │
 │                                                                  │
 │  Character n-grams:                                              │
 │    <wh, whe, her, ere, re>       (trigrams)                      │
 │    <whe, wher, here, ere>        (4-grams)                       │
 │    <wher, where, here>           (5-grams)                       │
 │    <where>                       (the word itself)               │
 │                                                                  │
 │  Embedding("where") = Σ embedding(n-gram) + embedding("<where>")│
 │                                                                  │
 │  OOV word "wheres":                                              │
 │    Still has valid embedding from shared n-grams!                │
 │    <wh, whe, her, ere, res, es>  ← many shared with "where"     │
 └──────────────────────────────────────────────────────────────────┘
```

**Key advantages:**
1. **Handles OOV words** — can compose embeddings from character n-grams
2. **Captures morphology** — "unhappy" shares n-grams with "happy"
3. **Works well for morphologically rich languages** (Turkish, Finnish, German)

```python
import fasttext
model = fasttext.train_unsupervised('corpus.txt', model='skipgram', dim=300)
# Works even for OOV words
vector = model.get_word_vector("unfamiliarword")
```

---

## **2.5 Static Embedding Comparison**

| Feature | Word2Vec | GloVe | FastText |
|---------|----------|-------|----------|
| **Method** | Neural (predictive) | Matrix factorization (count-based) | Neural + subword |
| **Training data** | Local context windows | Global co-occurrence matrix | Local context + char n-grams |
| **OOV handling** | No (UNK vector) | No (UNK vector) | Yes (subword composition) |
| **Morphology** | No | No | Yes |
| **Typical dims** | 100, 200, 300 | 50, 100, 200, 300 | 100, 300 |
| **Speed (training)** | Fast | Moderate | Slightly slower |
| **Best for** | General semantic tasks | Analogy, relatedness | Morphologically rich text, OOV |
| **Context-aware** | No (one vector per word) | No | No |

**Critical limitation of all static embeddings:** Each word gets exactly ONE vector regardless of context.

```
"I went to the bank to deposit money"  →  bank = financial institution
"I sat on the river bank"              →  bank = land beside river

Both get the SAME vector for "bank"!  This is the polysemy problem.
```

---

# **3. Contextual Embeddings**

---

## **3.1 Why Contextual Embeddings?**

Static embeddings assign a single vector per word. But language is inherently ambiguous:

```
 ┌──────────────────────────────────────────────────────────────────┐
 │               The Polysemy Problem                               │
 │                                                                  │
 │  "bank"                                                          │
 │    → "I deposited money at the bank"     (financial institution) │
 │    → "We sat on the river bank"          (land by water)         │
 │    → "Bank on it, it will work"          (rely on)               │
 │                                                                  │
 │  Static:     bank → [0.2, -0.1, 0.8, ...]  (SAME for all!)     │
 │                                                                  │
 │  Contextual: bank₁ → [0.9, -0.3, 0.1, ...]  (financial)        │
 │              bank₂ → [-0.2, 0.7, 0.4, ...]   (river)           │
 │              bank₃ → [0.1, 0.5, -0.6, ...]   (rely on)         │
 │                                                                  │
 │  Each occurrence gets a DIFFERENT vector based on context!       │
 └──────────────────────────────────────────────────────────────────┘
```

---

## **3.2 ELMo (Embeddings from Language Models)**

**Peters et al., 2018** — The first major contextual embedding model.

```
 ┌──────────────────────────────────────────────────────────────────┐
 │                      ELMo Architecture                           │
 │                                                                  │
 │  Input: "The cat sat on the bank"                                │
 │                                                                  │
 │  Layer 0: Character CNN → context-independent token embeddings   │
 │                │                                                 │
 │                ▼                                                 │
 │  Layer 1: Forward LSTM  ──→ h₁→  (reads left-to-right)          │
 │           Backward LSTM ←── ←h₁  (reads right-to-left)          │
 │                │                                                 │
 │                ▼                                                 │
 │  Layer 2: Forward LSTM  ──→ h₂→  (deeper representation)        │
 │           Backward LSTM ←── ←h₂                                 │
 │                │                                                 │
 │                ▼                                                 │
 │  ELMo(word) = γ · Σₗ sₗ · [h→ₗ ; h←ₗ]                          │
 │                                                                  │
 │  where:                                                          │
 │    sₗ = learned softmax weights per layer (task-specific)        │
 │    γ  = scalar scaling factor                                    │
 │    [;] = concatenation of forward & backward hidden states       │
 │                                                                  │
 │  Key insight: Different layers capture different linguistics:    │
 │    Layer 0: Morphology/syntax                                    │
 │    Layer 1: Syntax (POS, dependencies)                           │
 │    Layer 2: Semantics (word sense, sentiment)                    │
 └──────────────────────────────────────────────────────────────────┘
```

**Properties:**
- Bidirectional context (but shallow — forward and backward LSTMs are trained separately, then concatenated)
- Character-level input → handles OOV
- Produces 3 layers of representations (shallow→deep)
- Typical dimension: 1024 (512 per direction)

**Limitation:** LSTMs still struggle with very long-range dependencies compared to transformers.

---

## **3.3 BERT Embeddings (Devlin et al., 2019)**

BERT uses a **bidirectional transformer encoder** — every token attends to ALL other tokens in BOTH directions simultaneously (not just left-to-right or right-to-left separately like ELMo).

```
 ┌──────────────────────────────────────────────────────────────────┐
 │                   BERT Embedding Extraction                      │
 │                                                                  │
 │  Input: [CLS] The cat sat on the bank [SEP]                     │
 │                                                                  │
 │  ┌──────────────────────────────────────────┐                    │
 │  │          12 Transformer Layers            │                   │
 │  │  (each: Multi-Head Attention + FFN)       │                   │
 │  │                                           │                   │
 │  │  Layer 12: [h¹², h², h³, h⁴, h⁵, h⁶, h⁷]│  ← deep semantic │
 │  │  Layer 11: [....................................]│             │
 │  │  ...                                      │                   │
 │  │  Layer 1:  [h¹, h², h³, h⁴, h⁵, h⁶, h⁷] │  ← surface/syntax│
 │  └──────────────────────────────────────────┘                    │
 │                                                                  │
 │  Token embedding: h_i from specific layer(s)                     │
 │                                                                  │
 │  Sentence embedding strategies:                                  │
 │    1. [CLS] token from last layer                                │
 │    2. Mean-pool all token embeddings from last layer             │
 │    3. Mean-pool last 4 layers (often best)                       │
 │    4. [CLS] from second-to-last layer                            │
 └──────────────────────────────────────────────────────────────────┘
```

```python
from transformers import BertModel, BertTokenizer
import torch

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

text = "I deposited money at the bank"
inputs = tokenizer(text, return_tensors='pt')

with torch.no_grad():
    outputs = model(**inputs, output_hidden_states=True)

# Strategy 1: [CLS] token
cls_embedding = outputs.last_hidden_state[:, 0, :]  # shape: (1, 768)

# Strategy 2: Mean pooling
token_embeddings = outputs.last_hidden_state  # (1, seq_len, 768)
attention_mask = inputs['attention_mask'].unsqueeze(-1)  # (1, seq_len, 1)
mean_embedding = (token_embeddings * attention_mask).sum(1) / attention_mask.sum(1)

# Strategy 3: Last 4 layers concatenated (for per-token use)
hidden_states = outputs.hidden_states  # tuple of 13 tensors
last_four = torch.cat(hidden_states[-4:], dim=-1)  # (1, seq_len, 3072)
```

**BERT vs ELMo:**

| Feature | ELMo | BERT |
|---------|------|------|
| Architecture | BiLSTM (2 layers) | Transformer (12/24 layers) |
| Context | Shallow bidirectional (concatenate L→R, R→L) | Deep bidirectional (all tokens attend to all) |
| Pre-training | Next word prediction (both directions) | MLM + NSP |
| Embedding dim | 1024 | 768 (base) / 1024 (large) |
| Usage | Feature extraction (frozen) | Fine-tuning (typical) or feature extraction |

---

## **3.4 GPT Embeddings**

GPT uses a **unidirectional (autoregressive) transformer decoder**. Each token can only attend to tokens that come BEFORE it (causal masking).

```
 ┌──────────────────────────────────────────────────────────────────┐
 │              GPT: Unidirectional Attention                       │
 │                                                                  │
 │  "The cat sat on the bank"                                       │
 │                                                                  │
 │            The   cat   sat   on   the   bank                     │
 │  The       ✓     ✗     ✗     ✗    ✗     ✗                       │
 │  cat       ✓     ✓     ✗     ✗    ✗     ✗                       │
 │  sat       ✓     ✓     ✓     ✗    ✗     ✗                       │
 │  on        ✓     ✓     ✓     ✓    ✗     ✗                       │
 │  the       ✓     ✓     ✓     ✓    ✓     ✗                       │
 │  bank      ✓     ✓     ✓     ✓    ✓     ✓     ← causal mask    │
 │                                                                  │
 │  "bank" only sees left context ("The cat sat on the")            │
 │  It does NOT see what comes after — unidirectional                │
 │                                                                  │
 │  Contrast with BERT:                                             │
 │  BERT's "bank" sees FULL sentence in both directions             │
 └──────────────────────────────────────────────────────────────────┘
```

**Implication for embeddings:**
- GPT embeddings are **weaker for understanding** (sentence classification, NER) because they lack right context
- GPT is **stronger for generation** because autoregressive training directly models P(next token | previous tokens)
- For embedding tasks, BERT-style (encoder) models generally produce better representations than GPT-style (decoder) models

**However:** GPT-4 / text-embedding-3-* uses specialized training to produce high-quality embeddings despite the decoder architecture.

---

## **3.5 Why Contextual > Static (Summary)**

```
 ┌──────────────────────────────────────────────────────────────────┐
 │         Static vs Contextual Embeddings Summary                  │
 │                                                                  │
 │  Property              Static            Contextual              │
 │  ──────────            ──────            ──────────              │
 │  Vectors per word      1 (fixed)         Different per context   │
 │  Polysemy handling     ✗ None            ✓ Full                  │
 │  Long-range context    ✗ Local window    ✓ Full sequence         │
 │  Training              Unsupervised      Pre-trained on huge     │
 │                        (small corpus ok)  corpora (GB-TB)        │
 │  Dimensionality        50-300            768-4096                │
 │  Computational cost    Low               High                    │
 │  Typical use today     Baselines, edge   Standard for all NLP    │
 │                        devices                                   │
 │                                                                  │
 │  "bank" example:                                                 │
 │    Word2Vec: same vector for all senses                          │
 │    BERT:     different vector for each sentence context          │
 │              → financial bank clusters with "deposit", "loan"    │
 │              → river bank clusters with "water", "shore"         │
 └──────────────────────────────────────────────────────────────────┘
```

---

# **4. Sentence & Document Embeddings**

---

## **4.1 The Problem with Naive BERT Sentence Embeddings**

Using BERT's [CLS] token or mean-pooling directly for sentence similarity is **surprisingly bad**:

```
 ┌──────────────────────────────────────────────────────────────────┐
 │           Why Raw BERT Embeddings Fail for Similarity            │
 │                                                                  │
 │  BERT was trained with MLM (fill in blanks) and NSP (next       │
 │  sentence prediction), NOT to produce semantically meaningful   │
 │  sentence-level vectors.                                         │
 │                                                                  │
 │  Result: BERT [CLS] vectors form an "anisotropic" cone —        │
 │  all sentence embeddings point in roughly the same direction.    │
 │                                                                  │
 │  cos("A man walks a dog", "A man is eating") ≈ 0.95             │
 │  cos("A man walks a dog", "A dog walks in the park") ≈ 0.93    │
 │  cos("A man walks a dog", "The weather is sunny") ≈ 0.89       │
 │                                                                  │
 │  Everything is "similar"! Useless for retrieval/ranking.        │
 └──────────────────────────────────────────────────────────────────┘
```

**Solution:** Train sentence embedding models with contrastive objectives that explicitly learn to push similar sentences together and dissimilar sentences apart.

---

## **4.2 Sentence-BERT (SBERT) — Reimers & Gurevych, 2019**

The breakthrough model for efficient sentence embeddings. Uses a **siamese/twin network** architecture.

```
 ┌──────────────────────────────────────────────────────────────────┐
 │                 Sentence-BERT Architecture                       │
 │                                                                  │
 │   Sentence A                        Sentence B                   │
 │       │                                 │                        │
 │       ▼                                 ▼                        │
 │  ┌─────────┐                      ┌─────────┐                   │
 │  │  BERT    │  (shared weights)    │  BERT    │                  │
 │  └────┬────┘                      └────┬────┘                   │
 │       │                                │                        │
 │       ▼                                ▼                        │
 │  Mean Pooling                    Mean Pooling                    │
 │       │                                │                        │
 │       ▼                                ▼                        │
 │     u (768-d)                       v (768-d)                   │
 │       │                                │                        │
 │       └──────────┬─────────────────────┘                        │
 │                  │                                               │
 │           Similarity / Loss                                      │
 │                                                                  │
 │  Training objectives:                                            │
 │    Regression: cos(u, v) → similarity score (STS tasks)          │
 │    Classification: [u; v; |u-v|] → softmax (NLI data)           │
 │    Contrastive: push positive pairs close, negatives apart       │
 └──────────────────────────────────────────────────────────────────┘
```

**Why SBERT is critical:**
- BERT cross-encoder: encode (A, B) together → O(n²) comparisons for n sentences
- SBERT: encode A and B independently → O(n) encodings, then cosine similarity

```
 ┌──────────────────────────────────────────────────────────────────┐
 │  Finding most similar pair among 10,000 sentences:               │
 │                                                                  │
 │  BERT cross-encoder: 10,000 × 9,999 / 2 = ~50M forward passes  │
 │                      ≈ 65 hours on V100                          │
 │                                                                  │
 │  SBERT bi-encoder:   10,000 forward passes + cosine similarity  │
 │                      ≈ 5 seconds on V100                         │
 └──────────────────────────────────────────────────────────────────┘
```

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')  # 384-d, fast

sentences = [
    "A man is eating food.",
    "A man is eating a piece of bread.",
    "A man is riding a horse.",
]
embeddings = model.encode(sentences)  # shape: (3, 384)

# Compute cosine similarity
from sklearn.metrics.pairwise import cosine_similarity
sim_matrix = cosine_similarity(embeddings)
# [[1.0, 0.69, 0.08],
#  [0.69, 1.0, 0.06],
#  [0.08, 0.06, 1.0]]
```

---

## **4.3 Modern Embedding Models**

### **OpenAI Embeddings**

| Model | Dimensions | Max Tokens | Cost (per 1M tokens) | Best For |
|-------|-----------|-----------|----------------------|----------|
| `text-embedding-3-small` | 1536 (default), 512 | 8191 | $0.02 | Cost-effective general use |
| `text-embedding-3-large` | 3072 (default), 256–3072 | 8191 | $0.13 | Highest quality |
| `text-embedding-ada-002` | 1536 | 8191 | $0.10 | Legacy |

Key feature: **Matryoshka support** — you can truncate the embedding to a lower dimension (e.g., 256, 512) with minimal quality loss.

```python
from openai import OpenAI
client = OpenAI()

response = client.embeddings.create(
    input="Legal contract for property sale in Maryland",
    model="text-embedding-3-small",
    dimensions=512  # Matryoshka: truncate to 512-d
)
embedding = response.data[0].embedding  # list of 512 floats
```

### **Cohere Embed**

| Model | Dimensions | Languages | Best For |
|-------|-----------|-----------|----------|
| `embed-english-v3.0` | 1024 | English | Search, classification |
| `embed-english-light-v3.0` | 384 | English | Fast, lightweight |
| `embed-multilingual-v3.0` | 1024 | 100+ languages | Multilingual tasks |

Unique feature: **Input types** — Cohere lets you specify whether the input is a `search_document`, `search_query`, `classification`, or `clustering`, optimizing the embedding accordingly.

```python
import cohere
co = cohere.Client("API_KEY")

response = co.embed(
    texts=["Legal compliance framework for financial institutions"],
    model="embed-english-v3.0",
    input_type="search_document"
)
embedding = response.embeddings[0]  # 1024-d
```

### **Open-Source Leaders: BGE and E5**

| Model | Dimensions | MTEB Rank | Key Feature |
|-------|-----------|-----------|-------------|
| `BAAI/bge-large-en-v1.5` | 1024 | Top-5 | Instruction-tuned, best open-source |
| `BAAI/bge-base-en-v1.5` | 768 | Top-10 | Good balance of quality/speed |
| `BAAI/bge-small-en-v1.5` | 384 | Top-15 | Fast, lightweight |
| `intfloat/e5-large-v2` | 1024 | Top-5 | Requires "query:" / "passage:" prefix |
| `intfloat/e5-mistral-7b-instruct` | 4096 | Top-1 | LLM-based, highest quality |
| `BAAI/bge-m3` | 1024 | Top-3 | Multi-lingual, multi-granularity |

```python
from sentence_transformers import SentenceTransformer

# BGE model
model = SentenceTransformer('BAAI/bge-large-en-v1.5')
query_embedding = model.encode("Represent this sentence for retrieval: What is NLP?")
doc_embedding = model.encode("Natural language processing is a subfield of AI.")

# E5 model (requires prefix)
model = SentenceTransformer('intfloat/e5-large-v2')
query_embedding = model.encode("query: What is NLP?")
doc_embedding = model.encode("passage: Natural language processing is a subfield of AI.")
```

---

## **4.4 Dimensionality Landscape**

```
 ┌──────────────────────────────────────────────────────────────────┐
 │              Embedding Dimensions in Practice                    │
 │                                                                  │
 │  Dimension    Models                     Trade-off               │
 │  ─────────   ──────                     ─────────               │
 │    384       all-MiniLM-L6-v2,          Fast, memory-efficient, │
 │              bge-small, Cohere light    good for most tasks     │
 │                                                                  │
 │    768       BERT-base, bge-base,       Standard quality,       │
 │              all-mpnet-base-v2          moderate resources       │
 │                                                                  │
 │   1024       bge-large, Cohere v3,     High quality,            │
 │              GTE-large, E5-large       more storage/compute     │
 │                                                                  │
 │   1536       OpenAI text-embedding-3,  Very high quality,       │
 │              ada-002                   API-dependent             │
 │                                                                  │
 │   3072       OpenAI text-embedding-    Highest quality,          │
 │              3-large                   most storage              │
 │                                                                  │
 │   4096       e5-mistral-7b-instruct   LLM-based, SOTA quality  │
 │                                                                  │
 │  Rule of thumb:                                                  │
 │  - Prototype: 384-d (fast iteration)                             │
 │  - Production: 768–1024 (good quality/cost balance)              │
 │  - Maximum quality: 1536–3072                                    │
 └──────────────────────────────────────────────────────────────────┘
```

---

# **5. Embedding Similarity Metrics**

---

## **5.1 Cosine Similarity**

The **most widely used** metric for embedding similarity. Measures the angle between two vectors, ignoring magnitude.

```
                    A · B           Σᵢ aᵢ · bᵢ
cos(A, B) = ─────────────── = ─────────────────────
              ‖A‖ · ‖B‖       √(Σᵢ aᵢ²) · √(Σᵢ bᵢ²)

Range: [-1, 1]
  1.0  → identical direction (most similar)
  0.0  → orthogonal (unrelated)
 -1.0  → opposite direction (most dissimilar)
```

```
 ┌──────────────────────────────────────────────────────────────┐
 │              Cosine Similarity Geometric Intuition            │
 │                                                              │
 │           B                                                  │
 │          /                                                   │
 │         / θ = small → cos ≈ 1 (similar)                     │
 │        /                                                     │
 │  ─────A───────────────────→                                  │
 │                                                              │
 │           B                                                  │
 │          |                                                   │
 │          | θ = 90° → cos = 0 (orthogonal)                   │
 │          |                                                   │
 │  ────────A────────────────→                                  │
 │                                                              │
 │  Key: Only the ANGLE matters, not the vector lengths!        │
 │  [1, 2, 3] and [2, 4, 6] have cosine similarity = 1.0       │
 └──────────────────────────────────────────────────────────────┘
```

### **Why L2 Normalization Matters**

When vectors are L2-normalized (‖A‖ = ‖B‖ = 1), cosine similarity simplifies to a **dot product**:

```
If ‖A‖ = ‖B‖ = 1:
    cos(A, B) = A · B = Σᵢ aᵢ · bᵢ     (just a dot product!)

Also:
    ‖A - B‖² = ‖A‖² + ‖B‖² - 2(A · B) = 2 - 2·cos(A, B)

So for normalized vectors:
    cosine similarity ↔ dot product ↔ L2 distance (monotonic relationship)
    Maximizing cosine = Maximizing dot product = Minimizing L2 distance
```

**This is why many vector databases normalize embeddings on insertion** — it lets them use the faster dot product operation instead of full cosine computation.

```python
import numpy as np

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# With normalization
a_norm = a / np.linalg.norm(a)
b_norm = b / np.linalg.norm(b)
# Now cosine = dot product
cos_sim = np.dot(a_norm, b_norm)
```

---

## **5.2 Dot Product (Inner Product)**

```
dot(A, B) = Σᵢ aᵢ · bᵢ = ‖A‖ · ‖B‖ · cos(θ)

Range: (-∞, +∞)
```

**Dot product = cosine similarity × magnitudes.** It encodes both direction similarity AND magnitude.

**When dot product differs from cosine:**
- If embeddings have variable norms, dot product favors vectors with larger magnitude
- This can be useful when magnitude encodes importance (e.g., document length, relevance score)

**When they're equivalent:**
- With L2-normalized vectors: dot(A, B) = cos(A, B)
- Most embedding models (SBERT, OpenAI) produce near-unit-norm vectors

---

## **5.3 L2 (Euclidean) Distance**

```
L2(A, B) = ‖A - B‖ = √(Σᵢ (aᵢ - bᵢ)²)

Range: [0, +∞)
  0   → identical vectors
  +∞  → maximally different
```

**Relationship to cosine (for normalized vectors):**

```
‖A - B‖² = 2 - 2·cos(A, B)     (when ‖A‖ = ‖B‖ = 1)

So: small L2 distance ↔ high cosine similarity
```

---

## **5.4 When to Use Which**

| Metric | Best For | Properties | Used By |
|--------|----------|------------|---------|
| **Cosine Similarity** | Text embeddings, semantic search | Scale-invariant, normalized | Most NLP systems, Pinecone, Qdrant |
| **Dot Product** | Normalized embeddings, speed-critical | Faster than cosine (no norm division) | FAISS (IndexFlatIP), Milvus |
| **L2 Distance** | Image embeddings, clustering | Sensitive to magnitude | k-NN, FAISS (IndexFlatL2), K-Means |
| **Manhattan (L1)** | High-dimensional sparse vectors | More robust to outliers | Some recommendation systems |

```python
import numpy as np
from scipy.spatial.distance import cosine, euclidean

a = np.array([0.2, 0.5, 0.1, 0.8])
b = np.array([0.3, 0.4, 0.2, 0.7])

# Cosine similarity (1 - cosine distance)
cos_sim = 1 - cosine(a, b)         # 0.988

# Dot product
dot_prod = np.dot(a, b)            # 0.79

# L2 distance
l2_dist = euclidean(a, b)          # 0.2

# For normalized vectors — all metrics agree
a_n, b_n = a / np.linalg.norm(a), b / np.linalg.norm(b)
print(np.dot(a_n, b_n))            # = cosine similarity
print(np.sqrt(2 - 2*np.dot(a_n, b_n)))  # = L2 distance
```

---

# **6. Embedding Training Techniques**

---

## **6.1 Contrastive Learning**

The dominant paradigm for training embedding models. Core idea: learn representations where **similar items are close** and **dissimilar items are far apart** in the embedding space.

```
 ┌──────────────────────────────────────────────────────────────────┐
 │               Contrastive Learning Overview                      │
 │                                                                  │
 │  Positive pair: (query, relevant_doc)  → pull together           │
 │  Negative pair: (query, random_doc)    → push apart              │
 │                                                                  │
 │  Before training:          After training:                       │
 │                                                                  │
 │    q •    • d+              q • d+                               │
 │           • d-                                                   │
 │      • d-                          • d-                          │
 │                               • d-                               │
 │  (randomly scattered)    (positives close, negatives far)        │
 └──────────────────────────────────────────────────────────────────┘
```

### **InfoNCE Loss (Noise-Contrastive Estimation)**

The most widely used contrastive loss. Used by CLIP, SimCLR, DPR, and most modern embedding models.

```
                         exp(sim(q, d⁺) / τ)
L = -log ──────────────────────────────────────
          exp(sim(q, d⁺) / τ) + Σₖ exp(sim(q, dₖ⁻) / τ)

where:
  q   = query/anchor embedding
  d⁺  = positive (similar) embedding
  dₖ⁻ = k-th negative (dissimilar) embedding
  τ   = temperature parameter (typically 0.05–0.1)
  sim = cosine similarity (usually)
```

**Temperature τ controls sharpness:**
- Low τ (0.01): very sharp distribution → model focuses on hardest negatives
- High τ (1.0): soft distribution → model treats all negatives more equally
- τ = 0.07 is common (CLIP, SimCLR)

### **SimCLR (Chen et al., 2020)**

Self-supervised contrastive learning for visual representations (also applied to text).

```
 ┌──────────────────────────────────────────────────────────────────┐
 │                    SimCLR Framework                              │
 │                                                                  │
 │  Image x                                                         │
 │    │                                                             │
 │    ├──→ Augmentation 1 ──→ Encoder ──→ Projection ──→ z₁        │
 │    │    (crop, flip,       (ResNet)    (MLP)                     │
 │    │     color jitter)                                           │
 │    │                                                             │
 │    └──→ Augmentation 2 ──→ Encoder ──→ Projection ──→ z₂        │
 │         (different aug)    (shared      (shared                  │
 │                            weights)     weights)                 │
 │                                                                  │
 │  Positive pair: (z₁, z₂) from SAME image                       │
 │  Negatives:     all other images in the batch                    │
 │                                                                  │
 │  Loss: InfoNCE with in-batch negatives                           │
 │        (batch size matters a lot — 4096+ recommended)            │
 └──────────────────────────────────────────────────────────────────┘
```

### **CLIP (Radford et al., 2021)**

Contrastive Language–Image Pre-training. Jointly trains text and image encoders.

```
 ┌──────────────────────────────────────────────────────────────────┐
 │                     CLIP Training                                │
 │                                                                  │
 │  Batch of N (image, text) pairs:                                 │
 │                                                                  │
 │  Images:  I₁, I₂, ..., Iₙ  ──→  Image Encoder  ──→  e_img     │
 │  Texts:   T₁, T₂, ..., Tₙ  ──→  Text Encoder   ──→  e_txt     │
 │                                                                  │
 │  Similarity Matrix (N × N):                                      │
 │           T₁    T₂    T₃    ...   Tₙ                            │
 │    I₁   [✓     ✗     ✗     ...   ✗ ]   ← diagonal = positive   │
 │    I₂   [✗     ✓     ✗     ...   ✗ ]   ← off-diagonal = neg    │
 │    I₃   [✗     ✗     ✓     ...   ✗ ]                            │
 │    ...                                                           │
 │    Iₙ   [✗     ✗     ✗     ...   ✓ ]                            │
 │                                                                  │
 │  Loss = (image→text InfoNCE + text→image InfoNCE) / 2           │
 └──────────────────────────────────────────────────────────────────┘
```

---

## **6.2 Triplet Loss**

Operates on triplets of (anchor, positive, negative).

```
L = max(0, ‖f(a) - f(p)‖² - ‖f(a) - f(n)‖² + margin)

where:
  a = anchor
  p = positive (similar to anchor)
  n = negative (dissimilar to anchor)
  margin = minimum desired gap (typically 0.2–1.0)
```

```
 ┌──────────────────────────────────────────────────────────────────┐
 │                    Triplet Loss Geometry                         │
 │                                                                  │
 │  Before training:         After training:                        │
 │                                                                  │
 │       n •                       n •                              │
 │                                                                  │
 │    a •                        a • p •                            │
 │          • p                     ↑ margin                        │
 │                                  ↓                               │
 │                                       • n                        │
 │                                                                  │
 │  d(a,p) should be at least `margin` less than d(a,n)            │
 └──────────────────────────────────────────────────────────────────┘
```

**Hard negative mining** is critical: choosing negatives that are close to the anchor (hard negatives) makes training much more effective than random negatives.

```python
import torch
import torch.nn as nn

triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)
anchor = model.encode(anchor_texts)    # (batch, dim)
positive = model.encode(positive_texts)
negative = model.encode(negative_texts)
loss = triplet_loss(anchor, positive, negative)
```

---

## **6.3 Matryoshka Representation Learning (MRL)**

**Kusupati et al., 2022** — Train embeddings that are useful at **multiple dimensionalities** simultaneously.

```
 ┌──────────────────────────────────────────────────────────────────┐
 │              Matryoshka Embeddings                                │
 │                                                                  │
 │  Full embedding: [d₁, d₂, d₃, ..., d₇₆₈]                      │
 │                                                                  │
 │  Truncated versions (all useful!):                               │
 │    First  64 dims: [d₁, ..., d₆₄]        → fast retrieval      │
 │    First 128 dims: [d₁, ..., d₁₂₈]       → good quality        │
 │    First 256 dims: [d₁, ..., d₂₅₆]       → high quality        │
 │    First 512 dims: [d₁, ..., d₅₁₂]       → near-full quality   │
 │    Full  768 dims: [d₁, ..., d₇₆₈]       → maximum quality     │
 │                                                                  │
 │  Like Russian nesting dolls (Matryoshka):                        │
 │   ┌──────────────────────────────────┐                           │
 │   │ ┌────────────────────────┐       │                           │
 │   │ │ ┌──────────────┐      │       │                           │
 │   │ │ │ ┌──────┐     │      │       │                           │
 │   │ │ │ │ 64-d │256-d│512-d │ 768-d │                           │
 │   │ │ │ └──────┘     │      │       │                           │
 │   │ │ └──────────────┘      │       │                           │
 │   │ └────────────────────────┘       │                           │
 │   └──────────────────────────────────┘                           │
 └──────────────────────────────────────────────────────────────────┘
```

**Training:** Apply the contrastive loss at multiple truncation points simultaneously:

```
L_MRL = Σ_{d ∈ {64, 128, 256, 512, 768}} L_contrastive(embed[:d])
```

**Why it matters:**
- **Flexible compute/quality trade-off at inference** — use 64-d for fast filtering, 768-d for final ranking
- **Reduces storage** — store 256-d instead of 768-d with ~1% quality drop
- Supported by OpenAI `text-embedding-3-*` models (just pass `dimensions=256`)

```python
# OpenAI Matryoshka in action
from openai import OpenAI
client = OpenAI()

# Full quality
full = client.embeddings.create(input="NLP interview", model="text-embedding-3-small")
# 1536-d

# Compact (Matryoshka truncation)
compact = client.embeddings.create(
    input="NLP interview", model="text-embedding-3-small", dimensions=256
)
# 256-d — 6x smaller, ~1-2% quality drop
```

---

# **7. Language Models: From N-grams to LLMs**

---

## **7.1 Statistical Language Models**

A language model assigns a probability to a sequence of words:

```
P(w₁, w₂, ..., wₙ) = ∏ᵢ P(wᵢ | w₁, ..., wᵢ₋₁)

Using chain rule of probability.
```

### **N-gram Language Models**

Markov assumption: the probability of a word depends only on the previous (n-1) words.

```
Bigram:   P(wᵢ | wᵢ₋₁)                    → "the" → P("cat" | "the")
Trigram:  P(wᵢ | wᵢ₋₂, wᵢ₋₁)              → "the cat" → P("sat" | "the", "cat")

Estimated by counting:
P("sat" | "the", "cat") = count("the cat sat") / count("the cat")
```

**Smoothing techniques** (handle zero counts):
- **Add-k (Laplace):** Add k to all counts
- **Kneser-Ney:** Redistribute probability mass from frequent to rare n-grams
- **Backoff:** Fall back to lower-order n-gram if count is zero

### **Perplexity — The Evaluation Metric for Language Models**

```
PP(W) = P(w₁, w₂, ..., wₙ)^(-1/N)

     = exp( -(1/N) Σᵢ log P(wᵢ | w₁, ..., wᵢ₋₁) )

     = 2^H(W)     where H is the cross-entropy
```

**Intuition:** Perplexity = "how many equally likely next words the model is choosing from on average."

```
 ┌──────────────────────────────────────────────────────────────┐
 │                  Perplexity Examples                          │
 │                                                              │
 │  PP = 1    → model is perfectly certain (impossible)         │
 │  PP = 10   → model is choosing among ~10 equally likely words│
 │  PP = 50   → model is choosing among ~50 words               │
 │  PP = 250  → typical bigram model on Penn Treebank           │
 │  PP = 100  → typical trigram + Kneser-Ney                    │
 │  PP = 25   → GPT-2 (1.5B) on Penn Treebank                  │
 │  PP < 10   → GPT-3/4 level                                   │
 │                                                              │
 │  Lower perplexity = better language model                    │
 └──────────────────────────────────────────────────────────────┘
```

---

## **7.2 Neural Language Models**

```
 ┌──────────────────────────────────────────────────────────────────┐
 │            Evolution of Neural Language Models                    │
 │                                                                  │
 │  2003  Bengio Neural LM (feedforward, fixed context window)      │
 │    │                                                             │
 │    ▼                                                             │
 │  2013  Word2Vec (not an LM, but learned embeddings)              │
 │    │                                                             │
 │    ▼                                                             │
 │  2015  RNN/LSTM Language Models (variable-length context)        │
 │    │   └─ ELMo (2018): BiLSTM, contextual embeddings            │
 │    │                                                             │
 │    ▼                                                             │
 │  2017  Transformer (Vaswani et al.)  ← "Attention Is All You    │
 │    │                                     Need"                   │
 │    ├──→ 2018: GPT-1 (decoder-only, 117M params)                 │
 │    ├──→ 2018: BERT (encoder-only, 340M params)                  │
 │    ├──→ 2019: GPT-2 (1.5B), T5, BART, XLNet                    │
 │    ├──→ 2020: GPT-3 (175B)                                      │
 │    ├──→ 2022: ChatGPT, PaLM, LLaMA                              │
 │    └──→ 2023-2025: GPT-4, LLaMA 2/3, Mistral, Claude, Gemini   │
 └──────────────────────────────────────────────────────────────────┘
```

---

## **7.3 Encoder Models (Bidirectional)**

See the full context (both left and right). Best for **understanding** tasks.

```
 ┌──────────────────────────────────────────────────────────────────┐
 │                Encoder Models (BERT Family)                      │
 │                                                                  │
 │  Pre-training: Masked Language Modeling (MLM)                    │
 │                                                                  │
 │  Input:  "The [MASK] sat on the [MASK]"                          │
 │  Output: "The  cat   sat on the  mat"                            │
 │                                                                  │
 │  Each token sees ALL other tokens (bidirectional attention)      │
 │                                                                  │
 │  Best for:                                                       │
 │  ✓ Text classification                                           │
 │  ✓ Named Entity Recognition                                      │
 │  ✓ Question answering (extractive)                               │
 │  ✓ Sentence embeddings                                           │
 │  ✗ Text generation (not designed for it)                         │
 └──────────────────────────────────────────────────────────────────┘
```

| Model | Params | Pre-training | Key Innovation |
|-------|--------|-------------|----------------|
| **BERT** | 110M / 340M | MLM + NSP | First bidirectional transformer |
| **RoBERTa** | 125M / 355M | MLM only (no NSP), more data, longer training | Better training recipe |
| **DeBERTa** | 140M / 400M | MLM + disentangled attention | Relative position encoding, decoupled content/position |
| **ALBERT** | 12M–60M | MLM + SOP (Sentence Order Prediction) | Parameter sharing, factorized embeddings |
| **ELECTRA** | 14M–335M | Replaced Token Detection | More sample-efficient than MLM |

---

## **7.4 Decoder Models (Autoregressive)**

Left-to-right generation. Best for **text generation**.

```
 ┌──────────────────────────────────────────────────────────────────┐
 │             Decoder Models (GPT Family)                          │
 │                                                                  │
 │  Pre-training: Causal Language Modeling (CLM)                    │
 │                                                                  │
 │  P(xₜ | x₁, x₂, ..., xₜ₋₁)  →  predict next token             │
 │                                                                  │
 │  "The cat" → predict "sat"                                       │
 │  "The cat sat" → predict "on"                                    │
 │  "The cat sat on" → predict "the"                                │
 │                                                                  │
 │  Causal mask: each position can only attend to earlier positions │
 │                                                                  │
 │  Best for:                                                       │
 │  ✓ Text generation                                               │
 │  ✓ Code generation                                               │
 │  ✓ In-context learning (few-shot)                                │
 │  ✓ Instruction following                                         │
 │  △ Classification (via prompting, not as efficient as encoders)  │
 └──────────────────────────────────────────────────────────────────┘
```

| Model | Params | Key Innovation |
|-------|--------|----------------|
| **GPT-2** | 1.5B | Showed zero-shot task transfer |
| **GPT-3** | 175B | In-context learning, few-shot prompting |
| **GPT-4** | ~1.8T (MoE) | Multimodal, strongest reasoning |
| **LLaMA** | 7B–70B | Open-weights, efficient training |
| **LLaMA 2** | 7B–70B | RLHF, chat-tuned, commercially licensed |
| **LLaMA 3** | 8B–405B | Scaling to 405B, multilingual |
| **Mistral** | 7B | Sliding window attention, MoE (Mixtral) |

---

## **7.5 Encoder-Decoder Models**

Full sequence-to-sequence architecture. Best for **sequence transduction** tasks.

```
 ┌──────────────────────────────────────────────────────────────────┐
 │            Encoder-Decoder Models (T5, BART)                     │
 │                                                                  │
 │  Encoder (bidirectional) → Decoder (autoregressive)              │
 │                                                                  │
 │  T5 formulates EVERY task as text-to-text:                       │
 │    Input:  "classify: This movie was amazing"                    │
 │    Output: "positive"                                            │
 │                                                                  │
 │    Input:  "translate English to French: Hello world"            │
 │    Output: "Bonjour le monde"                                    │
 │                                                                  │
 │    Input:  "summarize: [long document]"                          │
 │    Output: "[short summary]"                                     │
 └──────────────────────────────────────────────────────────────────┘
```

| Model | Params | Pre-training | Best For |
|-------|--------|-------------|----------|
| **T5** | 60M–11B | Span corruption (denoising) | Summarization, translation, classification |
| **BART** | 140M–400M | Denoising (mask, delete, permute, rotate) | Summarization, generation |
| **Flan-T5** | 80M–11B | Multi-task instruction tuning on T5 | Zero-shot, instruction following |
| **mBART** | 680M | Denoising across 25 languages | Multilingual translation |

---

## **7.6 Pre-training Objectives**

```
 ┌──────────────────────────────────────────────────────────────────┐
 │               Pre-training Objectives Compared                   │
 │                                                                  │
 │  1. Masked Language Modeling (MLM) — BERT, RoBERTa               │
 │     Mask 15% of tokens, predict them                             │
 │     "The [MASK] sat on the mat" → predict "cat"                  │
 │     Bidirectional — sees both left and right context              │
 │                                                                  │
 │  2. Causal Language Modeling (CLM) — GPT, LLaMA                  │
 │     Predict next token given all previous tokens                 │
 │     P(xₜ | x₁, ..., xₜ₋₁)                                      │
 │     Unidirectional — only sees left context                      │
 │                                                                  │
 │  3. Denoising (Span Corruption) — T5, BART                      │
 │     Corrupt input (mask spans, shuffle, delete), reconstruct     │
 │     Input:  "The <X> on the mat"  (masked "cat sat")             │
 │     Output: "<X> cat sat"                                        │
 │     Encoder sees corrupted; decoder generates original           │
 │                                                                  │
 │  4. Next Sentence Prediction (NSP) — BERT                       │
 │     Given (Sentence A, Sentence B), predict if B follows A       │
 │     Later found to be unnecessary (RoBERTa removed it)           │
 │                                                                  │
 │  5. Replaced Token Detection (RTD) — ELECTRA                    │
 │     Small generator replaces tokens; discriminator detects which │
 │     More sample-efficient than MLM (trains on ALL tokens)        │
 └──────────────────────────────────────────────────────────────────┘
```

---

# **8. Text Classification**

---

## **8.1 Traditional ML Approaches**

```
 ┌──────────────────────────────────────────────────────────────────┐
 │          Traditional Text Classification Pipeline                │
 │                                                                  │
 │  Raw Text → Preprocessing → Feature Extraction → Classifier     │
 │               │                    │                  │          │
 │            lowercase,           BoW / TF-IDF /     NB, SVM,     │
 │            stem, stop           n-grams             LogReg      │
 └──────────────────────────────────────────────────────────────────┘
```

### **Naive Bayes**

```
P(class | document) ∝ P(class) · ∏ᵢ P(wordᵢ | class)

Assumes words are conditionally independent given class (naive assumption).
```

**Variants:**
- **Multinomial NB:** For word counts / TF-IDF (most common for text)
- **Bernoulli NB:** For binary word presence/absence
- **Complement NB:** Better for imbalanced datasets

```python
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=10000, ngram_range=(1, 2))),
    ('clf', MultinomialNB(alpha=0.1))
])
pipeline.fit(X_train, y_train)
accuracy = pipeline.score(X_test, y_test)
```

**Pros:** Very fast, works well with small data, good baseline.
**Cons:** Independence assumption is wrong, can't capture word interactions.

### **SVM (Support Vector Machine)**

Linear SVM with TF-IDF features was the **gold standard** before deep learning.

```python
from sklearn.svm import LinearSVC

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=20000, ngram_range=(1, 2),
                               sublinear_tf=True)),
    ('clf', LinearSVC(C=1.0, class_weight='balanced'))
])
```

**Why SVM works well for text:**
- High-dimensional sparse data (TF-IDF produces very high-D vectors)
- SVM is effective in high-D spaces (maximizes margin)
- Linear kernel is usually sufficient (text is often linearly separable in high-D)

### **Logistic Regression**

Often the strongest traditional baseline. Outputs calibrated probabilities.

```python
from sklearn.linear_model import LogisticRegression

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=15000, ngram_range=(1, 2))),
    ('clf', LogisticRegression(C=1.0, max_iter=1000, class_weight='balanced'))
])
```

---

## **8.2 Deep Learning Approaches**

### **CNN for Text (Kim, 2014)**

```
 ┌──────────────────────────────────────────────────────────────────┐
 │                  TextCNN Architecture                             │
 │                                                                  │
 │  Input: "The legal contract is binding"                          │
 │                                                                  │
 │  Embedding layer (300-d each):                                   │
 │  ┌─────┬─────┬─────┬─────┬─────┬─────┐                          │
 │  │ The │legal│cont-│ is  │bind-│     │  ← (seq_len × 300)       │
 │  │     │     │ract │     │ ing │     │                           │
 │  └─────┴─────┴─────┴─────┴─────┴─────┘                          │
 │    │           │           │                                     │
 │    ▼           ▼           ▼                                     │
 │  Filter     Filter      Filter                                  │
 │  size=2     size=3      size=4      (multiple filter sizes)     │
 │    │           │           │                                     │
 │    ▼           ▼           ▼                                     │
 │  Conv1D    Conv1D      Conv1D       (128 filters each)          │
 │    │           │           │                                     │
 │    ▼           ▼           ▼                                     │
 │  MaxPool   MaxPool     MaxPool      (global max-over-time)      │
 │    │           │           │                                     │
 │    └─────────┼─────────────┘                                     │
 │              ▼                                                   │
 │          Concatenate ──→ Dropout ──→ FC ──→ Softmax             │
 │          (384-d)          (0.5)       (n_classes)               │
 └──────────────────────────────────────────────────────────────────┘
```

**Why CNNs work for text:** Convolutional filters capture local n-gram patterns (similar to n-gram features but learned).

### **LSTM / BiLSTM**

```
 ┌──────────────────────────────────────────────────────────────────┐
 │              BiLSTM for Text Classification                      │
 │                                                                  │
 │  Input: x₁, x₂, x₃, ..., xₙ  (word embeddings)                │
 │                                                                  │
 │  Forward:   h→₁ → h→₂ → h→₃ → ... → h→ₙ                       │
 │  Backward:  h←₁ ← h←₂ ← h←₃ ← ... ← h←ₙ                      │
 │                                                                  │
 │  Representation: [h→ₙ ; h←₁]  or  attention-weighted mean      │
 │                      │                                           │
 │                      ▼                                           │
 │                   FC → Softmax → class probabilities              │
 └──────────────────────────────────────────────────────────────────┘
```

### **Transformer Fine-tuning (The Modern Standard)**

```
 ┌──────────────────────────────────────────────────────────────────┐
 │         Transformer Fine-tuning for Classification               │
 │                                                                  │
 │  Input: [CLS] This legal document concerns a property sale [SEP] │
 │                                                                  │
 │  ┌──────────────────────────────────┐                            │
 │  │   Pre-trained BERT / RoBERTa     │ ← frozen or fine-tuned    │
 │  │   (12 transformer layers)         │                           │
 │  └────────────────┬─────────────────┘                            │
 │                   │                                              │
 │              [CLS] embedding (768-d)                             │
 │                   │                                              │
 │              Dropout (0.1)                                       │
 │                   │                                              │
 │              Linear (768 → n_classes)                            │
 │                   │                                              │
 │              Softmax → class probabilities                       │
 └──────────────────────────────────────────────────────────────────┘
```

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer

model = AutoModelForSequenceClassification.from_pretrained(
    "roberta-base", num_labels=5
)
tokenizer = AutoTokenizer.from_pretrained("roberta-base")

# Tokenize
inputs = tokenizer(texts, padding=True, truncation=True, max_length=512,
                   return_tensors="pt")

# Fine-tune with Trainer API
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    learning_rate=2e-5,          # lower LR for fine-tuning
    weight_decay=0.01,
    warmup_steps=500,
    evaluation_strategy="epoch",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)
trainer.train()
```

---

## **8.3 Few-Shot Classification with LLMs**

```python
from openai import OpenAI
client = OpenAI()

prompt = """Classify the following legal document excerpt into one of these categories:
- CONTRACT
- LITIGATION
- REGULATORY
- INTELLECTUAL_PROPERTY
- EMPLOYMENT

Examples:
Text: "The parties agree to the terms of sale for the property located at..."
Category: CONTRACT

Text: "The plaintiff alleges that the defendant failed to..."
Category: LITIGATION

Text: "Pursuant to Section 12 of the Securities Exchange Act..."
Category: REGULATORY

Now classify:
Text: "{document_text}"
Category:"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}],
    temperature=0,
    max_tokens=20
)
category = response.choices[0].message.content.strip()
```

---

## **8.4 Your Experience: Legal Document & Call Transcript Classification**

```
 ┌──────────────────────────────────────────────────────────────────┐
 │        Your Project: Legal Document Classification               │
 │                                                                  │
 │  Problem: Classify legal documents into 5+ categories            │
 │          (contracts, litigation, regulatory, IP, employment)      │
 │                                                                  │
 │  Approach:                                                       │
 │  1. Baseline: TF-IDF + Logistic Regression (F1: 0.78)           │
 │  2. Improvement: Fine-tuned RoBERTa-base (F1: 0.91)             │
 │  3. Added NER features (party names, dates) → F1: 0.94          │
 │                                                                  │
 │  Pipeline:                                                       │
 │  Raw PDF → OCR/Parsing → Text Cleaning → Chunking (512 tokens)  │
 │    → RoBERTa Encoding → Classification Head → Category           │
 │                                                                  │
 │  Challenges:                                                     │
 │  - Long documents (10-50 pages): used hierarchical approach      │
 │    (classify chunks → aggregate with attention)                  │
 │  - Class imbalance: used focal loss + oversampling               │
 │  - Domain-specific language: continued pre-training on legal     │
 │    corpus (legal-BERT) improved by 3%                            │
 └──────────────────────────────────────────────────────────────────┘

 ┌──────────────────────────────────────────────────────────────────┐
 │        Your Project: Call Transcript Classification              │
 │                                                                  │
 │  Problem: Classify customer support call transcripts             │
 │          (complaint, inquiry, feedback, escalation, resolution)  │
 │                                                                  │
 │  Approach:                                                       │
 │  1. Preprocessing: speaker diarization, filler word removal      │
 │  2. Feature engineering:                                         │
 │     - Sentiment trajectory (how sentiment changes over call)     │
 │     - TF-IDF on agent vs customer utterances separately          │
 │     - Call duration, silence ratio, interruption count           │
 │  3. Model: Fine-tuned DeBERTa on transcript text + metadata     │
 │                                                                  │
 │  Key insight: Combining text classification with call metadata   │
 │  features outperformed text-only models by 5% F1                 │
 └──────────────────────────────────────────────────────────────────┘
```

**Interview framing tips:**
- Always mention the **baseline → improvement** progression
- Quantify improvements with metrics (F1, accuracy, latency)
- Highlight **domain-specific challenges** and how you solved them
- Show awareness of **production concerns**: latency, cost, model size

---

# **9. Information Retrieval**

---

## **9.1 BM25 (Best Matching 25)**

The evolution of TF-IDF. The **most widely used** lexical retrieval algorithm (Elasticsearch, Lucene, Solr default).

```
                    (k₁ + 1) · tf(t,d)                          N - df(t) + 0.5
BM25(q, d) = Σₜ ──────────────────────────────── · log ──────────────────────
              t∈q   tf(t,d) + k₁ · (1 - b + b · |d|/avgdl)         df(t) + 0.5

where:
  tf(t,d)  = term frequency of term t in document d
  df(t)    = number of documents containing term t
  N        = total number of documents
  |d|      = length of document d (in words)
  avgdl    = average document length
  k₁       = term frequency saturation (typically 1.2–2.0)
  b        = length normalization (typically 0.75)
```

```
 ┌──────────────────────────────────────────────────────────────────┐
 │                BM25 vs TF-IDF                                    │
 │                                                                  │
 │  TF-IDF problems:                                                │
 │  1. TF is unbounded — a word appearing 100× gets 100× weight    │
 │  2. No document length normalization                             │
 │                                                                  │
 │  BM25 fixes:                                                     │
 │  1. TF saturation: diminishing returns after k₁ occurrences     │
 │                                                                  │
 │     TF weight                                                    │
 │       │                                                          │
 │       │        ╭──────────── BM25 (saturates)                    │
 │       │       ╱                                                  │
 │       │      ╱                                                   │
 │       │     ╱     ╱ TF-IDF (linear, unbounded)                   │
 │       │    ╱    ╱                                                 │
 │       │   ╱  ╱                                                   │
 │       │  ╱╱                                                      │
 │       └────────────────── Term frequency                         │
 │                                                                  │
 │  2. Length normalization: long docs don't unfairly dominate      │
 │     (controlled by parameter b)                                  │
 └──────────────────────────────────────────────────────────────────┘
```

```python
# BM25 with rank_bm25
from rank_bm25 import BM25Okapi

corpus = [
    "the cat sat on the mat".split(),
    "the dog played in the park".split(),
    "a legal contract was signed today".split(),
]
bm25 = BM25Okapi(corpus)

query = "legal contract".split()
scores = bm25.get_scores(query)
# [0.0, 0.0, 1.29]  — only doc 3 matches
```

---

## **9.2 Dense Retrieval: Bi-Encoders**

Encode queries and documents independently into dense vectors, then use ANN (approximate nearest neighbor) search.

```
 ┌──────────────────────────────────────────────────────────────────┐
 │                   Bi-Encoder Retrieval                            │
 │                                                                  │
 │  Offline (indexing):                                              │
 │    doc₁ ──→ [Encoder] ──→ e₁ ──→ ┌─────────────┐               │
 │    doc₂ ──→ [Encoder] ──→ e₂ ──→ │ Vector Index │               │
 │    ...                    ...     │ (FAISS/HNSW) │               │
 │    docₙ ──→ [Encoder] ──→ eₙ ──→ └──────┬──────┘               │
 │                                          │                       │
 │  Online (search):                        │                       │
 │    query ──→ [Encoder] ──→ q ──→ ANN search                     │
 │                                          │                       │
 │                                   Top-k documents                │
 │                                                                  │
 │  Advantages:                                                     │
 │  ✓ Semantic matching ("car" matches "automobile")                │
 │  ✓ Sub-millisecond search (pre-computed embeddings)              │
 │  ✓ Works across languages (with multilingual models)             │
 │                                                                  │
 │  Disadvantages:                                                  │
 │  ✗ Misses exact keyword matches sometimes                        │
 │  ✗ Requires embedding model + vector index infrastructure        │
 │  ✗ Harder to debug than BM25                                     │
 └──────────────────────────────────────────────────────────────────┘
```

---

## **9.3 Cross-Encoders for Re-ranking**

Cross-encoders encode the (query, document) **pair together** — much more accurate but too slow for full corpus search.

```
 ┌──────────────────────────────────────────────────────────────────┐
 │              Cross-Encoder Re-ranking                             │
 │                                                                  │
 │  Input: [CLS] query [SEP] document [SEP]                        │
 │                                                                  │
 │  ┌──────────────────────────────────┐                            │
 │  │   Transformer (BERT/RoBERTa)      │                           │
 │  │   Full cross-attention between     │                          │
 │  │   query and document tokens        │                          │
 │  └────────────────┬─────────────────┘                            │
 │                   │                                              │
 │              [CLS] → Linear → sigmoid → relevance score (0–1)   │
 │                                                                  │
 │  Why more accurate:                                              │
 │  - Full token-level interaction between query and doc            │
 │  - Can capture: "query: president → doc mentions 'commander      │
 │    in chief'" (synonym via cross-attention)                      │
 │  - Bi-encoder can't do this (encodes independently)              │
 │                                                                  │
 │  Why slower:                                                     │
 │  - Must run transformer for EVERY (query, doc) pair              │
 │  - 1000 candidates × 1 forward pass = 1000 inferences           │
 │  - Bi-encoder: 1 query encoding + vector lookup = ~1ms           │
 └──────────────────────────────────────────────────────────────────┘
```

```python
from sentence_transformers import CrossEncoder

cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

query = "What is the termination clause?"
candidates = [
    "Either party may terminate this agreement with 30 days notice.",
    "The contract was signed on January 15, 2024.",
    "Termination for cause requires written notice specifying the breach.",
]

scores = cross_encoder.predict([(query, doc) for doc in candidates])
# [0.95, 0.02, 0.91]  — correctly identifies relevant documents
```

---

## **9.4 Hybrid Search**

Combines lexical (BM25) and semantic (dense) retrieval for the best of both worlds.

```
 ┌──────────────────────────────────────────────────────────────────┐
 │                    Hybrid Search Pipeline                         │
 │                                                                  │
 │  Query: "Section 12(b) termination clause"                       │
 │              │                                                   │
 │    ┌─────────┴────────────┐                                      │
 │    │                      │                                      │
 │    ▼                      ▼                                      │
 │  BM25 Search          Dense Search                               │
 │  (keyword match)     (semantic match)                            │
 │    │                      │                                      │
 │    │  Hits "12(b)"        │  Hits "termination",                 │
 │    │  exactly             │  "contract ending"                   │
 │    │                      │                                      │
 │    ▼                      ▼                                      │
 │  ┌──────────────────────────────┐                                │
 │  │   Reciprocal Rank Fusion     │                                │
 │  │   (RRF) or Weighted Sum      │                                │
 │  │                               │                               │
 │  │   RRF_score(d) =              │                               │
 │  │     1/(k + rank_bm25(d))      │                               │
 │  │   + 1/(k + rank_dense(d))     │                               │
 │  │                               │                               │
 │  │   k = 60 (typical)            │                               │
 │  └──────────────┬───────────────┘                                │
 │                 │                                                 │
 │                 ▼                                                 │
 │          Cross-Encoder Re-rank (optional)                        │
 │                 │                                                 │
 │                 ▼                                                 │
 │          Final ranked results                                    │
 └──────────────────────────────────────────────────────────────────┘
```

**Why hybrid works:**

| Query Type | BM25 Strength | Dense Strength | Winner |
|-----------|---------------|----------------|--------|
| Exact keyword: "GDPR Article 15" | Exact match | May miss exact terms | BM25 |
| Semantic: "data privacy laws in EU" | Misses if "GDPR" not in query | Understands meaning | Dense |
| Mixed: "Section 12(b) termination" | Hits "12(b)" | Hits "termination" semantics | Hybrid |

```python
# Hybrid search with reciprocal rank fusion
def hybrid_search(query, bm25_results, dense_results, k=60, alpha=0.5):
    scores = {}

    for rank, doc_id in enumerate(bm25_results):
        scores[doc_id] = scores.get(doc_id, 0) + alpha * (1 / (k + rank + 1))

    for rank, doc_id in enumerate(dense_results):
        scores[doc_id] = scores.get(doc_id, 0) + (1 - alpha) * (1 / (k + rank + 1))

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
```

---

# **10. Common Interview Questions with Strong Answers**

---

## **Q1: "Explain Word2Vec — how does it work?"**

**Strong answer:**

"Word2Vec learns dense vector representations of words by training a shallow neural network on a simple prediction task. There are two variants:

**Skip-gram** predicts context words given a center word. For example, given 'sat', it predicts 'cat', 'on', 'the'. The model learns embedding vectors where the dot product between center and context word embeddings approximates the log probability of co-occurrence.

**CBOW** does the reverse — predicts the center word from context words.

The key training trick is **negative sampling**: instead of computing a softmax over the entire vocabulary (which is expensive), we sample 5–20 random 'negative' words and train a binary classifier to distinguish real context words from random ones. This makes training tractable on large corpora.

The resulting embeddings capture semantic relationships — words appearing in similar contexts get similar vectors. This leads to famous properties like **vector arithmetic**: king - man + woman ≈ queen.

The main limitation is that each word gets a single static vector regardless of context — 'bank' in 'river bank' and 'bank account' get the same vector. This is the polysemy problem that contextual models like BERT solve."

---

## **Q2: "What's the difference between BERT and GPT architectures?"**

**Strong answer:**

"They both use transformers but differ fundamentally in architecture and pre-training:

**BERT** is an **encoder-only** model with **bidirectional** attention — every token can attend to every other token in both directions. It's pre-trained with **Masked Language Modeling** (predict masked tokens from full context). This makes it excellent for understanding tasks: classification, NER, question answering. However, it can't generate text natively.

**GPT** is a **decoder-only** model with **causal (unidirectional)** attention — each token can only attend to previous tokens. It's pre-trained with **Causal Language Modeling** (predict the next token). This makes it excellent for generation: text completion, code, dialogue. But for classification, it's less efficient than BERT because it doesn't see right context.

In practice, for my legal document classification project, I used **RoBERTa** (a BERT variant) because classification is an understanding task where bidirectional context is critical. For generating legal summaries, a GPT-style model would be more appropriate.

The broader distinction is: **encoders understand, decoders generate, encoder-decoders do sequence-to-sequence** (like T5 for translation or summarization)."

---

## **Q3: "What is BPE tokenization and why is it used?"**

**Strong answer:**

"BPE — Byte-Pair Encoding — is a subword tokenization algorithm used by GPT-2, GPT-3, GPT-4, and LLaMA. It solves the trade-off between word-level tokenization (huge vocabulary, can't handle OOV words) and character-level (tiny vocabulary but very long sequences).

The algorithm works by:
1. Starting with a vocabulary of individual characters
2. Counting all adjacent character pairs in the training corpus
3. Merging the most frequent pair into a new token
4. Repeating until reaching the desired vocabulary size (e.g., 50K)

For example, 'unbelievably' might tokenize as ['un', 'believ', 'ably']. Common words stay as single tokens ('the', 'and'), while rare words decompose into frequent subwords.

Key advantages: handles OOV words by decomposing them, captures morphology ('un-' prefix, '-ing' suffix are often tokens), produces a fixed-size vocabulary, and works across languages. The main alternative is **WordPiece** (used by BERT), which is similar but uses likelihood-based merging instead of frequency-based."

---

## **Q4: "How would you build a text classification system from scratch?"**

**Strong answer:**

"I'd follow a systematic approach, which is what I did for our legal document classification system:

**Step 1: Data & Baseline (Day 1–2)**
Start with labeled data. Even 500 examples per class can work. First, build a TF-IDF + Logistic Regression baseline — it takes 30 minutes and gives you a performance floor. In our legal project, this got us 78% F1.

**Step 2: Error Analysis (Day 2–3)**
Examine where the baseline fails. Are errors due to ambiguous labels? Class imbalance? Long documents? This guides model selection. We found that legal documents had domain-specific vocabulary the baseline couldn't capture.

**Step 3: Transformer Fine-tuning (Day 3–5)**
Fine-tune a pre-trained model. For English text, start with `roberta-base`. Key hyperparameters: learning rate 2e-5, batch size 16, 3 epochs, warmup 10% of steps. This jumped us to 91% F1.

**Step 4: Domain Adaptation (if needed)**
For specialized domains (legal, medical), consider continued pre-training on domain text before fine-tuning. We used a legal corpus for additional pre-training, gaining 3%.

**Step 5: Handle Edge Cases**
- Long documents: hierarchical approach (classify chunks, then aggregate)
- Class imbalance: focal loss, oversampling, class weights
- Low data: few-shot with GPT-4, data augmentation

**Step 6: Production**
Quantize (INT8) for latency, add confidence thresholds (reject uncertain predictions), set up monitoring for data drift. Our final pipeline ran in <50ms per document."

---

## **Q5: "Cosine similarity vs L2 distance — when would you use each?"**

**Strong answer:**

"**Cosine similarity** measures the angle between vectors, ignoring magnitude. It ranges from -1 to 1. **L2 (Euclidean) distance** measures the straight-line distance between vector endpoints.

For **normalized vectors** (L2 norm = 1), they're monotonically related: `‖A-B‖² = 2 - 2·cos(A,B)`. So maximizing cosine = minimizing L2 distance.

**Cosine similarity is preferred for text embeddings** because:
1. Most embedding models produce vectors where direction encodes semantics and magnitude is roughly constant
2. It's scale-invariant — a longer document doesn't get 'closer' to everything just because it has more content
3. It's the standard metric for SBERT, OpenAI embeddings, and most NLP benchmarks

**L2 distance is preferred for:**
1. Image embeddings (where magnitude can be meaningful)
2. Clustering (K-Means optimizes on L2 distance)
3. When working with already-normalized vectors (then they're equivalent, and L2 is slightly faster to compute)

In practice, many vector databases let you normalize on insert and use dot product (equivalent to cosine for normalized vectors) for maximum speed."

---

## **Q6: "What are contextual embeddings and why do they matter?"**

**Strong answer:**

"Static embeddings like Word2Vec assign one fixed vector per word, regardless of context. The word 'bank' gets the same vector whether it means a financial institution or a river bank. This is the polysemy problem.

**Contextual embeddings** (ELMo, BERT, GPT) produce **different vectors for the same word** depending on its surrounding context. BERT achieves this through bidirectional self-attention — when encoding 'bank' in 'river bank', it attends to 'river', producing a vector closer to 'shore' and 'water'. In 'deposit at the bank', it produces a vector closer to 'finance' and 'money'.

This matters because language is inherently ambiguous. In our call transcript classification project, we had phrases like 'this is not fair' (complaint) vs 'the fair begins tomorrow' (informational). Static embeddings couldn't distinguish these, but fine-tuned BERT could, improving classification accuracy significantly.

The progression was: Word2Vec (2013) → ELMo (2018, bidirectional LSTM) → BERT (2018, bidirectional transformer) → GPT/modern LLMs. Each step captured richer context, with transformers being the current standard due to their ability to model long-range dependencies through self-attention."

---

## **Q7: "Explain the difference between bi-encoders and cross-encoders."**

**Strong answer:**

"This is a fundamental trade-off in information retrieval between **speed and accuracy**.

**Bi-encoders** (like SBERT, DPR) encode the query and document **independently** into fixed vectors. Similarity is computed as cosine similarity between the two vectors. This enables pre-computing all document embeddings offline, so search is just an ANN lookup — sub-millisecond for millions of documents.

**Cross-encoders** (like ms-marco-MiniLM) process the query and document **together** as a single input: [CLS] query [SEP] document [SEP]. Every query token can attend to every document token and vice versa. This full cross-attention makes them much more accurate — they can capture nuanced relevance like synonym matching and contextual reasoning.

The trade-off:
- Bi-encoder: O(1) per query (after indexing). 1M docs searchable in milliseconds.
- Cross-encoder: O(n) per query. 1000 candidates takes ~1 second.

In practice, we use a **two-stage pipeline**: bi-encoder retrieves top-100 candidates, then cross-encoder re-ranks them. This gets the best of both worlds — speed of bi-encoders with accuracy of cross-encoders.

In our legal document retrieval system, this two-stage approach improved nDCG@10 by 12% over bi-encoder alone, while keeping latency under 200ms."

---

## **Q8: "What is Sentence-BERT and why was it needed?"**

**Strong answer:**

"Sentence-BERT addressed a critical limitation of vanilla BERT for sentence similarity tasks.

Naive BERT sentence embeddings (mean pooling or [CLS] token) are surprisingly bad for similarity — all sentences end up with cosine similarity of 0.8–0.95, making ranking impossible. This is because BERT was trained for MLM, not to produce semantically meaningful sentence-level vectors.

SBERT trains a **siamese network**: two copies of BERT with shared weights encode two sentences independently, then a training objective (contrastive loss on NLI data or regression on STS data) teaches the model to produce vectors where similar sentences are close and dissimilar ones are far apart.

The practical impact is enormous: finding the most similar pair among 10,000 sentences takes ~65 hours with BERT cross-encoder (50M pair comparisons) but only ~5 seconds with SBERT (10K encodings + cosine similarity matrix).

This is what makes modern semantic search, RAG systems, and embedding-based retrieval possible at scale. Models like `all-MiniLM-L6-v2` (384-d) give excellent quality at very low latency — I've used this in production for our document similarity features."

---

## **Q9: "What are Matryoshka embeddings?"**

**Strong answer:**

"Matryoshka Representation Learning, named after Russian nesting dolls, trains embeddings to be useful at **multiple dimensionalities** simultaneously.

The key insight: the first d dimensions of a 768-d embedding should form a valid d-dimensional embedding on their own. During training, the contrastive loss is applied at multiple truncation points (e.g., 64, 128, 256, 512, 768), so the model learns to pack the most important information into the first dimensions.

This gives you a flexible quality-cost trade-off at inference time: use 64 dimensions for fast coarse filtering, 256 for good quality, 768 for maximum quality — all from the same model and same forward pass.

OpenAI's `text-embedding-3-small` supports this natively — you just pass `dimensions=256` and get a 256-d embedding that's ~1-2% worse than the full 1536-d version but 6x smaller in storage and faster for similarity search.

This is particularly valuable in production where you might use low-dimensional embeddings for initial retrieval from millions of documents, then full-dimensional embeddings for final re-ranking on a smaller candidate set."

---

# **11. Key Takeaways**

---

```
 ┌──────────────────────────────────────────────────────────────────┐
 │                   KEY INTERVIEW TAKEAWAYS                        │
 │                                                                  │
 │  1. TOKENIZATION                                                 │
 │     BPE (GPT) and WordPiece (BERT) are the modern standards.    │
 │     Know the difference: frequency-based vs likelihood-based     │
 │     merging. SentencePiece is language-independent.              │
 │                                                                  │
 │  2. STATIC vs CONTEXTUAL EMBEDDINGS                              │
 │     Word2Vec/GloVe/FastText: one vector per word (static).      │
 │     BERT/GPT: different vectors per context (contextual).        │
 │     The "bank" example is your go-to illustration.               │
 │                                                                  │
 │  3. SENTENCE EMBEDDINGS                                          │
 │     Naive BERT ≠ good sentence embeddings.                       │
 │     SBERT (siamese network + contrastive loss) is the fix.      │
 │     OpenAI, Cohere, BGE, E5 are production-ready options.       │
 │                                                                  │
 │  4. SIMILARITY                                                   │
 │     Cosine similarity is default for text. For normalized        │
 │     vectors: cosine = dot product = f(L2 distance).              │
 │     Know the formula and when they're equivalent.                │
 │                                                                  │
 │  5. CONTRASTIVE LEARNING                                         │
 │     InfoNCE is the dominant loss for embedding training.         │
 │     Temperature, hard negatives, and batch size are critical.    │
 │     Matryoshka embeddings offer flexible dimensionality.         │
 │                                                                  │
 │  6. LANGUAGE MODELS                                              │
 │     Encoders (BERT): bidirectional, best for understanding.     │
 │     Decoders (GPT): autoregressive, best for generation.        │
 │     Encoder-Decoders (T5): best for seq2seq tasks.              │
 │     Know MLM, CLM, and denoising objectives.                    │
 │                                                                  │
 │  7. TEXT CLASSIFICATION PROGRESSION                              │
 │     TF-IDF + LogReg → CNN/LSTM → Transformer fine-tuning        │
 │     → Few-shot LLM. Always start with a simple baseline.        │
 │                                                                  │
 │  8. INFORMATION RETRIEVAL                                        │
 │     BM25 (lexical) + Dense (semantic) = Hybrid search.          │
 │     Bi-encoders for speed, cross-encoders for accuracy.         │
 │     Two-stage pipeline: retrieve → re-rank.                     │
 │                                                                  │
 │  9. YOUR DIFFERENTIATOR                                          │
 │     You've built NLP classification systems end-to-end:          │
 │     legal docs and call transcripts. You know the full stack    │
 │     from TF-IDF baselines to production transformer pipelines.  │
 │     Lead with concrete numbers: F1 scores, latency, data sizes. │
 └──────────────────────────────────────────────────────────────────┘
```

---

**Quick Reference Card:**

| Topic | Key Formula / Concept | Interview Mention |
|-------|----------------------|-------------------|
| BPE | Merge most frequent byte pairs iteratively | GPT tokenizer |
| Word2Vec | Skip-gram: P(context \| center) with negative sampling | Vector arithmetic: king - man + woman = queen |
| GloVe | wᵢᵀ · w̃ⱼ + biases ≈ log(Xᵢⱼ) | Global co-occurrence matrix factorization |
| FastText | Word = sum of character n-gram embeddings | Handles OOV, morphological awareness |
| TF-IDF | TF × log(N/df) | Baseline feature, evolved into BM25 |
| Cosine Similarity | A·B / (\|\|A\|\|·\|\|B\|\|), range [-1, 1] | Default for text, angle-based |
| InfoNCE | -log(exp(sim+/τ) / Σ exp(sim/τ)) | Core loss for embedding training |
| BERT | Bidirectional, MLM + NSP, encoder-only | Understanding tasks: classification, NER |
| GPT | Unidirectional, CLM, decoder-only | Generation tasks: text, code, chat |
| SBERT | Siamese BERT + contrastive training | Efficient sentence similarity |
| BM25 | TF saturation + length normalization | Standard lexical retrieval baseline |
| Hybrid Search | BM25 + Dense + RRF fusion | Production search systems |

---

*Last updated: February 2026*
