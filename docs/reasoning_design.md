# Reasoning Design: Multilingual & Code-Centric Logic for MIR-LM

This document explores how to achieve the most efficient reasoning capability for MIR-LM (135M), especially in a multilingual context, and whether English is strictly necessary as the "reasoning vehicle."

---

## 1. The "Token Tax" Challenge in Multilingual LLMs

Standard tokenizers (trained primarily on English) represent Indian languages (Hindi, Tamil, Bengali, etc.) highly inefficiently. 
- An English word like `algorithm` is usually 1 token.
- The Hindi translation `कलन विधि` or transliteration might be split into 6–8 tokens (or raw bytes) by a standard tokenizer.
- This "token tax" means that if the model attempts to reason in an Indian language, it consumes $3\times$ to $5\times$ more context window and compute steps than if it reasoned in English.

### Solution: Vocabulary Tuning
When we train our custom BPE tokenizer from scratch, we will ensure that:
1. The training corpus contains a balanced representation of Indian language texts (approx. 30%).
2. The tokenizer merges common syllables, words, and characters of Indian scripts (Devanagari, Tamil script, etc.) into single, unified tokens.
3. This levels the playing field, making reasoning in local languages computationally comparable to English.

---

## 2. Is Reasoning in English Necessary?

While English is not strictly necessary for general human reasoning, it holds a unique position in software engineering:
1. **Syntax & APIs**: Programming language keywords (`def`, `async`, `class`, `import`) and almost all libraries (standard libraries, frameworks) are written in English.
2. **Dataset Density**: 95%+ of high-quality code documentation, StackOverflow threads, and GitHub issues are in English.
3. **Cross-Language Generalization**: A model that can map a prompt in Hindi or Tamil to an English code structure must have some internal alignment where English code tokens act as the bridge.

Therefore, the model *must* understand English code structures, but it does **not** need to perform natural language reasoning in English to solve coding tasks.

---

## 3. Alternative Reasoning Vehicles

For a small 135M model, natural language reasoning (whether in English or Hindi) is often verbose and difficult for the model to generate cleanly. We can explore more efficient **digital mechanisms** and "languages of thought":

### A. Code-as-Thought (Pythonic Scratchpads)
Instead of asking the model to write a paragraph explaining how it will solve a problem, we train the model to write a **compressed pseudo-code scratchpad** inside a `<think>` block before outputting the final code.
- **Why it's efficient**: Code has zero ambiguity. A small model is much better at following strict structural syntax (like pseudo-code or Python variable updates) than the complex syntax of natural language.
- **Example**:
  ```
  User: [Task in Hindi]
  Model:
  <think>
  # 1. Input: array of numbers
  # 2. Filter even numbers: [x for x in arr if x % 2 == 0]
  # 3. Sort descending: sort(reverse=True)
  </think>
  [Final Python Code]
  ```

### B. Symbolic Logic & Compressed Dialect
We can train the tokenizer and the dataset to use a compressed notation for logical reasoning (resembling mathematical logic or prefix notation), which reduces token count and focuses the model's small parameter capacity on pure logical transitions rather than grammar.

### C. Multilingual Grounding
We will pre-train the model on parallel text-code pairs where comments are in local languages (e.g. Hindi/Tamil comments explaining Python code blocks). This grounds the programming concepts directly into the local languages, bypassing the need to translate to English text first.

---

## 4. Architectural Summary for MIR-LM
- **Input/Output**: Prompts can be in English or Indian languages. Output is clean, working code.
- **Reasoning Loop**: The model will be fine-tuned/pre-trained to use structured `<think>` tags containing python-style logical comments (highly compressed) as its internal chain-of-thought, rather than conversational English.
