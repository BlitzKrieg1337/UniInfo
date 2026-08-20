# UniInfo 🎓

A RAG-powered assistant for exploring Masters program admission requirements — ask questions in plain English and get answers grounded in and cited to the university's own admissions pages.

## Why this exists

Admission requirements for Masters programs are rarely laid out cleanly in one place — they're scattered across long admissions pages mixed in with unrelated content (fees, modules, campus info), and non-obvious hard requirements (like specific credit-hour or degree-equivalency rules) are easy to miss when reading through a page rather than being able to ask about it directly. UniInfo exists to make that information easy to find and easy to ask about — grounded in and cited to the university's own pages, instead of relying on a manual read-through.

## What it does

- Select a university and program from the sidebar
- Ask questions in natural language — requirements, deadlines, course structure, fees, whatever the source documents cover
- Get an answer grounded strictly in the retrieved source material, with a citation back to the originating document
- State your own profile in a question (e.g. *"I have a 6.5 IELTS score, does that meet the requirement?"*) and get an honest comparison — the assistant checks it against whatever is explicitly stated, and says so plainly when a requirement isn't a hard number or isn't specified at all, rather than guessing

## What it deliberately does not do

It does not predict admission chances. Admission is competitive and depends on factors no public document states — applicant pool strength, personal statement quality, references. The system is prompted to compare stated facts, not render a verdict on whether someone will be admitted.

## How it works

Four-stage pipeline:

1. **Extraction** (`extraction.py`) — parses admission pages (HTML) and program documents (PDF) via [Docling](https://github.com/docling-project/docling), which preserves table structure and strips page navigation/boilerplate more reliably than plain text extraction. Output is saved as Markdown per program.
2. **Embedding** (`embedding.py`) — loads the Markdown, splits it into chunks (`RecursiveCharacterTextSplitter`), embeds with a HuggingFace sentence-transformer (`all-MiniLM-L6-v2`), and stores everything in a local ChromaDB vector store. Each chunk is tagged with `college` and `program` metadata for scoped retrieval.
3. **Query** (`query.py`) — retrieval and answer generation, filtered to the selected university and program. Four retrieval strategies are available and user-selectable:
   - **Vector search** — standard semantic similarity search
   - **BM25** — keyword/lexical search, useful for exact terms (e.g. specific test names, course codes)
   - **Hybrid search** — vector + BM25 combined via Reciprocal Rank Fusion
   - **Multi-query** — the question is rewritten into several variants before retrieval, to catch relevant content phrased differently than the original question
4. **App** (`app.py`) — a Streamlit interface: pick a university and program, pick a retrieval strategy, ask a question, get a cited answer.


## Tech stack

- **Parsing**: Docling
- **Orchestration**: LangChain
- **Embeddings**: HuggingFace `sentence-transformers` (`all-MiniLM-L6-v2`)
- **Vector store**: ChromaDB
- **Keyword search**: `rank_bm25` (via LangChain's `BM25Retriever`)
- **LLM**: Google Gemini
- **UI**: Streamlit
- **Tracing** (optional): LangSmith

## Project structure

```
UniInfo/
├── data/
│   └── {University Name}/
│       └── {Program Name}.md
├── extraction.py       # parses raw sources into Markdown
├── embedding.py        # chunks, embeds, and stores in ChromaDB
├── query.py            # retrieval strategies + answer generation
├── app.py              # Streamlit UI
├── requirements.txt
├── .gitignore
└── README.md
```

## Note on requirement comparison

An earlier version of this project planned a separate, rule-based comparison module — a lookup of numeric minimum requirements (CGPA, IELTS) per program, checked against a user's profile in code rather than through the LLM.

Once real admissions data was collected, that assumption didn't hold: test scores (IELTS/TOEFL) are typically stated as explicit thresholds, but academic background requirements are usually expressed as a degree classification (e.g. "an upper second-class honours degree, 2.1, or equivalent") rather than a GPA cutoff — and there's no standardized conversion between international grading scales and this classification. A numeric lookup table would have been accurate for one field and empty or misleading for the rest.

Profile-comparison questions are instead handled entirely within the retrieval-and-generation pipeline: the prompt restricts the model to comparing only what's explicitly stated in retrieved source text, and instructs it to flag plainly when a requirement is qualitative, unspecified, or not a direct match — rather than infer a numeric equivalence the source material doesn't support.

## Setup

```bash
git clone https://github.com/BlitzKrieg1337/UniInfo.git
cd UniInfo

python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

Create a `.env` file in the project root:

```
api_key=your_gemini_api_key
langsmith_apikey=your_langsmith_key   # optional, enables tracing
```

To add a new source document:

```bash
python extraction.py
# prompts for: source URL/path, file name, folder name (university)
```

Build (or rebuild) the vector store from everything in `data/`:

```bash
python embedding.py
```

Launch the app:

```bash
streamlit run app.py
```

## Currently covered

- **Trinity College Dublin** — MSc Computer Science (Intelligent Systems), MSc Computer Science (Data Science)
- **University College Dublin** — MSc Advanced Artificial Intelligence
- **Dublin City University** — MSc in Computing

Scoped to Masters programs in Data/AI/Computing for now. Adding a new program is a two-command process (`extraction.py` then `embedding.py`), so the list is expected to grow.

## Roadmap

- Retrieval strategy evaluation — a small test question set to measure which of the four strategies performs best, and set that as the default
- A few additional programs/universities
- Deployment (Streamlit Community Cloud)

## Disclaimer

Information is sourced from official university pages at the time of extraction and may become outdated. Always confirm final details with the university's official admissions office before making application decisions.