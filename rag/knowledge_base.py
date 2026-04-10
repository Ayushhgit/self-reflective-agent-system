"""
A curated set of text passages used to seed the FAISS vector store on first
run.  The passages cover core AI/ML topics commonly queried during demos.
"""

DEFAULT_DOCUMENTS = [
    # ── Transformers ──────────────────────────────────────────────────────────
    """The Transformer architecture, introduced in "Attention Is All You Need"
(Vaswani et al., 2017), replaced recurrent networks with self-attention
mechanisms. A Transformer processes sequences in parallel by computing
attention scores between every pair of tokens. The scaled dot-product
attention is: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V.
The multi-head variant runs h attention heads in parallel, enabling the model
to jointly attend to information at different positions and subspaces.""",

    """BERT (Bidirectional Encoder Representations from Transformers) is a
pre-trained language model that uses masked language modelling (MLM) and
next-sentence prediction (NSP) as pre-training objectives. GPT (Generative
Pre-trained Transformer) uses autoregressive language modelling (predict the
next token). Both are based on the Transformer architecture but differ in
how they use context: BERT is bidirectional, GPT is left-to-right.""",

    # ── RAG ───────────────────────────────────────────────────────────────────
    """Retrieval-Augmented Generation (RAG) combines a dense retriever and a
generative model. Given a query, RAG first retrieves relevant passages from a
knowledge corpus using an embedding-based similarity search (e.g., FAISS or
ChromaDB). The retrieved passages are concatenated with the query and fed to
a generative LLM. RAG reduces hallucinations and allows the model to access
up-to-date information without retraining.""",

    """Fine-tuning vs RAG: Fine-tuning updates model weights on a task-specific
dataset, making knowledge parametric (baked into weights). RAG keeps
knowledge external (in a vector store) so it can be updated without
retraining. RAG is preferred when facts change frequently; fine-tuning is
preferred when a consistent style or specialised capability is needed.""",

    # ── LangGraph ─────────────────────────────────────────────────────────────
    """LangGraph is a library for building stateful, multi-actor applications
with LLMs. It models workflows as directed graphs where nodes are processing
units (Python functions or Runnables) and edges define data flow. Conditional
edges support branching logic. LangGraph's StateGraph persists a shared
TypedDict across all nodes, enabling complex multi-step reasoning pipelines
with loops and parallel branches.""",

    # ── LangChain ─────────────────────────────────────────────────────────────
    """LangChain is a framework for developing applications powered by language
models. Key components include: (1) Chains – sequences of calls to LLMs and
tools; (2) Agents – LLMs that decide which tools to use dynamically;
(3) Memory – persistence of conversation context; (4) Retrieval – RAG
pipelines with vector stores; (5) Callbacks – observability and logging hooks.
LangChain abstracts over many LLM providers (OpenAI, Anthropic, Groq, etc.).""",

    # ── LLM Evaluation ────────────────────────────────────────────────────────
    """LLM evaluation metrics include: BLEU (n-gram overlap with reference),
ROUGE (recall-oriented overlap), BERTScore (semantic similarity via BERT
embeddings), and model-based evaluators (GPT-4 judging quality dimensions
such as correctness, fluency, and faithfulness). For RAG, faithfulness
(answer grounded in retrieved context) and context recall are important.""",

    # ── Python Algorithms ────────────────────────────────────────────────────
    """Common Python sorting algorithms: (1) Timsort (Python built-in, O(n log n))
uses insertion sort for small arrays and merge sort otherwise. (2) Quicksort
uses a pivot to partition – average O(n log n), worst O(n^2). (3) Merge sort
is stable and always O(n log n) but requires O(n) extra space. Python's
list.sort() and sorted() both use Timsort and are highly optimised.""",

    # ── Neural Networks ──────────────────────────────────────────────────────
    """A neural network is a function approximator composed of layers of
neurons. Each neuron computes: output = activation(W·x + b). Activation
functions include ReLU (max(0,x)), Sigmoid (1/(1+e^-x)), and Tanh. Training
uses gradient descent with backpropagation to minimise a loss function
(e.g., cross-entropy for classification, MSE for regression). Regularisation
techniques include dropout, L2 weight decay, and batch normalisation.""",

    # ── Agentic AI ───────────────────────────────────────────────────────────
    """Agentic AI systems are AI models that autonomously plan, use tools, and
take multi-step actions to achieve goals. Key capabilities include: (1) Tool
use – calling external APIs or code executors; (2) Planning – decomposing
goals into sub-tasks; (3) Memory – retaining context across steps;
(4) Self-reflection – evaluating outputs and retrying when needed.
Frameworks like LangGraph, AutoGen, and CrewAI facilitate building these
systems.""",

    # ── Vector Databases ─────────────────────────────────────────────────────
    """Vector databases store high-dimensional embedding vectors and support
approximate nearest-neighbour (ANN) search. Popular options: FAISS (Facebook,
in-memory, GPU-accelerated), ChromaDB (open-source, persistent), Pinecone
(managed cloud service), Weaviate (supports hybrid search), and Qdrant
(Rust-based, filterable). Indexing methods include IVF, HNSW, and PQ.""",
]
