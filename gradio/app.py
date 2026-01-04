import os
import re
import xml.etree.ElementTree as ET
import gradio as gr
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from huggingface_hub import InferenceClient
from sentence_transformers import SentenceTransformer

# Configuration
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
HOPSWORKS_HOST = os.environ.get("HOPSWORKS_HOST", "c.app.hopsworks.ai")
HOPSWORKS_FEATURE_STORE = os.environ.get(
    "HOPSWORKS_FEATURE_STORE", "kingaedwin_featurestore"
)
FEATURE_GROUP_NAME = "arxiv_embeddings_with_cats"
FEATURE_GROUP_VERSION = 1
ARXIV_CATEGORIES_URL = "https://raw.githubusercontent.com/Edwinexd/arxiv-rag-agent/refs/heads/master/data/arxiv_v2.csv"

# Global state for lazy loading
_embedding_model = None
_hopsworks_data = None
_categories_df = None


def get_embedding_model():
    """Lazy load the embedding model."""
    global _embedding_model
    if _embedding_model is None:
        print("Loading embedding model...")
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        print(f"Model loaded. Dimension: {_embedding_model.get_sentence_embedding_dimension()}")
    return _embedding_model


def load_categories():
    """Load arXiv categories from CSV."""
    global _categories_df
    if _categories_df is None:
        print("Loading categories...")
        _categories_df = pd.read_csv(ARXIV_CATEGORIES_URL)
        print(f"Loaded {len(_categories_df)} categories")
    return _categories_df


def get_main_categories():
    """Get list of main categories."""
    df = load_categories()
    return sorted(df["main_category"].unique().tolist())


def get_subcategories(main_category: str) -> list[str]:
    """Get subcategories for a main category."""
    df = load_categories()
    if main_category == "All":
        return ["All"] + sorted(df["code"].tolist())
    filtered = df[df["main_category"] == main_category]
    return ["All"] + sorted(filtered["code"].tolist())


def connect_to_hopsworks():
    """Connect to Hopsworks and load feature data."""
    global _hopsworks_data
    if _hopsworks_data is not None:
        return _hopsworks_data

    try:
        import hopsworks

        print("Connecting to Hopsworks...")
        project = hopsworks.login(host=HOPSWORKS_HOST)
        fs = project.get_feature_store(name=HOPSWORKS_FEATURE_STORE)
        fg = fs.get_feature_group(
            name=FEATURE_GROUP_NAME,
            version=FEATURE_GROUP_VERSION,
        )
        print("Reading feature data...")
        _hopsworks_data = fg.read()
        print(f"Loaded {len(_hopsworks_data)} papers from Hopsworks")
        return _hopsworks_data
    except Exception as e:
        print(f"Error connecting to Hopsworks: {e}")
        return None


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def fetch_arxiv_papers(arxiv_ids: list[float]) -> dict[float, dict]:
    """Fetch paper details from arXiv API.

    Args:
        arxiv_ids: List of arXiv IDs as floats (e.g., 2401.12345)

    Returns:
        Dict mapping arxiv_id to paper details (title, abstract, authors)
    """
    if not arxiv_ids:
        return {}

    # Convert float IDs to string format for arXiv API
    id_strings = [str(aid) for aid in arxiv_ids]
    id_list = ",".join(id_strings)

    url = f"http://export.arxiv.org/api/query?id_list={id_list}&max_results={len(arxiv_ids)}"

    try:
        print(f"[DEBUG] Fetching {len(arxiv_ids)} papers from arXiv API...")
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        # Parse XML response
        root = ET.fromstring(response.content)

        # Define namespace
        ns = {
            "atom": "http://www.w3.org/2005/Atom",
            "arxiv": "http://arxiv.org/schemas/atom",
        }

        papers = {}
        for entry in root.findall("atom:entry", ns):
            # Extract ID from the entry
            id_elem = entry.find("atom:id", ns)
            if id_elem is None:
                continue

            # Parse ID: http://arxiv.org/abs/2401.12345v1 -> 2401.12345
            id_text = id_elem.text
            match = re.search(r"abs/([\d.]+)", id_text)
            if not match:
                continue

            arxiv_id = float(match.group(1))

            # Extract title
            title_elem = entry.find("atom:title", ns)
            title = title_elem.text.strip().replace("\n", " ") if title_elem is not None else "Unknown"

            # Extract abstract
            summary_elem = entry.find("atom:summary", ns)
            abstract = summary_elem.text.strip().replace("\n", " ") if summary_elem is not None else "No abstract available"

            # Extract authors
            authors = []
            for author in entry.findall("atom:author", ns):
                name_elem = author.find("atom:name", ns)
                if name_elem is not None:
                    authors.append(name_elem.text)

            papers[arxiv_id] = {
                "title": title,
                "abstract": abstract,
                "authors": ", ".join(authors),
            }

        print(f"[DEBUG] Successfully fetched {len(papers)} paper details")
        return papers

    except Exception as e:
        print(f"[DEBUG] Error fetching from arXiv API: {e}")
        return {}


def fetch_arxiv_html(arxiv_id: float, max_chars: int = 15000) -> str | None:
    """Fetch full paper text from arXiv HTML version.

    Args:
        arxiv_id: arXiv ID as float (e.g., 2511.17836)
        max_chars: Maximum characters to return (to limit token usage)

    Returns:
        Extracted text content or None if not available
    """
    # Convert float to string ID
    id_str = str(arxiv_id)

    # Try with v1 first (most common)
    url = f"https://arxiv.org/html/{id_str}v1"

    try:
        print(f"[DEBUG] Fetching HTML for arXiv:{id_str}...")
        response = requests.get(url, timeout=15)

        # If v1 fails, the paper might not have HTML version
        if response.status_code == 404:
            print(f"[DEBUG] No HTML version available for arXiv:{id_str}")
            return None

        response.raise_for_status()

        # Parse HTML
        soup = BeautifulSoup(response.content, "html.parser")

        # Remove script, style, nav elements
        for element in soup(["script", "style", "nav", "header", "footer", "aside"]):
            element.decompose()

        # Try to find the main article content
        # arXiv HTML uses <article> or <main> or <div class="ltx_page_content">
        content = None

        # Try different selectors for arXiv HTML format
        selectors = [
            "article.ltx_document",
            "div.ltx_page_content",
            "main",
            "article",
            "div.content",
        ]

        for selector in selectors:
            content = soup.select_one(selector)
            if content:
                break

        if not content:
            content = soup.body if soup.body else soup

        # Extract text
        text = content.get_text(separator="\n", strip=True)

        # Clean up excessive whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)

        # Truncate if too long
        if len(text) > max_chars:
            text = text[:max_chars] + "\n\n[... truncated for length ...]"

        print(f"[DEBUG] Fetched {len(text)} chars of HTML content for arXiv:{id_str}")
        return text

    except requests.exceptions.Timeout:
        print(f"[DEBUG] Timeout fetching HTML for arXiv:{id_str}")
        return None
    except Exception as e:
        print(f"[DEBUG] Error fetching HTML for arXiv:{id_str}: {e}")
        return None


def search_papers(
    query: str,
    main_category: str = "All",
    sub_category: str = "All",
    top_k: int = 5,
) -> list[dict]:
    """Search for relevant papers using embedding similarity.

    Args:
        query: User's search query
        main_category: Main category filter (e.g., "Computer Science")
        sub_category: Subcategory code filter (e.g., "cs.AI")
        top_k: Number of top results to return

    Returns:
        List of papers with id, categories, sub_categories, similarity, and arxiv details
    """
    print(f"\n{'='*60}")
    print(f"[DEBUG] SEARCH PAPERS")
    print(f"{'='*60}")
    print(f"[DEBUG] Query: {query}")
    print(f"[DEBUG] Main Category: {main_category}")
    print(f"[DEBUG] Sub Category: {sub_category}")

    data = connect_to_hopsworks()
    if data is None:
        print("[DEBUG] No data from Hopsworks!")
        return []

    print(f"[DEBUG] Total papers in Hopsworks: {len(data)}")

    model = get_embedding_model()
    query_embedding = model.encode(query, convert_to_numpy=True)

    # Filter by category if specified
    filtered_data = data.copy()

    if main_category != "All":
        filtered_data = filtered_data[
            filtered_data["categories"].apply(
                lambda x: main_category in x if isinstance(x, list) else main_category == x
            )
        ]
        print(f"[DEBUG] After main category filter: {len(filtered_data)} papers")

    if sub_category != "All":
        filtered_data = filtered_data[
            filtered_data["sub_categories"].apply(
                lambda x: sub_category in x if isinstance(x, list) else sub_category == x
            )
        ]
        print(f"[DEBUG] After sub category filter: {len(filtered_data)} papers")

    if len(filtered_data) == 0:
        print("[DEBUG] No papers match the category filter!")
        return []

    # Compute similarities
    print(f"[DEBUG] Computing similarities for {len(filtered_data)} papers...")
    similarities = []
    for idx, row in filtered_data.iterrows():
        emb = np.array(row["embedding"])
        sim = cosine_similarity(query_embedding, emb)
        similarities.append({
            "id": row["id"],
            "categories": row["categories"],
            "sub_categories": row["sub_categories"],
            "similarity": sim,
        })

    # Sort by similarity and return top_k
    similarities.sort(key=lambda x: x["similarity"], reverse=True)
    top_results = similarities[:top_k]

    # Fetch paper details from arXiv
    arxiv_ids = [p["id"] for p in top_results]
    arxiv_details = fetch_arxiv_papers(arxiv_ids)

    # Enrich results with arXiv details and full text
    for paper in top_results:
        details = arxiv_details.get(paper["id"], {})
        paper["title"] = details.get("title", "Title not available")
        paper["abstract"] = details.get("abstract", "Abstract not available")
        paper["authors"] = details.get("authors", "Unknown authors")

        # Fetch full text from HTML (only for top 3 to limit latency)
        if top_results.index(paper) < 3:
            full_text = fetch_arxiv_html(paper["id"])
            paper["full_text"] = full_text
        else:
            paper["full_text"] = None

    # Debug output for matches
    print(f"\n[DEBUG] TOP {len(top_results)} MATCHES:")
    print("-" * 60)
    for i, paper in enumerate(top_results, 1):
        has_full_text = "YES" if paper.get("full_text") else "NO"
        print(f"  {i}. arXiv:{paper['id']} (similarity: {paper['similarity']:.4f})")
        print(f"     Title: {paper['title'][:80]}...")
        print(f"     Categories: {paper['categories']} / {paper['sub_categories']}")
        print(f"     Full text: {has_full_text}")
    print("-" * 60)

    return top_results


def format_context(papers: list[dict]) -> str:
    """Format retrieved papers as context for the LLM."""
    if not papers:
        return "No relevant papers found."

    context_parts = []
    for i, paper in enumerate(papers, 1):
        arxiv_id = paper["id"]
        title = paper.get("title", "Title not available")
        abstract = paper.get("abstract", "Abstract not available")
        authors = paper.get("authors", "Unknown authors")
        full_text = paper.get("full_text")
        categories = paper["categories"]
        sub_cats = paper["sub_categories"]
        sim = paper["similarity"]

        # Format the paper with full details
        paper_text = (
            f"--- Paper {i} ---\n"
            f"arXiv ID: {arxiv_id}\n"
            f"Title: {title}\n"
            f"Authors: {authors}\n"
            f"Categories: {categories} ({sub_cats})\n"
            f"Relevance Score: {sim:.3f}\n"
            f"Link: https://arxiv.org/abs/{arxiv_id}\n"
        )

        if full_text:
            paper_text += f"\n=== Full Paper Content ===\n{full_text}"
        else:
            paper_text += f"\nAbstract: {abstract}"

        context_parts.append(paper_text)

    return "\n\n".join(context_parts)


def build_system_prompt(papers: list[dict], main_category: str, sub_category: str) -> str:
    """Build system prompt with retrieved paper context."""
    context = format_context(papers)

    category_info = ""
    if main_category != "All":
        category_info = f"The user is interested in {main_category}"
        if sub_category != "All":
            category_info += f", specifically {sub_category}"
        category_info += ".\n\n"

    system_prompt = f"""You are an expert research assistant specializing in arXiv papers. {category_info}You help users understand and explore scientific papers.

Based on the user's query, here are the most relevant papers from the arXiv database:

{context}

When answering:
1. Reference specific papers by their arXiv ID and title when relevant
2. Provide direct links to papers (https://arxiv.org/abs/[id])
3. Summarize key findings from the abstracts provided
4. Explain concepts clearly and connect ideas across papers when applicable
5. If asked about topics not covered by the retrieved papers, acknowledge this and provide general guidance
6. Suggest how the user might explore related research directions"""

    # Debug output
    print(f"\n{'='*60}")
    print(f"[DEBUG] SYSTEM PROMPT")
    print(f"{'='*60}")
    print(system_prompt[:2000])
    if len(system_prompt) > 2000:
        print(f"... [truncated, total length: {len(system_prompt)} chars]")
    print(f"{'='*60}\n")

    return system_prompt


def respond(
    message: str,
    history: list[dict],
    main_category: str,
    sub_category: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    hf_token: gr.OAuthToken | None,
):
    """Generate a response using RAG with Hopsworks embeddings."""
    if hf_token is None:
        yield "Please log in with your Hugging Face account to use the chatbot."
        return

    # Status: Searching
    yield "🔍 Searching for relevant papers..."

    # Search for relevant papers
    papers = search_papers(
        query=message,
        main_category=main_category,
        sub_category=sub_category,
        top_k=5,
    )

    # Status: Building context
    num_papers = len(papers)
    num_with_fulltext = sum(1 for p in papers if p.get("full_text"))
    yield f"📚 Found {num_papers} papers ({num_with_fulltext} with full text). Building context..."

    # Build system prompt with context
    system_prompt = build_system_prompt(papers, main_category, sub_category)

    # Status: Generating
    yield f"🤖 Generating response using {num_papers} papers as context..."

    # Create inference client
    client = InferenceClient(token=hf_token.token, model="meta-llama/Llama-3.3-70B-Instruct")

    # Build message list
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history)
    messages.append({"role": "user", "content": message})

    # Stream response
    response = ""
    for chunk in client.chat_completion(
        messages,
        max_tokens=max_tokens,
        stream=True,
        temperature=temperature,
        top_p=top_p,
    ):
        if chunk.choices and chunk.choices[0].delta.content:
            response += chunk.choices[0].delta.content
            yield response


def update_subcategories(main_category: str):
    """Update subcategory dropdown based on selected main category."""
    subcats = get_subcategories(main_category)
    return gr.Dropdown(choices=subcats, value="All")


# Build UI
with gr.Blocks(title="arXiv RAG Agent") as demo:
    gr.Markdown(
        """
        # arXiv RAG Agent

        Ask questions about scientific papers from arXiv. Filter by category to focus your search.
        The chatbot retrieves relevant papers from a Hopsworks feature store and uses them as context.

        **To use this app:**
        1. Enable third-party cookies in your browser (required for authentication)
        2. Sign in to Hugging Face using the login button in the left sidebar
        """
    )

    with gr.Sidebar():
        gr.LoginButton()
        gr.Markdown("### Category Filters")

        main_category = gr.Dropdown(
            choices=["All"] + get_main_categories(),
            value="All",
            label="Main Category",
            interactive=True,
        )

        sub_category = gr.Dropdown(
            choices=["All"],
            value="All",
            label="Subcategory",
            interactive=True,
        )

        main_category.change(
            fn=update_subcategories,
            inputs=[main_category],
            outputs=[sub_category],
        )

        gr.Markdown("### Generation Settings")

        max_tokens = gr.Slider(
            minimum=64,
            maximum=2048,
            value=512,
            step=64,
            label="Max tokens",
        )

        temperature = gr.Slider(
            minimum=0.1,
            maximum=2.0,
            value=0.7,
            step=0.1,
            label="Temperature",
        )

        top_p = gr.Slider(
            minimum=0.1,
            maximum=1.0,
            value=0.95,
            step=0.05,
            label="Top-p",
        )

    chatbot = gr.ChatInterface(
        fn=respond,
        type="messages",
        additional_inputs=[
            main_category,
            sub_category,
            max_tokens,
            temperature,
            top_p,
        ],
    )


if __name__ == "__main__":
    demo.launch()
