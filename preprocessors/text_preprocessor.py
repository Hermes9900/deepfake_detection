import spacy
from newspaper import Article

nlp = spacy.load("en_core_web_sm")

def preprocess_text(url: str = None, html_content: str = None):
    """
    Extract and normalize text from HTML or URL.
    Returns a dictionary with 'text', 'sentences', 'entities'.
    """
    if url:
        article = Article(url)
        article.download()
        article.parse()
        text = article.text
    elif html_content:
        article = Article('')
        article.set_html(html_content)
        article.parse()
        text = article.text
    else:
        raise ValueError("Provide either URL or HTML content")

    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    
    return {"text": text, "sentences": sentences, "entities": entities}
