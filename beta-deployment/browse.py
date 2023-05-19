import requests
from bs4 import BeautifulSoup
from config import Config
import openai
openai.api_key = "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"


# Define and check for local file address prefixes
def check_local_file_access(url):
    local_prefixes = ['file:///', 'file://localhost', 'http://localhost', 'https://localhost']
    return any(url.startswith(prefix) for prefix in local_prefixes)

def scrape_text(url):
    """Scrape text from a webpage"""
    # Most basic check if the URL is valid:
    if not url.startswith('http'):
        return "Error: Invalid URL"

    # Restrict access to local files
    if check_local_file_access(url):
        return "Error: Access to local files is restricted"
    
    try:
        response = requests.get(url, headers={"User-Agent":"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.97 Safari/537.36"})
    except requests.exceptions.RequestException as e:
        return "Error: " + str(e)

    # Check if the response contains an HTTP error
    if response.status_code >= 400:
        return "Error: HTTP " + str(response.status_code) + " error"

    soup = BeautifulSoup(response.text, "html.parser")

    for script in soup(["script", "style"]):
        script.extract()

    text = soup.get_text()
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    text = '\n'.join(chunk for chunk in chunks if chunk)

    return text


def extract_hyperlinks(soup):
    """Extract hyperlinks from a BeautifulSoup object"""
    hyperlinks = []
    for link in soup.find_all('a', href=True):
        hyperlinks.append((link.text, link['href']))
    return hyperlinks


def format_hyperlinks(hyperlinks):
    """Format hyperlinks into a list of strings"""
    formatted_links = []
    for link_text, link_url in hyperlinks:
        formatted_links.append(f"{link_text} ({link_url})")
    return formatted_links


def scrape_links(url):
    """Scrape links from a webpage"""
    response = requests.get(url, headers={"User-Agent":"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.97 Safari/537.36"})

    # Check if the response contains an HTTP error
    if response.status_code >= 400:
        return "error"

    soup = BeautifulSoup(response.text, "html.parser")

    for script in soup(["script", "style"]):
        script.extract()

    hyperlinks = extract_hyperlinks(soup)

    return format_hyperlinks(hyperlinks)


def split_text(text, max_length=8192):
    """Split text into chunks of a maximum length"""
    paragraphs = text.split("\n")
    current_length = 0
    current_chunk = []

    for paragraph in paragraphs:
        if current_length + len(paragraph) + 1 <= max_length:
            current_chunk.append(paragraph)
            current_length += len(paragraph) + 1
        else:
            yield "\n".join(current_chunk)
            current_chunk = [paragraph]
            current_length = len(paragraph) + 1

    if current_chunk:
        yield "\n".join(current_chunk)


def create_message(chunk, question):
    """Create a message for the user to summarize a chunk of text"""
    return {
        "role": "user",
        "content": f"\"\"\"{chunk}\"\"\" Using the above text, please answer the following question: \"{question}\" -- if the question cannot be answered using the text, please summarize the text."
    }

def summarize_text(text, question):
    """Summarize text using the LLM model"""
    if not text:
        return "Error: No text to summarize"

    text_length = len(text)
    print(f"Text length: {text_length} characters")

    summaries = []
    chunks = list(split_text(text))

    for i, chunk in enumerate(chunks):
        print(f"Summarizing chunk {i + 1} / {len(chunks)}")
        messages = [create_message(chunk, question)]

        summary = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0, 
            max_tokens=300,
        )
        summaries.append(summary.choices[0].message["content"])

    print(f"Summarized {len(chunks)} chunks.")

    combined_summary = "\n".join(summaries)
    messages = [create_message(combined_summary, question)]

    final_summary = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0, 
            max_tokens=300,
    )

    return final_summary.choices[0].message["content"]
