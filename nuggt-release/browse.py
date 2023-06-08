import requests
from bs4 import BeautifulSoup

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

