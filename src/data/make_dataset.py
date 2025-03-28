import requests
import xml.etree.ElementTree as ET
from pathlib import Path
from ..utils.config import load_params

def fetch_arxiv_papers():
    params = load_params()
    domains = params["arxiv"]["domains"]
    max_results = params["arxiv"]["max_results"]
    save_dir = Path(params["arxiv"]["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    all_papers = []
    for domain in domains:
        url = f"http://export.arxiv.org/api/query?search_query=cat:{domain}&start=0&max_results={max_results}"
        response = requests.get(url)
        if response.status_code != 200:
            print(f"Error fetching data for domain: {domain}")
            continue

        root = ET.fromstring(response.text)
        for entry in root.findall("{http://www.w3.org/2005/Atom}entry"):
            title = entry.find("{http://www.w3.org/2005/Atom}title").text
            summary = entry.find("{http://www.w3.org/2005/Atom}summary").text
            pdf_link = entry.find("{http://www.w3.org/2005/Atom}id").text.replace("abs", "pdf") + ".pdf"
            all_papers.append({
                "Title": title.strip(),
                "Abstract": summary.strip(),
                "PDF_Link": pdf_link,
                "Domain": domain
            })
    return all_papers

def download_pdf(pdf_url, save_path):
    response = requests.get(pdf_url)
    if response.status_code == 200:
        with open(save_path, "wb") as f:
            f.write(response.content)
        return save_path
    return None


if __name__ == "__main__":
    papers = fetch_arxiv_papers()
    params = load_params()
    download_pdf(papers, params["arxiv"]["save_dir"])