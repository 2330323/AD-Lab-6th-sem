from flask import Flask, render_template, request, jsonify
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import time
from datetime import datetime

app = Flask(__name__)

PALETTE = {
    'darkest': '#061E29',
    'dark_blue': '#1D546D',
    'teal': '#5F9598',
    'light': '#F3F4F4'
}

class WebScraper:
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
    
    def scrape_url(self, url):
        try:
            response = requests.get(url, headers=self.headers, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'lxml')
            
            # Clean up DOM
            for tag in soup(["script", "style", "nav", "footer", "iframe", "noscript"]):
                tag.decompose()
            
            # Extract
            data = {
                'url': url,
                'title': soup.title.string.strip() if soup.title else 'No Title Found',
                'headings': self._extract_headings(soup),
                'paragraphs': self._extract_paragraphs(soup),
                'links': self._extract_links(soup, url),
                'images': self._extract_images(soup, url),
                'meta_description': self._get_meta_description(soup),
                'text_content': soup.get_text(separator=' ', strip=True),
                'scraped_at': datetime.now().isoformat()
            }
            
            return {'success': True, 'data': data}
        
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _extract_headings(self, soup):
        headings = {}
        for i in range(1, 4): 
            tags = soup.find_all(f'h{i}')
            if tags:
                headings[f'h{i}'] = [t.get_text(strip=True) for t in tags if t.get_text(strip=True)]
        return headings
    
    def _extract_paragraphs(self, soup):
        return [p.get_text(strip=True) for p in soup.find_all('p') if len(p.get_text(strip=True)) > 40]
    
    def _extract_links(self, soup, base_url):
        links = []
        for link in soup.find_all('a', href=True):
            href = link['href']
            if href.startswith('http') or href.startswith('/'):
                full_url = urljoin(base_url, href)
                links.append({'text': link.get_text(strip=True)[:50], 'url': full_url})
        return links[:20] 
    
    def _extract_images(self, soup, base_url):
        images = []
        for img in soup.find_all('img', src=True):
            src = urljoin(base_url, img['src'])
            images.append({'src': src, 'alt': img.get('alt', '')})
        return images[:10]
    
    def _get_meta_description(self, soup):
        meta = soup.find('meta', attrs={'name': 'description'})
        return meta['content'] if meta and meta.get('content') else 'No description available.'

    def search_and_scrape(self, topic):
        
        simulated_results = [
            {'title': f'Wikipedia: {topic}', 'url': f'https://en.wikipedia.org/wiki/{topic.replace(" ", "_")}'},
            {'title': f'News about {topic}', 'url': f'https://www.google.com/search?q={topic}'},
            {'title': f'Reddit Discussion: {topic}', 'url': f'https://www.reddit.com/search/?q={topic}'},
            {'title': f'Medium Articles: {topic}', 'url': f'https://medium.com/search?q={topic}'},
            {'title': f'Quora Questions: {topic}', 'url': f'https://www.quora.com/search?q={topic}'},
            {'title': f'Twitter/X Trends: {topic}', 'url': f'https://twitter.com/search?q={topic}'},
            {'title': f'YouTube Videos: {topic}', 'url': f'https://www.youtube.com/results?search_query={topic}'},
            {'title': f'StackOverflow: {topic}', 'url': f'https://stackoverflow.com/search?q={topic}'},
            {'title': f'Google Scholar: {topic}', 'url': f'https://scholar.google.com/scholar?q={topic}'},
            {'title': f'Bing Search: {topic}', 'url': f'https://www.bing.com/search?q={topic}'}
        ]
        return {'success': True, 'results': simulated_results}

class LLMProcessor:
    def summarize_content(self, data):

        text = data.get('text_content', '')
        paras = data.get('paragraphs', [])
        
        # Calculate approximate reading time
        word_count = len(text.split())
        read_time = max(1, round(word_count / 200))
        
        # Create a "summary" from the first few substantial paragraphs
        summary_text = " ".join(paras[:2]) if paras else "No content substantial enough to summarize."
        
        return {
            'title': data.get('title'),
            'summary_text': summary_text[:500] + "...",
            'key_points': paras[:3],
            'stats': {
                'words': word_count,
                'read_time_min': read_time
            }
        }
    
    def compare_content(self, results_list):
        return {
            'total_processed': len(results_list),
            'avg_length': sum(len(r.get('text_content', '')) for r in results_list) // max(1, len(results_list))
        }

scraper = WebScraper()
processor = LLMProcessor()

# --- Routes ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/scrape', methods=['POST'])
def scrape():
    url = request.json.get('url')
    if not url: return jsonify({'success': False, 'error': 'URL required'})
    
    result = scraper.scrape_url(url)
    if result['success']:
        result['summary'] = processor.summarize_content(result['data'])
    return jsonify(result)

@app.route('/api/search-scrape', methods=['POST'])
def search_scrape():
    topic = request.json.get('topic')
    if not topic: return jsonify({'success': False, 'error': 'Topic required'})
    
    # 1. Get Search Results
    search_res = scraper.search_and_scrape(topic)
    
    # 2. Auto-scrape the first valid Wiki result for demonstration
    if search_res['results']:
        target_url = search_res['results'][0]['url']
        scrape_res = scraper.scrape_url(target_url)
        if scrape_res['success']:
            search_res['scraped_data'] = scrape_res['data']
            search_res['summary'] = processor.summarize_content(scrape_res['data'])
            
    return jsonify(search_res)

@app.route('/api/batch-scrape', methods=['POST'])
def batch_scrape():
    urls = request.json.get('urls', [])
    results = []
    
    for url in urls[:5]: # Limit to 5
        res = scraper.scrape_url(url)
        if res['success']:
            res['summary'] = processor.summarize_content(res['data'])
        results.append(res)
        time.sleep(0.5)
        
    return jsonify({
        'success': True, 
        'results': results,
        'comparison': processor.compare_content([r['data'] for r in results if r['success']])
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)