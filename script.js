// State Management
function switchTab(tabId) {
    // Hide all views
    document.querySelectorAll('.view-section').forEach(el => el.classList.remove('active'));
    document.querySelectorAll('.nav-item').forEach(el => el.classList.remove('active'));
    
    // Show selected
    document.getElementById(tabId).classList.add('active');
    
    // Highlight nav
    const index = ['url-scrape', 'topic-search', 'batch-scrape'].indexOf(tabId);
    document.querySelectorAll('.nav-item')[index].classList.add('active');
    
    // Reset UI
    document.getElementById('results').innerHTML = '';
    document.getElementById('results').classList.add('hidden');
}

async function handleRequest(endpoint, payload) {
    const loading = document.getElementById('loading');
    const resultsArea = document.getElementById('results');
    
    loading.classList.remove('hidden');
    resultsArea.classList.add('hidden');
    resultsArea.innerHTML = '';
    
    try {
        const res = await fetch(endpoint, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(payload)
        });
        
        const data = await res.json();
        renderResults(data);
    } catch (err) {
        resultsArea.innerHTML = `<div class="error-msg">Connection Error: ${err.message}</div>`;
    } finally {
        loading.classList.add('hidden');
        resultsArea.classList.remove('hidden');
    }
}

// Actions
function scrapeUrl() {
    const url = document.getElementById('url-input').value.trim();
    if (!url) return alert('Please enter a URL');
    handleRequest('/api/scrape', { url });
}

function searchAndScrape() {
    const topic = document.getElementById('topic-input').value.trim();
    if (!topic) return alert('Please enter a topic');
    handleRequest('/api/search-scrape', { topic });
}

function batchScrape() {
    const text = document.getElementById('batch-urls').value.trim();
    const urls = text.split('\n').map(u => u.trim()).filter(u => u);
    if (!urls.length) return alert('Please enter URLs');
    handleRequest('/api/batch-scrape', { urls });
}

// Rendering
function renderResults(data) {
    const container = document.getElementById('results');
    
    if (!data.success) {
        container.innerHTML = `<div class="error-msg">Error: ${data.error}</div>`;
        return;
    }

    let html = '';

    // Handle Search Results List
    if (data.results && !data.scraped_data && !data.comparison) {
        html += `<h3>Search Results</h3>`;
        data.results.forEach(r => {
            html += `
            <div class="result-card">
                <a href="${r.url}" target="_blank" class="result-title">${r.title}</a>
                <p class="summary-text">${r.url}</p>
            </div>`;
        });
    }
    
    // Handle Single Scrape or First Search Result
    const item = data.scraped_data || (data.data ? data.data : null);
    const summary = data.summary;
    
    if (item) {
        html += `
        <div class="result-card">
            <a href="${item.url}" target="_blank" class="result-title">${item.title}</a>
            <div class="meta-tags">
                <span>${summary?.stats?.words || 0} words</span>
                <span>~${summary?.stats?.read_time_min || 1} min read</span>
            </div>
            
            ${summary ? `
            <div class="summary-box">
                <p class="summary-text">${summary.summary_text}</p>
            </div>
            <div class="key-points">
                <h4>Key Takeaways</h4>
                <ul>
                    ${summary.key_points.map(p => `<li>${p.substring(0, 150)}...</li>`).join('')}
                </ul>
            </div>
            ` : ''}
        </div>`;
    }

    // Handle Batch
    if (data.comparison) {
        html += `<div class="result-card"><h3>Batch Summary</h3><p>Processed ${data.comparison.total_processed} pages.</p></div>`;
        data.results.forEach(r => {
             if(r.success) {
                 html += `
                 <div class="result-card">
                    <a href="${r.data.url}" class="result-title" target="_blank">${r.data.title}</a>
                    <p class="summary-text">${r.summary.summary_text}</p>
                 </div>`;
             }
        });
    }

    container.innerHTML = html;
}