// ===== Tab Management =====
function switchTab(tabId) {
    document.querySelectorAll('.content-section').forEach(el => {
        el.classList.remove('active');
    });
    document.querySelectorAll('.tab-btn').forEach(el => {
        el.classList.remove('active');
    });

    document.getElementById(tabId).classList.add('active');
    document.querySelector(`[data-tab="${tabId}"]`).classList.add('active');
}

// Tab button listeners
document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', (e) => {
        const tabId = e.currentTarget.dataset.tab;
        switchTab(tabId);
    });
});

// ===== Load Available Models =====
async function loadAvailableModels() {
    try {
        const response = await fetch('http://127.0.0.1:5000/available_models');
        const data = await response.json();
        
        if (data.models) {
            updateModelSelector(data.models);
        }
    } catch (error) {
        console.log('Using default model options');
    }
}

function updateModelSelector(models) {
    const select = document.getElementById('modelSelect');
    const customGroup = document.getElementById('customModels');
    
    // Enable/disable custom models based on availability
    models.forEach(model => {
        if (model.type === 'custom') {
            const option = select.querySelector(`option[value="${model.value}"]`);
            if (option) {
                option.disabled = false;
                option.textContent = model.label;
            }
        }
    });
}

// ===== EXPERIMENT 1: Classification =====
const imageInput = document.getElementById('imageInput');
const uploadArea = document.getElementById('uploadArea');
const preview = document.getElementById('preview');
const classifyBtn = document.getElementById('classifyBtn');
const resultContainer = document.getElementById('resultContainer');
const modelSelect = document.getElementById('modelSelect');

// Upload area click
uploadArea.addEventListener('click', () => {
    imageInput.click();
});

// Drag and drop handlers
uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('dragover');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('dragover');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleImageUpload(files[0]);
    }
});

// File input change
imageInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) {
        handleImageUpload(file);
    }
});

function handleImageUpload(file) {
    if (!file.type.startsWith('image/')) {
        alert('Please select an image file');
        return;
    }

    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
        preview.src = e.target.result;
        preview.hidden = false;
        document.querySelector('.upload-placeholder').style.display = 'none';
        classifyBtn.disabled = false;
        resultContainer.classList.add('hidden');
    };
    reader.readAsDataURL(file);
}

// Classify button
classifyBtn.addEventListener('click', uploadImage);

async function uploadImage() {
    const file = imageInput.files[0];
    if (!file) {
        alert("Please select an image first!");
        return;
    }

    const selectedModel = modelSelect.value;
    
    // Show loading state
    const btnText = classifyBtn.querySelector('.btn-text');
    btnText.textContent = 'Analyzing...';
    classifyBtn.disabled = true;
    
    // Hide previous results
    resultContainer.classList.add('hidden');

    const formData = new FormData();
    formData.append('file', file);
    formData.append('model', selectedModel);

    try {
        const response = await fetch('http://127.0.0.1:5000/classify', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (data.error) {
            alert(data.error);
            return;
        }
        
        displayClassificationResult(data);
        
    } catch (error) {
        console.error('Error:', error);
        alert('Error connecting to backend. Please ensure the server is running.');
    } finally {
        btnText.textContent = 'Classify Image';
        classifyBtn.disabled = false;
    }
}

function displayClassificationResult(data) {
    // Update result elements
    document.getElementById('predictionText').textContent = data.label;
    document.getElementById('confidenceText').textContent = data.confidence;
    document.getElementById('modelUsed').textContent = data.model || 'Unknown';
    
    // Update model description
    const modelDesc = document.getElementById('modelDesc');
    if (data.description) {
        modelDesc.textContent = data.description;
        modelDesc.style.display = 'block';
    } else {
        modelDesc.style.display = 'none';
    }
    
    // Update confidence bar
    const confidenceValue = parseFloat(data.confidence);
    const confidenceFill = document.getElementById('confidenceFill');
    confidenceFill.style.width = confidenceValue + '%';
    
    // Set appropriate icon based on prediction
    const resultIcon = document.getElementById('resultIcon');
    const label = data.label.toLowerCase();
    
    if (label.includes('cat')) {
        resultIcon.textContent = '🐱';
    } else if (label.includes('dog')) {
        resultIcon.textContent = '🐶';
    } else if (label.includes('bird')) {
        resultIcon.textContent = '🐦';
    } else if (label.includes('car') || label.includes('vehicle')) {
        resultIcon.textContent = '🚗';
    } else if (label.includes('person') || label.includes('human')) {
        resultIcon.textContent = '👤';
    } else if (label.includes('food') || label.includes('pizza') || label.includes('banana')) {
        resultIcon.textContent = '🍕';
    } else if (label.includes('flower') || label.includes('plant')) {
        resultIcon.textContent = '🌸';
    } else {
        resultIcon.textContent = '🎯';
    }
    
    // Display top 3 predictions if available (ImageNet models)
    const top3Container = document.getElementById('top3Container');
    const top3List = document.getElementById('top3List');
    
    if (data.top_3 && data.top_3.length > 0) {
        top3List.innerHTML = '';
        
        data.top_3.forEach((pred, index) => {
            const item = document.createElement('div');
            item.className = 'top3-item';
            item.innerHTML = `
                <span class="top3-label">${index + 1}. ${pred.label}</span>
                <span class="top3-confidence">${pred.confidence.toFixed(2)}%</span>
            `;
            top3List.appendChild(item);
        });
        
        top3Container.classList.remove('hidden');
    } else {
        top3Container.classList.add('hidden');
    }
    
    // Show results
    resultContainer.classList.remove('hidden');
    resultContainer.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// ===== EXPERIMENT 2: Stock Prediction =====
let stockChart = null;
const tickerInput = document.getElementById('tickerInput');
const predictBtn = document.getElementById('predictBtn');

// Enter key support
tickerInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
        predictStock();
    }
});

predictBtn.addEventListener('click', predictStock);

async function predictStock() {
    const ticker = tickerInput.value.trim().toUpperCase();
    
    if (!ticker) {
        alert('Please enter a stock ticker symbol');
        tickerInput.focus();
        return;
    }

    // Show loading state
    const btnText = predictBtn.querySelector('.btn-text');
    const btnLoader = predictBtn.querySelector('.btn-loader');
    
    btnText.classList.add('hidden');
    btnLoader.classList.remove('hidden');
    predictBtn.disabled = true;

    try {
        const response = await fetch('http://127.0.0.1:5000/predict_stock', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ ticker: ticker })
        });
        
        const data = await response.json();
        
        if (data.error) {
            alert(data.error);
            return;
        }

        displayStockResults(data);
        
    } catch (error) {
        console.error('Error:', error);
        alert('Failed to fetch prediction. Please check your connection and try again.');
    } finally {
        btnText.classList.remove('hidden');
        btnLoader.classList.add('hidden');
        predictBtn.disabled = false;
    }
}

function displayStockResults(data) {
    // Update summary cards
    const currentPrice = data.current_price;
    const nextDayPrice = data.next_day_prediction;
    const change = ((nextDayPrice - currentPrice) / currentPrice) * 100;
    
    document.getElementById('currentPrice').textContent = `$${currentPrice.toFixed(2)}`;
    document.getElementById('nextDayPrice').textContent = `$${nextDayPrice.toFixed(2)}`;
    
    const priceChange = document.getElementById('priceChange');
    priceChange.textContent = `${change >= 0 ? '+' : ''}${change.toFixed(2)}%`;
    priceChange.className = 'summary-change ' + (change >= 0 ? 'positive' : 'negative');
    
    // Show summary
    document.getElementById('predictionSummary').classList.remove('hidden');
    
    // Update metrics
    if (data.metrics) {
        document.getElementById('lrMSE').textContent = data.metrics.linear_regression.mse.toFixed(2);
        document.getElementById('lrR2').textContent = data.metrics.linear_regression.r2.toFixed(4);
        document.getElementById('lstmMSE').textContent = data.metrics.lstm.mse.toFixed(2);
        document.getElementById('metricsContainer').classList.remove('hidden');
    }
    
    // Render chart
    renderChart(data);
}

function renderChart(data) {
    const ctx = document.getElementById('stockChart').getContext('2d');
    
    // Destroy existing chart
    if (stockChart) {
        stockChart.destroy();
    }

    stockChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: data.dates,
            datasets: [
                {
                    label: 'Actual Price',
                    data: data.actual,
                    borderColor: '#1f2937',
                    backgroundColor: 'rgba(31, 41, 55, 0.1)',
                    borderWidth: 2,
                    pointRadius: 0,
                    tension: 0.1,
                    fill: false
                },
                {
                    label: 'Linear Regression',
                    data: data.linear_preds,
                    borderColor: '#ef4444',
                    borderDash: [5, 5],
                    borderWidth: 2,
                    pointRadius: 0,
                    fill: false
                },
                {
                    label: 'LSTM Prediction',
                    data: data.lstm_preds,
                    borderColor: '#10b981',
                    borderWidth: 2.5,
                    pointRadius: 0,
                    fill: false
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            interaction: {
                mode: 'index',
                intersect: false,
            },
            plugins: {
                title: {
                    display: true,
                    text: `${data.ticker} - Linear Regression vs LSTM`,
                    font: {
                        size: 16,
                        weight: '600'
                    },
                    color: '#1f2937'
                },
                legend: {
                    display: true,
                    position: 'top',
                    labels: {
                        usePointStyle: true,
                        padding: 15,
                        font: {
                            size: 12
                        }
                    }
                },
                tooltip: {
                    backgroundColor: 'rgba(0, 0, 0, 0.8)',
                    padding: 12,
                    titleFont: {
                        size: 13
                    },
                    bodyFont: {
                        size: 12
                    },
                    callbacks: {
                        label: function(context) {
                            let label = context.dataset.label || '';
                            if (label) {
                                label += ': ';
                            }
                            if (context.parsed.y !== null) {
                                label += '$' + context.parsed.y.toFixed(2);
                            }
                            return label;
                        }
                    }
                }
            },
            scales: {
                x: {
                    display: true,
                    grid: {
                        display: false
                    },
                    ticks: {
                        maxTicksLimit: 10,
                        font: {
                            size: 11
                        }
                    }
                },
                y: {
                    display: true,
                    grid: {
                        color: 'rgba(0, 0, 0, 0.05)'
                    },
                    ticks: {
                        callback: function(value) {
                            return '$' + value.toFixed(0);
                        },
                        font: {
                            size: 11
                        }
                    }
                }
            }
        }
    });
    
    // Scroll to chart
    document.querySelector('.chart-container').scrollIntoView({ 
        behavior: 'smooth', 
        block: 'nearest' 
    });
}

// ===== Initialize =====
console.log('🚀 ML Experiments loaded successfully');

// Load available models on page load
loadAvailableModels();