/* ============================================
   PRODUCT RECOMMENDER - MAIN JAVASCRIPT
   ============================================ */

// Theme Toggle
function toggleTheme() {
    const html = document.documentElement;
    const currentTheme = html.getAttribute('data-theme');
    const newTheme = currentTheme === 'dark' ? 'light' : 'dark';

    html.setAttribute('data-theme', newTheme);
    localStorage.setItem('theme', newTheme);

    // Update toggle button
    const toggleBtn = document.querySelector('.theme-toggle');
    if (toggleBtn) {
        toggleBtn.textContent = newTheme === 'dark' ? '☀️' : '🌙';
    }

    // Update charts if they exist
    if (window.categoryChart && window.ratingChart) {
        updateChartColors(newTheme);
    }
}

function updateChartColors(theme) {
    const color = theme === 'dark' ? '#eaeaea' : '#1a1a2e';
    const gridColor = theme === 'dark' ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.1)';

    [window.categoryChart, window.ratingChart].forEach(chart => {
        if (chart) {
            chart.options.scales.x.ticks.color = color;
            chart.options.scales.y.ticks.color = color;
            chart.options.scales.x.grid.color = gridColor;
            chart.options.scales.y.grid.color = gridColor;
            chart.options.plugins.legend.labels.color = color;
            chart.update();
        }
    });
}

// Load saved theme on page load
document.addEventListener('DOMContentLoaded', function () {
    const savedTheme = localStorage.getItem('theme') || 'light';
    document.documentElement.setAttribute('data-theme', savedTheme);

    const toggleBtn = document.querySelector('.theme-toggle');
    if (toggleBtn) {
        toggleBtn.textContent = savedTheme === 'dark' ? '☀️' : '🌙';
    }

    // Auto-hide flash messages after 5 seconds
    const flashMessages = document.querySelectorAll('.flash');
    flashMessages.forEach(flash => {
        setTimeout(() => {
            flash.style.opacity = '0';
            flash.style.transform = 'translateX(100%)';
            setTimeout(() => flash.remove(), 300);
        }, 5000);
    });

    // Initialize charts if on dashboard
    if (document.getElementById('categoryChart')) {
        initCharts();
    }

    // Lazy Load Images
    const images = document.querySelectorAll('img[data-src]');
    const imageObserver = new IntersectionObserver((entries, observer) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const img = entry.target;
                img.src = img.dataset.src;
                img.removeAttribute('data-src');
                observer.unobserve(img);
            }
        });
    });
    images.forEach(img => imageObserver.observe(img));
});

// Initialize Charts
function initCharts() {
    fetch('/api/stats')
        .then(response => response.json())
        .then(data => {
            const theme = document.documentElement.getAttribute('data-theme');
            const textColor = theme === 'dark' ? '#eaeaea' : '#1a1a2e';
            const gridColor = theme === 'dark' ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.1)';

            // Category Chart
            const ctxCat = document.getElementById('categoryChart').getContext('2d');
            window.categoryChart = new Chart(ctxCat, {
                type: 'bar',
                data: {
                    labels: data.categories.map(c => c.main_category),
                    datasets: [{
                        label: 'Products per Category',
                        data: data.categories.map(c => c.count),
                        backgroundColor: '#6366f1',
                        borderRadius: 8
                    }]
                },
                options: {
                    responsive: true,
                    plugins: {
                        legend: { labels: { color: textColor } }
                    },
                    scales: {
                        y: {
                            beginAtZero: true,
                            grid: { color: gridColor },
                            ticks: { color: textColor }
                        },
                        x: {
                            grid: { display: false },
                            ticks: { color: textColor }
                        }
                    }
                }
            });

            // Rating Chart
            const ctxRate = document.getElementById('ratingChart').getContext('2d');
            window.ratingChart = new Chart(ctxRate, {
                type: 'line',
                data: {
                    labels: data.ratings.map(r => r.rating_bin + ' Stars'),
                    datasets: [{
                        label: 'Distribution',
                        data: data.ratings.map(r => r.count),
                        borderColor: '#10b981',
                        backgroundColor: 'rgba(16, 185, 129, 0.1)',
                        fill: true,
                        tension: 0.4
                    }]
                },
                options: {
                    responsive: true,
                    plugins: {
                        legend: { labels: { color: textColor } }
                    },
                    scales: {
                        y: {
                            beginAtZero: true,
                            grid: { color: gridColor },
                            ticks: { color: textColor }
                        },
                        x: {
                            grid: { display: false },
                            ticks: { color: textColor }
                        }
                    }
                }
            });
        });
}

// Live Search (AJAX)
function liveSearch(query) {
    if (query.length < 2) return;

    fetch('/api/search', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ query: query })
    })
        .then(response => response.json())
        .then(data => {
            console.log('Search results:', data);
        })
        .catch(error => console.error('Error:', error));
}

// Product Card Hover Effect
document.querySelectorAll('.product-card').forEach(card => {
    card.addEventListener('mouseenter', function () {
        this.style.transition = 'all 0.3s ease';
    });
});

// Form Validation
function validateForm(form) {
    const inputs = form.querySelectorAll('input[required]');
    let isValid = true;

    inputs.forEach(input => {
        if (!input.value.trim()) {
            input.style.borderColor = '#ef4444';
            isValid = false;
        } else {
            input.style.borderColor = '';
        }
    });

    return isValid;
}

// Add loading state to buttons
document.querySelectorAll('form').forEach(form => {
    form.addEventListener('submit', function (e) {
        const button = this.querySelector('button[type="submit"]');
        if (button) {
            button.innerHTML = '⏳ Loading...';
            button.disabled = true;
        }
    });
});

console.log('🚀 ProductRec JS loaded successfully!');
