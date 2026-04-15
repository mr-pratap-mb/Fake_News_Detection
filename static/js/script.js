document.addEventListener('DOMContentLoaded', () => {
    const textTab = document.getElementById('tab-text');
    const urlTab = document.getElementById('tab-url');
    const textInputGroup = document.getElementById('input-group-text');
    const urlInputGroup = document.getElementById('input-group-url');
    const analyzeBtn = document.getElementById('analyze-btn');
    const loadingOverlay = document.getElementById('loading-overlay');
    const inputSection = document.querySelector('.input-section');
    const resultsContainer = document.getElementById('results-container');
    const form = document.getElementById('analyze-form');

    let currentInputMode = 'text';

    // Tab Switching
    if (textTab && urlTab) {
        textTab.addEventListener('click', () => {
            currentInputMode = 'text';
            textTab.classList.add('active');
            urlTab.classList.remove('active');
            textInputGroup.classList.remove('hidden');
            urlInputGroup.classList.add('hidden');
        });

        urlTab.addEventListener('click', () => {
            currentInputMode = 'url';
            urlTab.classList.add('active');
            textTab.classList.remove('active');
            urlInputGroup.classList.remove('hidden');
            textInputGroup.classList.add('hidden');
        });
    }

    if (form) {
        form.addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const textValue = document.getElementById('claim-text').value;
            const urlValue = document.getElementById('claim-url').value;
            
            const payloadValue = currentInputMode === 'text' ? textValue : urlValue;

            if (!payloadValue.trim()) return;

            // UI updates
            inputSection.style.display = 'none';
            loadingOverlay.style.display = 'flex';
            resultsContainer.style.display = 'none';

            try {
                const response = await fetch('/analyze', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ 
                        text: payloadValue, 
                        is_url: currentInputMode === 'url' 
                    })
                });

                const data = await response.json();

                if (!response.ok) {
                    throw new Error(data.error || 'Something went wrong');
                }

                renderResults(data);

            } catch (error) {
                alert(`Error: ${error.message}`);
                // Revert UI on error
                inputSection.style.display = 'block';
                loadingOverlay.style.display = 'none';
            }
        });
    }

    function renderResults(data) {
        loadingOverlay.style.display = 'none';
        resultsContainer.style.display = 'block';
        
        let containerHTML = `
            <button onclick="window.location.reload()" class="btn-submit" style="margin-bottom: 2rem; width: auto; padding: 0.8rem 2rem;">&larr; Analyze Another</button>
            <div class="score-card">
                <h2 style="margin-bottom: 1.5rem;">Credibility Assessment</h2>
                <div class="probability-circle" style="--percentage: ${data.fake_probability}">
                    <span class="prob-value ${data.fake_probability > 60 ? 'text-danger' : (data.fake_probability < 40 ? 'text-success' : 'text-warning')}">
                        ${data.fake_probability}%
                    </span>
                </div>
                <h3 style="margin-bottom: 0.5rem">Fake Probability</h3>
                <p class="explanation">${data.explanation}</p>
                
                <div style="display: flex; justify-content: center; gap: 2rem; margin-top: 2rem; color: var(--text-muted); font-size: 0.9rem;">
                    <div>ML Score: <strong style="color: white">${data.ml_score}%</strong></div>
                    <div>Text Match: <strong style="color: white">${data.similarity_score}%</strong></div>
                    <div>Source Credibility: <strong style="color: white">${data.credibility_score}%</strong></div>
                </div>
            </div>
            
            <h3 style="margin-bottom: 1.5rem">Live Web Evidence Found (${data.evidences.length})</h3>
            <div class="evidence-grid">
        `;

        if (data.evidences && data.evidences.length > 0) {
            data.evidences.forEach(item => {
                const badgeClass = item.source_type === 'newsapi' ? 'source-api' : 'source-rss';
                containerHTML += `
                    <div class="evidence-card">
                        <span class="source-badge ${badgeClass}">${item.domain}</span>
                        <h4 class="evidence-title">${item.title}</h4>
                        <p class="evidence-desc">${item.description.substring(0, 120)}...</p>
                        <a href="${item.url}" target="_blank" class="evidence-link">Read Full Source &rarr;</a>
                    </div>
                `;
            });
        } else {
            containerHTML += `<p style="color: var(--text-muted);">No supporting live evidence was found for this query.</p>`;
        }

        containerHTML += `</div>`;
        resultsContainer.innerHTML = containerHTML;
    }
});
