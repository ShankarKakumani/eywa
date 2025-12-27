// ==================== Dashboard ====================
const Dashboard = {
    async init() {
        this.render();
        await this.loadData();
    },

    render() {
        const panel = document.getElementById('dashboard');
        panel.innerHTML = `
            <div class="welcome">
                <h1>Welcome back!</h1>
                <p>Your knowledge base is ready.</p>
            </div>

            <div class="stats-grid">
                <div class="stat-card" onclick="App.switchTab('explorer')" style="cursor: pointer">
                    <div class="stat-icon">📚</div>
                    <div class="stat-value" id="statSources">-</div>
                    <div class="stat-label">Sources</div>
                </div>
                <div class="stat-card" onclick="App.switchTab('explorer')" style="cursor: pointer">
                    <div class="stat-icon">📄</div>
                    <div class="stat-value" id="statDocuments">-</div>
                    <div class="stat-label">Documents</div>
                </div>
                <div class="stat-card" onclick="App.switchTab('explorer')" style="cursor: pointer">
                    <div class="stat-icon">🧩</div>
                    <div class="stat-value" id="statChunks">-</div>
                    <div class="stat-label">Chunks</div>
                </div>
            </div>

            <h2 class="section-title">Models</h2>
            <div class="model-cards" id="modelsInfo">
                <div class="model-card">
                    <div class="loading">Loading</div>
                </div>
            </div>

            <div class="info-section">
                <h2>Storage</h2>
                <div id="storageInfo">
                    <div class="loading">Loading storage info</div>
                </div>
            </div>
        `;
    },

    async loadData() {
        try {
            // Load info from /api/info
            const info = await api.getInfo();
            this.updateStats(info.stats);
            this.updateModels(info);
            this.updateStorage(info);
        } catch (e) {
            // Fallback: compute stats from sources
            console.warn('Failed to load /api/info, falling back to sources:', e);
            await this.loadFromSources();
        }
    },

    async loadFromSources() {
        try {
            const data = await api.getSources();
            const stats = {
                source_count: data.sources.length,
                document_count: data.sources.reduce((sum, s) => sum + (s.doc_count || 0), 0),
                chunk_count: data.sources.reduce((sum, s) => sum + (s.chunk_count || 0), 0)
            };
            this.updateStats(stats);

            // Show placeholder for models
            document.getElementById('modelsInfo').innerHTML = `
                <div class="model-card">
                    <div class="model-header">
                        <span class="model-type">Embedding</span>
                        <span class="model-badge">Default</span>
                    </div>
                    <div class="model-name">bge-base-en-v1.5</div>
                    <div class="model-meta">768 dimensions</div>
                </div>
            `;
            document.getElementById('storageInfo').innerHTML = `
                <div class="info-row">
                    <span class="info-label">Data Directory</span>
                    <span class="info-value">~/.eywa/data/</span>
                </div>
            `;
        } catch (e) {
            console.error('Failed to load sources:', e);
        }
    },

    updateStats(stats) {
        if (!stats) return;
        document.getElementById('statSources').textContent = formatNumber(stats.source_count || 0);
        document.getElementById('statDocuments').textContent = formatNumber(stats.document_count || 0);
        document.getElementById('statChunks').textContent = formatNumber(stats.chunk_count || 0);
    },

    updateModels(info) {
        const container = document.getElementById('modelsInfo');

        let html = '';

        // Embedding model card
        if (info.embedding_model) {
            const em = info.embedding_model;
            html += `
                <div class="model-card">
                    <div class="model-type">Embedding</div>
                    <div class="model-name-row">
                        <span class="model-name">${escapeHtml(em.name)}</span>
                        <span class="model-badge">Selected</span>
                    </div>
                    <div class="model-meta">
                        ${em.dimensions ? em.dimensions + ' dimensions' : ''}
                        ${em.size_mb ? ' · ' + em.size_mb + ' MB' : ''}
                    </div>
                </div>
            `;
        } else {
            html += `
                <div class="model-card">
                    <div class="model-type">Embedding</div>
                    <div class="model-name-row">
                        <span class="model-name">bge-base-en-v1.5</span>
                        <span class="model-badge">Default</span>
                    </div>
                    <div class="model-meta">768 dimensions · 418 MB</div>
                </div>
            `;
        }

        // Reranker model card
        if (info.reranker_model) {
            const rm = info.reranker_model;
            html += `
                <div class="model-card">
                    <div class="model-type">Reranker</div>
                    <div class="model-name-row">
                        <span class="model-name">${escapeHtml(rm.name)}</span>
                        <span class="model-badge">Selected</span>
                    </div>
                    <div class="model-meta">
                        Cross-encoder${rm.size_mb ? ' · ' + rm.size_mb + ' MB' : ''}
                    </div>
                </div>
            `;
        } else {
            html += `
                <div class="model-card">
                    <div class="model-type">Reranker</div>
                    <div class="model-name-row">
                        <span class="model-name">ms-marco-MiniLM-L-6-v2</span>
                        <span class="model-badge">Default</span>
                    </div>
                    <div class="model-meta">Cross-encoder · 86 MB</div>
                </div>
            `;
        }

        container.innerHTML = html;
    },

    updateStorage(info) {
        const container = document.getElementById('storageInfo');
        const storage = info.storage || {};

        let rows = [];
        let dataTotal = 0;
        let modelTotal = 0;

        // Data storage section
        let html = `<div class="storage-section-label">Data</div>`;
        html += `<div class="info-rows">`;

        if (storage.content_db_bytes !== undefined) {
            html += `
                <div class="info-row">
                    <span class="info-label">Content DB (SQLite)</span>
                    <span class="info-value">${formatBytes(storage.content_db_bytes)}</span>
                </div>
            `;
            dataTotal += storage.content_db_bytes;
        }

        if (storage.vector_db_bytes !== undefined) {
            html += `
                <div class="info-row">
                    <span class="info-label">Vector DB (LanceDB)</span>
                    <span class="info-value">${formatBytes(storage.vector_db_bytes)}</span>
                </div>
            `;
            dataTotal += storage.vector_db_bytes;
        }

        if (storage.bm25_index_bytes !== undefined) {
            html += `
                <div class="info-row">
                    <span class="info-label">BM25 Index (Tantivy)</span>
                    <span class="info-value">${formatBytes(storage.bm25_index_bytes)}</span>
                </div>
            `;
            dataTotal += storage.bm25_index_bytes;
        }

        html += `</div>`; // Close data rows

        // Models storage section - use cached_models from API
        html += `<div class="storage-section-label" style="margin-top: 16px;">Models (cached from HuggingFace)</div>`;
        html += `<div class="info-rows">`;

        const cachedModels = info.cached_models || [];
        if (cachedModels.length > 0) {
            for (const model of cachedModels) {
                html += `
                    <div class="info-row">
                        <span class="info-label">${escapeHtml(model.name)}</span>
                        <span class="info-value">${formatBytes(model.size_bytes)}</span>
                    </div>
                `;
                modelTotal += model.size_bytes;
            }
        } else {
            // Fallback to selected models if cached_models not available
            if (info.embedding_model) {
                const em = info.embedding_model;
                const emBytes = (em.size_mb || 0) * 1024 * 1024;
                html += `
                    <div class="info-row">
                        <span class="info-label">${escapeHtml(em.name)}</span>
                        <span class="info-value">${em.size_mb ? em.size_mb + ' MB' : '-'}</span>
                    </div>
                `;
                modelTotal += emBytes;
            }

            if (info.reranker_model) {
                const rm = info.reranker_model;
                const rmBytes = (rm.size_mb || 0) * 1024 * 1024;
                html += `
                    <div class="info-row">
                        <span class="info-label">${escapeHtml(rm.name)}</span>
                        <span class="info-value">${rm.size_mb ? rm.size_mb + ' MB' : '-'}</span>
                    </div>
                `;
                modelTotal += rmBytes;
            }
        }

        html += `</div>`; // Close model rows

        // Total
        const grandTotal = dataTotal + modelTotal;
        if (grandTotal > 0) {
            html += `
                <div class="info-total">
                    <span class="info-label">Total</span>
                    <span class="info-value">${formatBytes(grandTotal)}</span>
                </div>
            `;
        }

        container.innerHTML = html || `
            <div class="info-row">
                <span class="info-label">Data Directory</span>
                <span class="info-value">~/.eywa/data/</span>
            </div>
        `;
    }
};
