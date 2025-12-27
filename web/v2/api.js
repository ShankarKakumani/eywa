// ==================== API Client ====================
const API = '/api';

const api = {
    // System info
    async getInfo() {
        const res = await fetch(`${API}/info`);
        return res.json();
    },

    // Sources
    async getSources() {
        const res = await fetch(`${API}/sources`);
        return res.json();
    },

    async getSourceDocs(sourceId, limit = null) {
        const url = limit ? `${API}/sources/${encodeURIComponent(sourceId)}/docs?limit=${limit}`
                         : `${API}/sources/${encodeURIComponent(sourceId)}/docs`;
        const res = await fetch(url);
        return res.json();
    },

    async deleteSource(sourceId) {
        const res = await fetch(`${API}/sources/${encodeURIComponent(sourceId)}`, { method: 'DELETE' });
        return res.json();
    },

    async exportSource(sourceId) {
        window.location.href = `${API}/sources/${encodeURIComponent(sourceId)}/export`;
    },

    // SQLite-backed endpoints (for web UI - accurate document counts)
    async getSqlSources() {
        const res = await fetch(`${API}/sql/sources`);
        return res.json();
    },

    async getSqlSourceDocs(sourceId, limit = null, offset = null) {
        const params = new URLSearchParams();
        if (limit) params.set('limit', limit);
        if (offset) params.set('offset', offset);
        const query = params.toString();
        const url = `${API}/sql/sources/${encodeURIComponent(sourceId)}/docs${query ? '?' + query : ''}`;
        const res = await fetch(url);
        return res.json();
    },

    // Documents
    async getDocument(docId) {
        const res = await fetch(`${API}/docs/${encodeURIComponent(docId)}`);
        return res.json();
    },

    async deleteDocument(docId) {
        const res = await fetch(`${API}/docs/${encodeURIComponent(docId)}`, { method: 'DELETE' });
        return res.json();
    },

    // Ingestion
    async ingest(sourceId, documents) {
        const res = await fetch(`${API}/ingest`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ source_id: sourceId, documents })
        });
        return res.json();
    },

    async queue(sourceId, documents) {
        const res = await fetch(`${API}/queue`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ source_id: sourceId, documents })
        });
        return res.json();
    },

    async getJob(jobId) {
        const res = await fetch(`${API}/jobs/${jobId}`);
        return res.json();
    },

    async getJobs() {
        const res = await fetch(`${API}/jobs`);
        return res.json();
    },

    async getJobDocs(jobId) {
        const res = await fetch(`${API}/jobs/${jobId}/docs`);
        return res.json();
    },

    async fetchUrl(url, sourceId) {
        const res = await fetch(`${API}/fetch-url`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ url, source_id: sourceId })
        });
        return res.json();
    },

    async fetchPreview(url) {
        const res = await fetch(`${API}/fetch-preview`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ url })
        });
        return res.json();
    },

    // Search
    async search(query, limit = 10, sourceId = null) {
        const body = { query, limit };
        if (sourceId) body.source_id = sourceId;

        const res = await fetch(`${API}/search`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body)
        });
        return res.json();
    },

    // Export & Reset
    async exportAll() {
        window.location.href = `${API}/export`;
    },

    async reset() {
        const res = await fetch(`${API}/reset`, { method: 'DELETE' });
        return res.json();
    }
};

// ==================== Utility Functions ====================
function escapeHtml(text) {
    if (!text) return '';
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function formatBytes(bytes) {
    if (bytes === 0) return '0 B';
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    if (bytes < 1024 * 1024 * 1024) return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
    return (bytes / (1024 * 1024 * 1024)).toFixed(2) + ' GB';
}

function formatNumber(num) {
    if (num >= 1000000) return (num / 1000000).toFixed(1) + 'M';
    if (num >= 1000) return (num / 1000).toFixed(1) + 'K';
    return num.toString();
}
