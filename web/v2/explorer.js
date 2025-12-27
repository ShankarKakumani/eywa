// ==================== File Explorer ====================
const Explorer = {
    sources: [],
    expandedSources: new Set(),

    async init() {
        this.render();
        await this.loadSources();
    },

    render() {
        const panel = document.getElementById('explorer');
        panel.innerHTML = `
            <div class="explorer-header">
                <h2>File Explorer</h2>
                <div class="explorer-actions">
                    <button id="exportAllBtn" class="secondary small" onclick="api.exportAll()" disabled>Export All</button>
                    <button class="secondary small" onclick="Explorer.refresh()">Refresh</button>
                    <button class="danger small" onclick="Explorer.resetAll()">Reset All</button>
                </div>
            </div>
            <div id="sourceTree" class="source-tree">
                <div class="loading">Loading sources</div>
            </div>
        `;
    },

    async refresh() {
        await this.loadSources();
    },

    async loadSources() {
        const tree = document.getElementById('sourceTree');
        const exportBtn = document.getElementById('exportAllBtn');

        try {
            const data = await api.getSqlSources();
            this.sources = data.sources;

            if (this.sources.length === 0) {
                tree.innerHTML = `
                    <div class="empty-state">
                        <div class="empty-state-icon">📂</div>
                        <h3 class="empty-state-title">No sources yet</h3>
                        <p class="empty-state-text">Add your first documents to start building your knowledge base</p>
                        <button onclick="App.switchTab('add-docs')">Add Documents</button>
                    </div>
                `;
                this.expandedSources.clear();
                if (exportBtn) exportBtn.disabled = true;
                return;
            }

            if (exportBtn) exportBtn.disabled = false;
            tree.innerHTML = this.sources.map(s => this.renderSource(s)).join('');

            // Reload docs for any expanded sources
            for (const sourceId of this.expandedSources) {
                // Check if source still exists
                const exists = this.sources.some(s => (s.id || s.name) === sourceId);
                if (exists) {
                    this.loadDocs(sourceId);
                } else {
                    this.expandedSources.delete(sourceId);
                }
            }
        } catch (e) {
            tree.innerHTML = `<div class="message error">Error: ${escapeHtml(e.message)}</div>`;
        }
    },

    renderSource(source) {
        const id = source.id || source.name;
        const isExpanded = this.expandedSources.has(id);
        const docCount = source.doc_count || 0;
        const lastUpdated = (source.last_updated || source.last_indexed) ? formatRelativeTime(source.last_updated || source.last_indexed) : '';

        return `
            <div class="source-node">
                <div class="source-row" onclick="Explorer.toggleSource('${escapeHtml(id)}')">
                    <span class="source-toggle">${isExpanded ? '▼' : '▶'}</span>
                    <span class="source-icon">📁</span>
                    <span class="source-name">${escapeHtml(source.name || source.id)}</span>
                    <span class="source-stats">${docCount} ${docCount === 1 ? 'doc' : 'docs'}</span>
                    ${lastUpdated ? `<span class="source-date">${lastUpdated}</span>` : ''}
                    <div class="source-menu" onclick="event.stopPropagation()">
                        <button class="menu-btn" onclick="Explorer.showSourceMenu(event, '${escapeHtml(id)}')">⋮</button>
                    </div>
                </div>
                <div class="source-docs ${isExpanded ? '' : 'hidden'}" id="docs-${id}">
                    ${isExpanded ? '<div class="loading">Loading</div>' : ''}
                </div>
            </div>
        `;
    },

    async toggleSource(sourceId) {
        const docsContainer = document.getElementById(`docs-${sourceId}`);
        if (!docsContainer) return;

        if (this.expandedSources.has(sourceId)) {
            this.expandedSources.delete(sourceId);
            docsContainer.classList.add('hidden');
            docsContainer.innerHTML = '';
            // Update toggle icon
            const row = docsContainer.previousElementSibling;
            row.querySelector('.source-toggle').textContent = '▶';
        } else {
            this.expandedSources.add(sourceId);
            docsContainer.classList.remove('hidden');
            docsContainer.innerHTML = '<div class="loading">Loading</div>';
            // Update toggle icon
            const row = docsContainer.previousElementSibling;
            row.querySelector('.source-toggle').textContent = '▼';
            await this.loadDocs(sourceId);
        }
    },

    async loadDocs(sourceId, loadAll = false) {
        const container = document.getElementById(`docs-${sourceId}`);
        if (!container) return;

        try {
            // Default to 20 docs, 'all' to load everything
            const limit = loadAll ? 'all' : 20;
            const data = await api.getSqlSourceDocs(sourceId, limit);
            const docs = data.documents || [];
            const total = data.total_documents || docs.length;

            if (docs.length === 0) {
                container.innerHTML = '<div class="empty-docs">No documents in this source</div>';
                return;
            }

            let html = docs.map(doc => this.renderDoc(doc)).join('');

            // Show "Load All" if there are more docs
            if (!loadAll && docs.length < total) {
                html += `
                    <div class="load-more">
                        <button class="secondary small" onclick="Explorer.loadDocs('${escapeHtml(sourceId)}', true)">
                            Load All (${total - docs.length} more)
                        </button>
                    </div>
                `;
            }

            container.innerHTML = html;
        } catch (e) {
            container.innerHTML = `<div class="message error">Error: ${escapeHtml(e.message)}</div>`;
        }
    },

    renderDoc(doc) {
        const createdAt = doc.created_at ? formatRelativeTime(doc.created_at) : '';
        const sizeDisplay = doc.content_length ? formatBytes(doc.content_length) : (doc.chunk_count ? `${doc.chunk_count} chunks` : '');
        return `
            <div class="doc-row">
                <span class="doc-icon">📄</span>
                <span class="doc-name" title="${escapeHtml(doc.title)}">${escapeHtml(doc.title)}</span>
                <span class="doc-stats">${sizeDisplay}</span>
                ${createdAt ? `<span class="doc-date">${createdAt}</span>` : ''}
                <button class="doc-view-btn" onclick="Explorer.viewDoc('${escapeHtml(doc.id)}')" title="View">📖</button>
                <div class="doc-menu">
                    <button class="menu-btn" onclick="Explorer.showDocMenu(event, '${escapeHtml(doc.id)}')">⋮</button>
                </div>
            </div>
        `;
    },

    showSourceMenu(event, sourceId) {
        event.stopPropagation();
        this.closeMenus();

        const menu = document.createElement('div');
        menu.className = 'context-menu';
        menu.id = 'contextMenu';
        menu.innerHTML = `
            <div class="context-menu-item" onclick="Explorer.exportSource('${escapeHtml(sourceId)}')">
                <span>📥</span> Export Source
            </div>
            <div class="context-menu-item danger" onclick="Explorer.deleteSource('${escapeHtml(sourceId)}')">
                <span>🗑️</span> Delete Source
            </div>
        `;

        document.body.appendChild(menu);
        this.positionMenu(menu, event);

        // Close on click outside
        setTimeout(() => {
            document.addEventListener('click', this.closeMenus, { once: true });
        }, 0);
    },

    showDocMenu(event, docId) {
        event.stopPropagation();
        this.closeMenus();

        const menu = document.createElement('div');
        menu.className = 'context-menu';
        menu.id = 'contextMenu';
        menu.innerHTML = `
            <div class="context-menu-item" onclick="Explorer.exportDoc('${escapeHtml(docId)}')">
                <span>📥</span> Export
            </div>
            <div class="context-menu-item danger" onclick="Explorer.deleteDoc('${escapeHtml(docId)}')">
                <span>🗑️</span> Delete
            </div>
        `;

        document.body.appendChild(menu);
        this.positionMenu(menu, event);

        setTimeout(() => {
            document.addEventListener('click', this.closeMenus, { once: true });
        }, 0);
    },

    positionMenu(menu, event) {
        const rect = event.target.getBoundingClientRect();
        menu.style.top = `${rect.bottom + 4}px`;
        menu.style.left = `${rect.left}px`;

        // Keep menu in viewport
        const menuRect = menu.getBoundingClientRect();
        if (menuRect.right > window.innerWidth) {
            menu.style.left = `${window.innerWidth - menuRect.width - 8}px`;
        }
    },

    closeMenus() {
        document.getElementById('contextMenu')?.remove();
    },

    currentViewedDoc: null,

    async viewDoc(docId) {
        this.closeMenus();

        const modal = document.getElementById('docModal');
        const title = document.getElementById('docModalTitle');
        const meta = document.getElementById('docModalMeta');
        const content = document.getElementById('docModalContent');

        title.textContent = 'Loading...';
        meta.textContent = '';
        content.textContent = '';
        this.currentViewedDoc = null;
        modal.classList.add('active');

        try {
            const doc = await api.getDocument(docId);
            if (doc.error) {
                title.textContent = 'Error';
                content.textContent = doc.error;
            } else {
                this.currentViewedDoc = doc;
                title.textContent = doc.title || 'Untitled';
                meta.innerHTML = `
                    Source: ${escapeHtml(doc.source_id || 'Unknown')} |
                    ${doc.content ? doc.content.length.toLocaleString() : 0} characters
                    ${doc.file_path ? ` | File: ${escapeHtml(doc.file_path)}` : ''}
                `;
                content.textContent = doc.content || '';
            }
        } catch (e) {
            title.textContent = 'Error';
            content.textContent = `Failed to load document: ${e.message}`;
        }
    },

    exportCurrentDoc() {
        if (!this.currentViewedDoc) {
            Toast.error('No document loaded');
            return;
        }
        const doc = this.currentViewedDoc;
        const blob = new Blob([doc.content || ''], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = doc.title || 'document.txt';
        a.click();
        URL.revokeObjectURL(url);
        Toast.success('Document exported');
    },

    exportSource(sourceId) {
        this.closeMenus();
        api.exportSource(sourceId);
    },

    async deleteSource(sourceId) {
        this.closeMenus();
        const confirmed = await ConfirmModal.show(
            `Delete source "<strong>${sourceId}</strong>" and all its documents? This cannot be undone.`,
            { title: 'Delete Source', confirmText: 'Delete' }
        );
        if (!confirmed) return;

        try {
            await api.deleteSource(sourceId);
            this.expandedSources.delete(sourceId);
            Toast.success(`Source "${sourceId}" deleted`);
            await this.loadSources();
        } catch (e) {
            Toast.error('Error deleting source: ' + e.message);
        }
    },

    async exportDoc(docId) {
        this.closeMenus();
        try {
            const doc = await api.getDocument(docId);
            if (doc.error) {
                Toast.error('Error: ' + doc.error);
                return;
            }
            // Download as text file
            const blob = new Blob([doc.content || ''], { type: 'text/plain' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = doc.title || 'document.txt';
            a.click();
            URL.revokeObjectURL(url);
            Toast.success('Document exported');
        } catch (e) {
            Toast.error('Error exporting document: ' + e.message);
        }
    },

    async deleteDoc(docId) {
        this.closeMenus();
        const confirmed = await ConfirmModal.show(
            'Delete this document? This cannot be undone.',
            { title: 'Delete Document', confirmText: 'Delete' }
        );
        if (!confirmed) return;

        try {
            await api.deleteDocument(docId);
            Toast.success('Document deleted');
            await this.loadSources();
        } catch (e) {
            Toast.error('Error deleting document: ' + e.message);
        }
    },

    async resetAll() {
        const confirmed = await ConfirmModal.show(
            'This will <strong>permanently delete all sources and documents</strong>. This cannot be undone.',
            { title: 'Reset All Data', confirmText: 'Reset All' }
        );
        if (!confirmed) return;

        try {
            await api.reset();
            Toast.success('All data has been reset');
            await this.loadSources();
        } catch (e) {
            Toast.error('Error resetting data: ' + e.message);
        }
    }
};
