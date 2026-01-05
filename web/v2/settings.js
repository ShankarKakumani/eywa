// ==================== Settings Page ====================
const Settings = {
    settings: null,
    embedders: [],
    rerankers: [],
    activeDownloads: new Map(), // jobId -> intervalId
    selectedEmbedder: null,
    selectedReranker: null,
    selectedDevice: null,

    async init() {
        const panel = document.getElementById('settings');
        // Show skeleton loader that matches actual layout
        panel.innerHTML = this.renderSkeleton();

        try {
            // Load all data in parallel
            const [settingsRes, embeddersRes, rerankersRes] = await Promise.all([
                fetch('/api/settings').then(r => r.json()),
                fetch('/api/models/embedders').then(r => r.json()),
                fetch('/api/models/rerankers').then(r => r.json())
            ]);

            this.settings = settingsRes;
            this.embedders = embeddersRes.models || [];
            this.rerankers = rerankersRes.models || [];

            // Set initial selections
            this.selectedEmbedder = this.settings.embedding_model?.id;
            this.selectedReranker = this.settings.reranker_model?.id;
            this.selectedDevice = this.settings.device;

            this.render();
        } catch (err) {
            panel.innerHTML = `<div class="settings-error">Failed to load settings: ${err.message}</div>`;
        }
    },

    render() {
        const panel = document.getElementById('settings');
        panel.innerHTML = `
            <div class="settings-container">
                <h2>Settings</h2>

                <section class="settings-section">
                    <h3>Embedding Model</h3>
                    <p class="settings-description">Used to convert text into vector embeddings for semantic search.</p>
                    <div class="model-list" id="embedderList">
                        ${this.renderModelList(this.embedders, 'embedder')}
                    </div>
                </section>

                <section class="settings-section">
                    <h3>Reranker Model</h3>
                    <p class="settings-description">Improves search result ranking after initial retrieval.</p>
                    <div class="model-list" id="rerankerList">
                        ${this.renderModelList(this.rerankers, 'reranker')}
                    </div>
                </section>

                <section class="settings-section">
                    <h3>Device Preference</h3>
                    <p class="settings-description">Hardware to use for model inference.</p>
                    <div class="device-pills" id="devicePills">
                        ${this.renderDevicePills()}
                    </div>
                </section>

                <div class="settings-warning">
                    <span class="warning-icon">⚠️</span>
                    Changing the embedding model requires re-indexing all documents.
                </div>
            </div>
        `;

        // Restore any active download progress polling
        this.checkActiveDownloads();
    },

    renderSkeleton() {
        const skeletonCard = `
            <div class="model-card skeleton">
                <div class="model-radio"><div class="skeleton-circle"></div></div>
                <div class="model-info">
                    <div class="skeleton-text" style="width: 60%"></div>
                    <div class="skeleton-text small" style="width: 40%"></div>
                </div>
                <div class="model-status"><div class="skeleton-text" style="width: 80px"></div></div>
            </div>
        `;
        return `
            <div class="settings-container">
                <h2>Settings</h2>
                <section class="settings-section">
                    <h3>Embedding Model</h3>
                    <p class="settings-description">Used to convert text into vector embeddings for semantic search.</p>
                    <div class="model-list">${skeletonCard.repeat(3)}</div>
                </section>
                <section class="settings-section">
                    <h3>Reranker Model</h3>
                    <p class="settings-description">Improves search result ranking after initial retrieval.</p>
                    <div class="model-list">${skeletonCard.repeat(2)}</div>
                </section>
                <section class="settings-section">
                    <h3>Device Preference</h3>
                    <p class="settings-description">Hardware to use for model inference.</p>
                    <div class="device-pills">
                        <div class="skeleton-pill"></div>
                        <div class="skeleton-pill"></div>
                        <div class="skeleton-pill"></div>
                    </div>
                </section>
            </div>
        `;
    },

    renderModelList(models, type) {
        return models.map(model => {
            const isSelected = type === 'embedder'
                ? this.selectedEmbedder === model.id
                : this.selectedReranker === model.id;
            const canSelect = model.downloaded;
            const isDownloading = this.activeDownloads.has(`${type}-${model.id}`);

            // Card is clickable only if model is downloaded
            const clickHandler = canSelect ? `onclick="Settings.selectModel('${type}', '${model.id}')"` : '';

            return `
                <div class="model-card ${isSelected ? 'selected' : ''} ${!canSelect ? 'disabled' : ''} ${canSelect ? 'clickable' : ''}"
                     data-model-id="${model.id}" data-model-type="${type}" ${clickHandler}>
                    <div class="model-radio">
                        <input type="radio" name="${type}-model"
                               id="${type}-${model.id}"
                               value="${model.id}"
                               ${isSelected ? 'checked' : ''}
                               ${!canSelect ? 'disabled' : ''}>
                        <label for="${type}-${model.id}"></label>
                    </div>
                    <div class="model-info">
                        <div class="model-name">${escapeHtml(model.name)}</div>
                        <div class="model-meta">
                            ${model.size_mb} MB
                            ${model.dimensions ? ` · ${model.dimensions} dimensions` : ''}
                        </div>
                    </div>
                    <div class="model-status" onclick="event.stopPropagation()">
                        ${this.renderModelStatus(model, type, isDownloading, isSelected)}
                    </div>
                </div>
            `;
        }).join('');
    },

    renderModelStatus(model, type, isDownloading, isSelected) {
        if (isDownloading) {
            const jobId = this.activeDownloads.get(`${type}-${model.id}`);
            return `<div class="download-progress" id="progress-${type}-${model.id}">
                <div class="progress-bar"><div class="progress-fill" style="width: 0%"></div></div>
                <span class="progress-text">0%</span>
            </div>`;
        }

        if (model.downloaded) {
            // Show delete button only for non-selected models
            if (isSelected) {
                return '<span class="status-downloaded">✓ In Use</span>';
            }
            return `<span class="status-downloaded">✓ Downloaded</span>
                <button class="delete-cache-btn" onclick="Settings.deleteModel('${type}', '${model.id}')" title="Delete cached model">
                    🗑
                </button>`;
        }

        return `<button class="download-btn" onclick="Settings.downloadModel('${type}', '${model.id}')">
            ⬇ Download
        </button>`;
    },

    renderDevicePills() {
        const devices = this.settings.available_devices || ['Auto', 'Cpu'];
        return devices.map(device => {
            const isSelected = this.selectedDevice === device;
            return `
                <button class="device-pill ${isSelected ? 'selected' : ''}"
                        onclick="Settings.selectDevice('${device}')">
                    ${device}
                </button>
            `;
        }).join('');
    },

    async selectModel(type, modelId) {
        // Don't re-select if already selected
        const currentSelection = type === 'embedder' ? this.selectedEmbedder : this.selectedReranker;
        if (currentSelection === modelId) return;

        if (type === 'embedder') {
            this.selectedEmbedder = modelId;
        } else {
            this.selectedReranker = modelId;
        }
        this.updateModelCards(type);
        await this.saveSettings();
    },

    async selectDevice(device) {
        if (this.selectedDevice === device) return;

        this.selectedDevice = device;
        document.querySelectorAll('.device-pill').forEach(pill => {
            pill.classList.toggle('selected', pill.textContent.trim() === device);
        });
        await this.saveSettings();
    },

    updateModelCards(type, forceRender = false) {
        const listId = type === 'embedder' ? 'embedderList' : 'rerankerList';

        // If force render needed (e.g., after download), re-render entire list
        if (forceRender) {
            const models = type === 'embedder' ? this.embedders : this.rerankers;
            document.getElementById(listId).innerHTML = this.renderModelList(models, type);
            return;
        }

        // Otherwise just update visual state (no blink)
        const selectedId = type === 'embedder' ? this.selectedEmbedder : this.selectedReranker;
        const container = document.getElementById(listId);

        container.querySelectorAll('.model-card').forEach(card => {
            const modelId = card.dataset.modelId;
            const isSelected = modelId === selectedId;

            // Toggle selected class
            card.classList.toggle('selected', isSelected);

            // Update radio button
            const radio = card.querySelector('input[type="radio"]');
            if (radio) radio.checked = isSelected;

            // Update status text
            const status = card.querySelector('.model-status');
            if (status && !status.querySelector('.download-progress') && !status.querySelector('.download-btn')) {
                // Only update if it's a downloaded model (not downloading/not downloaded)
                if (isSelected) {
                    status.innerHTML = '<span class="status-downloaded">✓ In Use</span>';
                } else {
                    status.innerHTML = `<span class="status-downloaded">✓ Downloaded</span>
                        <button class="delete-cache-btn" onclick="event.stopPropagation(); Settings.deleteModel('${type}', '${modelId}')" title="Delete cached model">
                            🗑
                        </button>`;
                }
            }
        });
    },

    async downloadModel(type, modelId) {
        const key = `${type}-${modelId}`;

        try {
            // Start download
            const res = await fetch('/api/models/download', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ type, model_id: modelId })
            });

            const data = await res.json();

            if (data.status === 'already_cached') {
                Toast.success(data.message);
                await this.refreshModels();
                return;
            }

            if (data.error) {
                Toast.error(data.error);
                return;
            }

            // Store job ID and start polling
            const jobId = data.job_id;
            this.activeDownloads.set(key, jobId);

            // Re-render to show progress bar
            this.updateModelCards(type, true);

            // Poll for progress
            this.pollDownloadProgress(type, modelId, jobId);

            Toast.info(`Downloading ${data.model_name}...`);
        } catch (err) {
            Toast.error(`Failed to start download: ${err.message}`);
        }
    },

    async pollDownloadProgress(type, modelId, jobId) {
        const key = `${type}-${modelId}`;
        const progressEl = document.getElementById(`progress-${type}-${modelId}`);

        const poll = async () => {
            try {
                const res = await fetch(`/api/models/download/${jobId}`);
                const data = await res.json();

                if (data.error) {
                    this.activeDownloads.delete(key);
                    Toast.error(`Download failed: ${data.error}`);
                    await this.refreshModels();
                    return;
                }

                // Update progress bar
                if (progressEl) {
                    const progress = Math.round((data.total_progress || 0) * 100);
                    const fillEl = progressEl.querySelector('.progress-fill');
                    const textEl = progressEl.querySelector('.progress-text');
                    if (fillEl) fillEl.style.width = `${progress}%`;
                    if (textEl) textEl.textContent = `${progress}%`;
                }

                if (data.status === 'done') {
                    this.activeDownloads.delete(key);
                    Toast.success(`Downloaded ${data.model_name}`);
                    await this.refreshModels();
                    return;
                }

                if (data.status === 'failed') {
                    this.activeDownloads.delete(key);
                    Toast.error(`Download failed: ${data.error || 'Unknown error'}`);
                    await this.refreshModels();
                    return;
                }

                // Continue polling
                setTimeout(poll, 500);
            } catch (err) {
                console.error('Poll error:', err);
                setTimeout(poll, 1000);
            }
        };

        poll();
    },

    async refreshModels() {
        try {
            const [embeddersRes, rerankersRes] = await Promise.all([
                fetch('/api/models/embedders').then(r => r.json()),
                fetch('/api/models/rerankers').then(r => r.json())
            ]);

            this.embedders = embeddersRes.models || [];
            this.rerankers = rerankersRes.models || [];

            this.updateModelCards('embedder', true);
            this.updateModelCards('reranker', true);
        } catch (err) {
            console.error('Failed to refresh models:', err);
        }
    },

    async deleteModel(type, modelId) {
        const model = type === 'embedder'
            ? this.embedders.find(m => m.id === modelId)
            : this.rerankers.find(m => m.id === modelId);

        if (!model) return;

        const confirmed = await ConfirmModal.show(
            `Delete cached model "${model.name}"? This will free up ${model.size_mb} MB of disk space.`,
            { title: 'Delete Model', confirmText: 'Delete', danger: true }
        );

        if (!confirmed) return;

        try {
            const res = await fetch(`/api/models/cache/${type}/${modelId}`, {
                method: 'DELETE'
            });

            const data = await res.json();

            if (data.error) {
                Toast.error(data.error);
            } else {
                Toast.success(data.message);
                await this.refreshModels();
            }
        } catch (err) {
            Toast.error(`Failed to delete model: ${err.message}`);
        }
    },

    async checkActiveDownloads() {
        // Check if there are any active downloads on page load
        try {
            const res = await fetch('/api/models/downloads');
            const data = await res.json();

            for (const download of (data.downloads || [])) {
                if (download.status === 'downloading' || download.status === 'pending') {
                    const key = `${download.model_type}-${download.model_id}`;
                    this.activeDownloads.set(key, download.job_id);
                    this.updateModelCards(download.model_type, true);
                    this.pollDownloadProgress(download.model_type, download.model_id, download.job_id);
                }
            }
        } catch (err) {
            console.error('Failed to check active downloads:', err);
        }
    },

    async saveSettings() {
        try {
            // Find full model configs
            const embedder = this.embedders.find(m => m.id === this.selectedEmbedder);
            const reranker = this.rerankers.find(m => m.id === this.selectedReranker);

            if (!embedder || !reranker) {
                Toast.error('Please select both an embedding model and a reranker model');
                return;
            }

            if (!embedder.downloaded || !reranker.downloaded) {
                Toast.error('Please download the selected models first');
                return;
            }

            const payload = {
                embedding_model: {
                    id: embedder.id,
                    name: embedder.name,
                    repo_id: embedder.repo_id,
                    dimensions: embedder.dimensions,
                    size_mb: embedder.size_mb,
                    curated: embedder.curated
                },
                reranker_model: {
                    id: reranker.id,
                    name: reranker.name,
                    repo_id: reranker.repo_id,
                    size_mb: reranker.size_mb,
                    curated: reranker.curated
                },
                device: this.selectedDevice
            };

            const res = await fetch('/api/settings', {
                method: 'PATCH',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            const data = await res.json();

            if (data.error) {
                Toast.error(data.error);
            } else if (data.model_changed) {
                Toast.info(data.message, 8000);
            } else {
                Toast.success('Settings saved');
            }
        } catch (err) {
            Toast.error(`Failed to save settings: ${err.message}`);
        }
    }
};
