// ==================== Add Documents ====================
const AddDocs = {
    selectedSource: null,
    selectedFiles: [],
    sources: [],
    dropdownOpen: false,

    async init() {
        this.render();
        await this.loadSources();
        this.setupDropdownClose();
    },

    render() {
        const panel = document.getElementById('add-docs');
        panel.innerHTML = `
            <div class="add-docs-step" id="step1">
                <h2 class="step-title">Step 1: Select or Create Source</h2>
                <div class="source-selector">
                    <div class="custom-dropdown" id="sourceDropdown">
                        <button class="dropdown-trigger" type="button" onclick="AddDocs.toggleDropdown()">
                            <span class="dropdown-icon">📁</span>
                            <span class="dropdown-text placeholder" id="dropdownText">Select source...</span>
                            <span class="dropdown-arrow">▼</span>
                        </button>
                        <div class="dropdown-menu" id="dropdownMenu">
                            <div class="dropdown-empty">No sources yet</div>
                        </div>
                    </div>
                    <button class="secondary" onclick="AddDocs.showCreateModal()">+ New Source</button>
                </div>
            </div>

            <div class="add-docs-step hidden" id="step2">
                <h2 class="step-title">Step 2: Add Documents to "<span id="selectedSourceName"></span>"</h2>

                <!-- File Upload -->
                <div class="upload-section">
                    <div class="drop-zone" id="dropZone">
                        <div class="drop-zone-icon">📁</div>
                        <div class="drop-zone-text">Drag & drop files here, or click to browse</div>
                        <div class="drop-zone-hint">Supports: md, txt, rs, py, js, ts, json, yaml, html, css, and more</div>
                    </div>
                    <input type="file" id="fileInput" class="hidden" multiple>
                    <input type="file" id="folderInput" class="hidden" webkitdirectory>

                    <div class="upload-actions">
                        <button class="secondary small" onclick="document.getElementById('fileInput').click()">Select Files</button>
                        <button class="secondary small" onclick="document.getElementById('folderInput').click()">Select Folder</button>
                    </div>

                    <div id="fileList" class="file-list"></div>

                    <div id="uploadProgress" class="progress-container hidden">
                        <div class="progress-bar"><div class="progress-fill" id="progressFill"></div></div>
                        <div class="progress-text" id="progressText">Uploading...</div>
                    </div>

                    <button id="uploadBtn" onclick="AddDocs.uploadFiles()" disabled>Upload Files</button>
                </div>

                <div class="section-divider"><span>add content</span></div>

                <!-- Unified Content Input -->
                <div class="content-section">
                    <div class="url-row compact">
                        <input type="url" id="urlInput" placeholder="Fetch from URL (optional)..." onkeydown="if(event.key==='Enter')AddDocs.fetchUrl()">
                        <button class="secondary small" onclick="AddDocs.fetchUrl()">Fetch</button>
                    </div>
                    <input type="text" id="docTitle" placeholder="Document title">
                    <textarea id="docContent" placeholder="Paste or type your content here..."></textarea>
                    <button onclick="AddDocs.addTextDoc()">Add Document</button>
                </div>

                <div id="addMessage"></div>
            </div>
        `;

        this.setupEventListeners();
    },

    setupEventListeners() {
        // Drop zone
        const dropZone = document.getElementById('dropZone');
        if (dropZone) {
            dropZone.addEventListener('click', () => document.getElementById('fileInput').click());
            dropZone.addEventListener('dragover', (e) => {
                e.preventDefault();
                dropZone.classList.add('drag-over');
            });
            dropZone.addEventListener('dragleave', () => {
                dropZone.classList.remove('drag-over');
            });
            dropZone.addEventListener('drop', (e) => {
                e.preventDefault();
                dropZone.classList.remove('drag-over');
                this.handleFiles(e.dataTransfer.files);
            });
        }

        // File inputs
        document.getElementById('fileInput')?.addEventListener('change', (e) => this.handleFiles(e.target.files));
        document.getElementById('folderInput')?.addEventListener('change', (e) => this.handleFiles(e.target.files));
    },

    async loadSources() {
        try {
            const data = await api.getSqlSources();
            this.sources = data.sources || [];
            this.renderDropdownOptions();

            // Restore selection if source still exists
            if (this.selectedSource) {
                const stillExists = this.sources.some(s => (s.id || s.name) === this.selectedSource);
                if (stillExists) {
                    this.selectSource(this.selectedSource);
                } else {
                    this.selectedSource = null;
                }
            }
        } catch (e) {
            console.error('Failed to load sources:', e);
        }
    },

    renderDropdownOptions() {
        const menu = document.getElementById('dropdownMenu');
        if (!menu) return;

        if (this.sources.length === 0) {
            menu.innerHTML = '<div class="dropdown-empty">No sources yet. Create one!</div>';
            return;
        }

        menu.innerHTML = this.sources.map(s => {
            const id = s.id || s.name;
            const isSelected = this.selectedSource === id;
            return `
                <div class="dropdown-option ${isSelected ? 'selected' : ''}"
                     onclick="AddDocs.selectSource('${escapeHtml(id)}')">
                    <span class="dropdown-option-icon">📁</span>
                    <span class="dropdown-option-text">${escapeHtml(s.name || s.id)}</span>
                    <span class="dropdown-option-meta">${s.doc_count || 0} docs</span>
                </div>
            `;
        }).join('');
    },

    toggleDropdown() {
        this.dropdownOpen = !this.dropdownOpen;
        const trigger = document.querySelector('.dropdown-trigger');
        const menu = document.getElementById('dropdownMenu');

        if (this.dropdownOpen) {
            trigger.classList.add('open');
            menu.classList.add('open');
        } else {
            trigger.classList.remove('open');
            menu.classList.remove('open');
        }
    },

    closeDropdown() {
        this.dropdownOpen = false;
        document.querySelector('.dropdown-trigger')?.classList.remove('open');
        document.getElementById('dropdownMenu')?.classList.remove('open');
    },

    setupDropdownClose() {
        document.addEventListener('click', (e) => {
            const dropdown = document.getElementById('sourceDropdown');
            if (dropdown && !dropdown.contains(e.target)) {
                this.closeDropdown();
            }
        });
    },

    selectSource(sourceId) {
        if (!sourceId) {
            this.selectedSource = null;
            document.getElementById('step2').classList.add('hidden');
            this.updateDropdownDisplay(null);
            return;
        }

        this.selectedSource = sourceId;
        this.closeDropdown();

        // Find source info
        const source = this.sources.find(s => (s.id || s.name) === sourceId);
        this.updateDropdownDisplay(source);

        document.getElementById('selectedSourceName').textContent = sourceId;
        document.getElementById('step2').classList.remove('hidden');
        this.renderDropdownOptions(); // Update selected state
    },

    updateDropdownDisplay(source) {
        const textEl = document.getElementById('dropdownText');
        if (!textEl) return;

        if (source) {
            textEl.textContent = source.name || source.id;
            textEl.classList.remove('placeholder');
        } else {
            textEl.textContent = 'Select source...';
            textEl.classList.add('placeholder');
        }
    },

    showCreateModal() {
        const modal = document.createElement('div');
        modal.className = 'modal-overlay active';
        modal.id = 'createSourceModal';
        modal.innerHTML = `
            <div class="modal">
                <div class="modal-header">
                    <div class="modal-title-row">
                        <span class="modal-icon">📁</span>
                        <h3>Create New Source</h3>
                    </div>
                    <button class="modal-close" onclick="AddDocs.closeCreateModal()">&times;</button>
                </div>
                <div class="modal-header-accent"></div>
                <div class="modal-body">
                    <label class="modal-label">Source Name</label>
                    <div class="modal-input-group">
                        <span class="modal-input-icon">📝</span>
                        <input type="text" id="newSourceName" placeholder="my-new-source" onkeydown="if(event.key==='Enter')AddDocs.createSource()">
                    </div>
                    <p class="modal-hint">Use lowercase letters, numbers, and hyphens only.</p>
                </div>
                <div class="modal-footer">
                    <button class="secondary" onclick="AddDocs.closeCreateModal()">Cancel</button>
                    <button class="btn-primary" onclick="AddDocs.createSource()">Create Source</button>
                </div>
            </div>
        `;
        document.body.appendChild(modal);
        document.getElementById('newSourceName').focus();
    },

    closeCreateModal() {
        document.getElementById('createSourceModal')?.remove();
    },

    createSource() {
        const name = document.getElementById('newSourceName').value.trim();
        if (!name) return;

        // Validate: lowercase, numbers, hyphens only
        if (!/^[a-z0-9-]+$/.test(name)) {
            alert('Source name can only contain lowercase letters, numbers, and hyphens.');
            return;
        }

        // Check if source already exists
        if (this.sources && this.sources.some(s => (s.id || s.name) === name)) {
            alert('A source with this name already exists.');
            return;
        }

        // Add to sources and update dropdown
        this.sources.push({ name, id: name, doc_count: 0, chunk_count: 0 });
        this.renderDropdownOptions();
        this.selectSource(name);
        this.closeCreateModal();
    },

    // File handling
    handleFiles(fileList) {
        const SUPPORTED = new Set([
            'md', 'txt', 'rs', 'py', 'js', 'ts', 'tsx', 'jsx', 'go', 'java',
            'c', 'cpp', 'h', 'hpp', 'json', 'yaml', 'yml', 'toml', 'xml',
            'html', 'css', 'scss', 'sql', 'sh', 'bash', 'zsh', 'fish',
            'dart', 'swift', 'kt', 'kts', 'rb', 'php', 'vue', 'svelte'
        ]);

        const newFiles = Array.from(fileList).filter(file => {
            const ext = file.name.split('.').pop()?.toLowerCase() || '';
            return SUPPORTED.has(ext);
        });

        newFiles.forEach(file => {
            const exists = this.selectedFiles.some(f => f.name === file.name && f.size === file.size);
            if (!exists) this.selectedFiles.push(file);
        });

        this.renderFileList();
    },

    renderFileList() {
        const list = document.getElementById('fileList');
        const btn = document.getElementById('uploadBtn');

        if (this.selectedFiles.length === 0) {
            list.innerHTML = '';
            btn.disabled = true;
            return;
        }

        btn.disabled = false;
        list.innerHTML = this.selectedFiles.map((file, i) => `
            <div class="file-item" data-index="${i}" style="animation-delay: ${i * 0.05}s">
                <span class="file-name" title="${escapeHtml(file.webkitRelativePath || file.name)}">
                    ${escapeHtml(file.webkitRelativePath || file.name)}
                </span>
                <span class="file-size">${formatBytes(file.size)}</span>
                <span class="file-remove" onclick="AddDocs.removeFile(${i})">×</span>
            </div>
        `).join('');
    },

    removeFile(index) {
        const item = document.querySelector(`.file-item[data-index="${index}"]`);
        if (item) {
            item.classList.add('removing');
            setTimeout(() => {
                this.selectedFiles.splice(index, 1);
                this.renderFileList();
            }, 200);
        } else {
            this.selectedFiles.splice(index, 1);
            this.renderFileList();
        }
    },

    async uploadFiles() {
        if (this.selectedFiles.length === 0 || !this.selectedSource) return;

        const msg = document.getElementById('addMessage');
        const progress = document.getElementById('uploadProgress');
        const progressFill = document.getElementById('progressFill');
        const progressText = document.getElementById('progressText');
        const uploadBtn = document.getElementById('uploadBtn');

        progress.classList.remove('hidden');
        uploadBtn.disabled = true;
        msg.innerHTML = '';
        progressText.textContent = 'Reading files...';

        // Read all files
        const documents = [];
        for (const file of this.selectedFiles) {
            try {
                const content = await this.readFile(file);
                const title = file.webkitRelativePath || file.name;
                documents.push({ content, title, file_path: title });
            } catch (e) {
                console.error(`Failed to read ${file.name}:`, e);
            }
        }

        if (documents.length === 0) {
            progress.classList.add('hidden');
            uploadBtn.disabled = false;
            Toast.error('No files could be read');
            return;
        }

        progressText.textContent = 'Queuing documents...';

        try {
            const data = await api.queue(this.selectedSource, documents);
            if (data.error) throw new Error(data.error);

            // Trigger Jobs indicator update
            Jobs.refresh();

            progressText.textContent = `Queued ${data.docs_queued} documents. Processing...`;
            await this.pollJob(data.job_id, progressFill, progressText, msg);
        } catch (e) {
            progress.classList.add('hidden');
            uploadBtn.disabled = false;
            Toast.error(`Error: ${e.message}`);
            return;
        }

        progress.classList.add('hidden');
        uploadBtn.disabled = false;
        this.selectedFiles = [];
        this.renderFileList();
        await this.loadSources(); // Refresh source list
    },

    readFile(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = () => resolve(reader.result);
            reader.onerror = () => reject(new Error('Failed to read file'));
            reader.readAsText(file);
        });
    },

    async pollJob(jobId, progressFill, progressText, msg) {
        return new Promise((resolve) => {
            const poll = async () => {
                try {
                    const job = await api.getJob(jobId);
                    if (job.error) {
                        Toast.error(`Error: ${job.error}`);
                        resolve();
                        return;
                    }

                    const completed = job.completed + job.failed;
                    const percent = job.total > 0 ? (completed / job.total) * 100 : 0;
                    progressFill.style.width = `${percent}%`;

                    progressText.textContent = job.current_doc
                        ? `Processing: ${job.current_doc} (${completed}/${job.total})`
                        : `Processing... (${completed}/${job.total})`;

                    if (job.status === 'done' || job.status === 'failed') {
                        let message = `Completed: ${job.completed} document(s) processed.`;
                        if (job.failed > 0) {
                            message += ` ${job.failed} failed.`;
                            Toast.error(message);
                        } else {
                            Toast.success(message);
                        }
                        resolve();
                        return;
                    }

                    setTimeout(poll, 500);
                } catch (e) {
                    Toast.error(`Polling error: ${e.message}`);
                    resolve();
                }
            };
            poll();
        });
    },

    async addTextDoc() {
        if (!this.selectedSource) return;

        const title = document.getElementById('docTitle').value.trim();
        const content = document.getElementById('docContent').value;
        const msg = document.getElementById('addMessage');

        if (!title) {
            Toast.error('Please enter a title');
            return;
        }

        if (!content) {
            Toast.error('Please enter content');
            return;
        }

        msg.innerHTML = '<div class="loading">Adding document</div>';

        try {
            const data = await api.ingest(this.selectedSource, [{ content, title }]);
            msg.innerHTML = '';
            Toast.success(`Added "${title}" - ${data.chunks_created} chunks created.`);
            document.getElementById('docContent').value = '';
            document.getElementById('docTitle').value = '';
            await this.loadSources();
        } catch (e) {
            msg.innerHTML = '';
            Toast.error(`Error: ${e.message}`);
        }
    },

    async fetchUrl() {
        if (!this.selectedSource) return;

        const url = document.getElementById('urlInput').value.trim();
        const msg = document.getElementById('addMessage');

        if (!url) {
            Toast.error('Please enter a URL');
            return;
        }

        msg.innerHTML = '<div class="loading">Fetching URL</div>';

        try {
            const data = await api.fetchPreview(url);
            msg.innerHTML = '';
            if (data.error) {
                Toast.error(`Error: ${data.error}`);
            } else {
                // Show preview modal
                this.showPreviewModal(data.title, data.content, url);
            }
        } catch (e) {
            msg.innerHTML = '';
            Toast.error(`Error: ${e.message}`);
        }
    },

    showPreviewModal(title, content, url) {
        const existingTitle = document.getElementById('docTitle').value.trim();
        const existingContent = document.getElementById('docContent').value.trim();
        const hasExisting = existingTitle || existingContent;

        const modal = document.createElement('div');
        modal.className = 'modal-overlay active';
        modal.id = 'previewModal';
        modal.innerHTML = `
            <div class="modal" style="max-width: 700px;">
                <div class="modal-header">
                    <h3>Preview: ${escapeHtml(title)}</h3>
                    <button class="modal-close" onclick="AddDocs.closePreviewModal()">&times;</button>
                </div>
                <div class="modal-body">
                    <p style="color: #888; margin-bottom: 12px; font-size: 13px;">Source: ${escapeHtml(url)}</p>
                    <div style="background: #0a0a14; padding: 16px; border-radius: 8px; max-height: 300px; overflow-y: auto; font-size: 13px; line-height: 1.6; white-space: pre-wrap;">${escapeHtml(content.substring(0, 5000))}${content.length > 5000 ? '\n\n[... truncated for preview]' : ''}</div>
                    ${hasExisting ? '<p style="color: #ff9800; margin-top: 12px; font-size: 13px;">⚠️ This will override your existing content in the paste section.</p>' : ''}
                </div>
                <div class="modal-footer">
                    <button class="secondary" onclick="AddDocs.closePreviewModal()">Cancel</button>
                    <button onclick="AddDocs.approvePreview('${escapeHtml(title.replace(/'/g, "\\'"))}')">Use This Content</button>
                </div>
            </div>
        `;
        // Store content for approval
        this.previewContent = content;
        this.previewUrl = url;
        document.body.appendChild(modal);
    },

    closePreviewModal() {
        document.getElementById('previewModal')?.remove();
        this.previewContent = null;
        this.previewUrl = null;
    },

    approvePreview(title) {
        // Pre-fill the paste section
        document.getElementById('docTitle').value = title;
        document.getElementById('docContent').value = this.previewContent || '';
        document.getElementById('urlInput').value = '';

        // Close modal and scroll to paste section
        this.closePreviewModal();

        // Show success toast
        Toast.success('Content loaded. Review and click "Add Document" to save.');

        // Scroll to paste section
        document.getElementById('docTitle').scrollIntoView({ behavior: 'smooth', block: 'center' });
        document.getElementById('docTitle').focus();
    }
};
