// ==================== Jobs Indicator ====================
const Jobs = {
    polling: null,
    jobs: [],
    panelOpen: false,
    expandedJobs: new Set(),

    async init() {
        this.renderIndicator();
        await this.refresh();
        this.startPolling();
    },

    async refresh() {
        try {
            const data = await api.getJobs();
            this.jobs = data.jobs || [];
            this.updateIndicator();
            if (this.panelOpen) {
                this.updatePanel();
            }
        } catch (e) {
            console.error('Failed to refresh jobs:', e);
        }
    },

    startPolling() {
        if (this.polling) return;
        this.polling = setInterval(() => this.refresh(), 2000);
    },

    stopPolling() {
        if (this.polling) {
            clearInterval(this.polling);
            this.polling = null;
        }
    },

    get activeJobs() {
        return this.jobs.filter(j => j.status === 'pending' || j.status === 'processing');
    },

    get pendingCount() {
        return this.activeJobs.reduce((sum, j) => sum + (j.total - j.completed - j.failed), 0);
    },

    get hasErrors() {
        return this.jobs.some(j => j.failed > 0);
    },

    renderIndicator() {
        const header = document.querySelector('header .actions');
        if (!header) return;

        const indicator = document.createElement('div');
        indicator.id = 'jobsIndicator';
        indicator.className = 'jobs-indicator';
        indicator.innerHTML = `
            <button onclick="Jobs.togglePanel()" title="Processing queue">
                <span class="jobs-icon">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <polygon points="12 2 2 7 12 12 22 7 12 2"/>
                        <polyline points="2 17 12 22 22 17"/>
                        <polyline points="2 12 12 17 22 12"/>
                    </svg>
                </span>
                <span class="jobs-badge" id="jobsBadge"></span>
            </button>
        `;
        header.insertBefore(indicator, header.firstChild);
    },

    updateIndicator() {
        const indicator = document.getElementById('jobsIndicator');
        const badge = document.getElementById('jobsBadge');
        if (!indicator || !badge) return;

        const pending = this.pendingCount;
        const hasActive = this.activeJobs.length > 0;

        if (pending > 0) {
            badge.textContent = pending > 99 ? '99+' : pending;
            badge.className = 'jobs-badge visible' + (this.hasErrors ? ' error' : '');
            indicator.classList.add('active');
        } else if (hasActive) {
            badge.textContent = '';
            badge.className = 'jobs-badge';
            indicator.classList.add('active');
        } else {
            badge.textContent = '';
            badge.className = 'jobs-badge';
            indicator.classList.remove('active');
        }

        // Show/hide indicator based on having any jobs
        if (this.jobs.length > 0) {
            indicator.classList.add('visible');
        } else {
            indicator.classList.remove('visible');
        }
    },

    togglePanel() {
        if (this.panelOpen) {
            this.closePanel();
        } else {
            this.openPanel();
        }
    },

    openPanel() {
        this.closePanel();

        const panel = document.createElement('div');
        panel.id = 'jobsPanel';
        panel.className = 'jobs-panel';
        panel.innerHTML = `
            <div class="jobs-panel-header">
                <h3>Processing Queue</h3>
                <button class="close-btn" onclick="Jobs.closePanel()">&times;</button>
            </div>
            <div class="jobs-panel-content" id="jobsPanelContent">
                <div class="loading">Loading...</div>
            </div>
        `;

        document.body.appendChild(panel);
        this.panelOpen = true;

        // Position panel below indicator
        const indicator = document.getElementById('jobsIndicator');
        if (indicator) {
            const rect = indicator.getBoundingClientRect();
            panel.style.top = `${rect.bottom + 8}px`;
            panel.style.right = `${window.innerWidth - rect.right}px`;
        }

        // Close on outside click
        setTimeout(() => {
            document.addEventListener('click', this.handleOutsideClick);
        }, 0);

        this.updatePanel();
    },

    closePanel() {
        const panel = document.getElementById('jobsPanel');
        if (panel) {
            panel.remove();
        }
        this.panelOpen = false;
        document.removeEventListener('click', this.handleOutsideClick);
    },

    handleOutsideClick: function(e) {
        const panel = document.getElementById('jobsPanel');
        const indicator = document.getElementById('jobsIndicator');
        if (panel && !panel.contains(e.target) && !indicator.contains(e.target)) {
            Jobs.closePanel();
        }
    },

    async updatePanel() {
        const content = document.getElementById('jobsPanelContent');
        if (!content) return;

        if (this.jobs.length === 0) {
            content.innerHTML = '<div class="empty-jobs">No processing jobs</div>';
            return;
        }

        // Show recent jobs (last 10, or all active ones)
        const recentJobs = this.jobs
            .filter(j => j.status === 'pending' || j.status === 'processing' || this.expandedJobs.has(j.job_id))
            .slice(0, 10);

        const completedJobs = this.jobs
            .filter(j => (j.status === 'done' || j.status === 'failed') && !this.expandedJobs.has(j.job_id))
            .slice(0, 10);

        let html = '';

        for (const job of recentJobs) {
            html += await this.renderJob(job, true);
        }

        if (completedJobs.length > 0) {
            html += '<div class="jobs-section-title">Completed</div>';
            for (const job of completedJobs) {
                html += this.renderJobSummary(job);
            }
        }

        content.innerHTML = html;
    },

    async renderJob(job, fetchDocs = false) {
        const isExpanded = this.expandedJobs.has(job.job_id);
        const progress = job.total > 0 ? ((job.completed + job.failed) / job.total * 100).toFixed(0) : 0;
        const statusIcon = this.getStatusIcon(job.status);

        let docsHtml = '';
        if (isExpanded && fetchDocs) {
            try {
                const data = await api.getJobDocs(job.job_id);
                const docs = data.documents || [];
                docsHtml = '<div class="job-docs">' +
                    docs.map(doc => this.renderDoc(doc)).join('') +
                    '</div>';
            } catch (e) {
                docsHtml = '<div class="job-docs error">Failed to load documents</div>';
            }
        }

        return `
            <div class="job-item ${job.status}" data-job-id="${job.job_id}">
                <div class="job-header" onclick="Jobs.toggleJob('${job.job_id}')">
                    <span class="job-toggle">${isExpanded ? '&#9660;' : '&#9654;'}</span>
                    <span class="job-icon">${statusIcon}</span>
                    <span class="job-source">${escapeHtml(job.source_id)}</span>
                    <span class="job-progress">${job.completed}/${job.total}</span>
                    ${job.failed > 0 ? `<span class="job-errors">${job.failed} failed</span>` : ''}
                </div>
                <div class="job-progress-bar">
                    <div class="job-progress-fill" style="width: ${progress}%"></div>
                </div>
                ${job.current_doc ? `<div class="job-current">Processing: ${escapeHtml(job.current_doc)}</div>` : ''}
                ${docsHtml}
            </div>
        `;
    },

    renderJobSummary(job) {
        const statusIcon = this.getStatusIcon(job.status);
        return `
            <div class="job-item summary ${job.status}" onclick="Jobs.toggleJob('${job.job_id}')">
                <span class="job-icon">${statusIcon}</span>
                <span class="job-source">${escapeHtml(job.source_id)}</span>
                <span class="job-progress">${job.completed}/${job.total}</span>
                ${job.failed > 0 ? `<span class="job-errors">${job.failed} failed</span>` : ''}
            </div>
        `;
    },

    renderDoc(doc) {
        const icon = this.getDocIcon(doc.status);
        const title = doc.title || doc.file_path || 'Untitled';
        return `
            <div class="job-doc ${doc.status}" title="${doc.error ? escapeHtml(doc.error) : ''}">
                <span class="doc-status">${icon}</span>
                <span class="doc-title">${escapeHtml(title)}</span>
                ${doc.error ? '<span class="doc-error-indicator" title="Click for error">!</span>' : ''}
            </div>
        `;
    },

    getStatusIcon(status) {
        switch (status) {
            case 'pending': return '<span class="status-pending">&#9711;</span>';
            case 'processing': return '<span class="status-processing">&#8987;</span>';
            case 'done': return '<span class="status-done">&#10004;</span>';
            case 'failed': return '<span class="status-failed">&#10008;</span>';
            default: return '?';
        }
    },

    getDocIcon(status) {
        switch (status) {
            case 'pending': return '&#9711;';
            case 'processing': return '&#8987;';
            case 'done': return '&#10004;';
            case 'failed': return '&#10008;';
            default: return '?';
        }
    },

    async toggleJob(jobId) {
        if (this.expandedJobs.has(jobId)) {
            this.expandedJobs.delete(jobId);
        } else {
            this.expandedJobs.add(jobId);
        }
        await this.updatePanel();
    }
};
