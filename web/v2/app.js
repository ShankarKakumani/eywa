// ==================== Toast Notifications ====================
const Toast = {
    container: null,

    init() {
        this.container = document.createElement('div');
        this.container.id = 'toast-container';
        document.body.appendChild(this.container);
    },

    show(message, type = 'info', duration = 4000) {
        if (!this.container) this.init();

        const toast = document.createElement('div');
        toast.className = `toast toast-${type}`;
        const icon = type === 'success' ? '✓' : type === 'error' ? '✕' : 'ℹ';
        toast.innerHTML = `
            <span class="toast-icon">${icon}</span>
            <span class="toast-message">${message}</span>
            <button class="toast-close" onclick="Toast.dismiss(this.parentElement)">×</button>
        `;
        this.container.appendChild(toast);

        if (duration > 0) {
            setTimeout(() => this.dismiss(toast), duration);
        }

        return toast;
    },

    dismiss(toast) {
        if (!toast || !toast.parentElement) return;
        toast.classList.add('removing');
        setTimeout(() => toast.remove(), 300);
    },

    success(msg, duration = 4000) { return this.show(msg, 'success', duration); },
    error(msg, duration = 6000) { return this.show(msg, 'error', duration); },
    info(msg, duration = 4000) { return this.show(msg, 'info', duration); }
};

// ==================== Confirm Modal ====================
const ConfirmModal = {
    show(message, options = {}) {
        return new Promise((resolve) => {
            const {
                title = 'Confirm',
                confirmText = 'Delete',
                cancelText = 'Cancel',
                danger = true
            } = options;

            const modal = document.createElement('div');
            modal.className = 'modal-overlay active';
            modal.id = 'confirmModal';
            modal.innerHTML = `
                <div class="modal modal-confirm">
                    <div class="modal-header">
                        <div class="modal-title-row">
                            <span class="modal-icon">${danger ? '⚠️' : 'ℹ️'}</span>
                            <h3>${title}</h3>
                        </div>
                    </div>
                    <div class="modal-header-accent"></div>
                    <div class="modal-body">
                        <p class="confirm-message">${message}</p>
                    </div>
                    <div class="modal-footer">
                        <button class="secondary" id="confirmCancel">${cancelText}</button>
                        <button class="${danger ? 'danger' : ''}" id="confirmOk">${confirmText}</button>
                    </div>
                </div>
            `;

            document.body.appendChild(modal);

            const cleanup = (result) => {
                modal.remove();
                resolve(result);
            };

            modal.querySelector('#confirmOk').onclick = () => cleanup(true);
            modal.querySelector('#confirmCancel').onclick = () => cleanup(false);
            modal.onclick = (e) => { if (e.target === modal) cleanup(false); };

            // Focus the cancel button by default (safer)
            modal.querySelector('#confirmCancel').focus();
        });
    }
};

// ==================== Utility Functions ====================
function formatRelativeTime(isoString) {
    if (!isoString) return '';
    const date = new Date(isoString);
    const now = new Date();
    const diffMs = now - date;
    const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24));

    if (diffDays === 0) return 'today';
    if (diffDays === 1) return 'yesterday';
    if (diffDays < 7) return `${diffDays}d ago`;
    if (diffDays < 30) return `${Math.floor(diffDays / 7)}w ago`;
    if (diffDays < 365) return `${Math.floor(diffDays / 30)}mo ago`;
    return date.toLocaleDateString();
}

// ==================== Main App ====================
const App = {
    currentTab: 'dashboard',

    async init() {
        this.setupTabs();
        this.setupModals();

        // Initialize jobs indicator
        await Jobs.init();

        // Restore tab from URL hash
        const hash = window.location.hash.slice(1);
        if (hash && ['dashboard', 'add-docs', 'explorer'].includes(hash)) {
            await this.switchTab(hash);
        } else {
            await Dashboard.init();
        }
    },

    setupTabs() {
        document.querySelectorAll('.tab').forEach(tab => {
            tab.addEventListener('click', () => {
                const tabName = tab.dataset.tab;
                this.switchTab(tabName);
            });
        });
    },

    async switchTab(tabName) {
        // Update tab buttons
        document.querySelectorAll('.tab').forEach(t => {
            t.classList.toggle('active', t.dataset.tab === tabName);
        });

        // Update panels
        document.querySelectorAll('.panel').forEach(p => {
            p.classList.toggle('active', p.id === tabName);
        });

        this.currentTab = tabName;
        history.replaceState(null, '', `#${tabName}`);

        // Initialize tab content
        switch (tabName) {
            case 'dashboard':
                await Dashboard.init();
                break;
            case 'add-docs':
                await AddDocs.init();
                break;
            case 'explorer':
                await Explorer.init();
                break;
        }
    },

    setupModals() {
        // Close modals on overlay click
        document.querySelectorAll('.modal-overlay').forEach(overlay => {
            overlay.addEventListener('click', (e) => {
                if (e.target === overlay) {
                    overlay.classList.remove('active');
                }
            });
        });

        // Close on Escape
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                document.querySelectorAll('.modal-overlay.active').forEach(m => {
                    m.classList.remove('active');
                });
            }
        });
    }
};

// Modal close function (called from HTML)
function closeDocModal() {
    document.getElementById('docModal').classList.remove('active');
}

// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', () => App.init());
