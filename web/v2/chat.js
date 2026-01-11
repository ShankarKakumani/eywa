
// Chat Interface Logic
const Chat = {
    history: [],

    init() {
        this.cacheDOM();
        this.bindEvents();
        this.renderHistory();
    },

    cacheDOM() {
        this.container = document.getElementById('chat-container');
        this.msgList = document.getElementById('chat-messages');
        this.input = document.getElementById('chat-input');
        this.sendBtn = document.getElementById('chat-send-btn');
    },

    bindEvents() {
        this.sendBtn.addEventListener('click', () => this.sendMessage());
        this.input.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });

        // Auto-resize input
        this.input.addEventListener('input', function () {
            this.style.height = 'auto';
            this.style.height = (this.scrollHeight) + 'px';
        });
    },

    async sendMessage() {
        const text = this.input.value.trim();
        if (!text) return;

        // Clear input
        this.input.value = '';
        this.input.style.height = 'auto';

        // Add user message
        this.addMessage({ role: 'user', content: text });

        // Show loading state
        this.setLoading(true);

        try {
            // Prepare payload
            const payload = this.history.map(msg => ({
                role: msg.role,
                content: msg.content
            }));

            // Allow for a system prompt if we want one later
            // payload.unshift({ role: 'system', content: 'You are Eywa, a helpful assistant.' });

            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            if (!response.ok) {
                const err = await response.json();
                throw new Error(err.error || 'Failed to send message');
            }

            const data = await response.json();

            // Add assistant response
            this.addMessage({
                role: 'assistant',
                content: data.content,
                usage: data.usage
            });

        } catch (error) {
            console.error('Chat error:', error);
            this.addMessage({
                role: 'assistant',
                content: `⚠️ Error: ${error.message}`,
                isError: true
            });
        } finally {
            this.setLoading(false);
        }
    },

    addMessage(msg) {
        this.history.push(msg);
        this.renderMessage(msg);
        this.scrollToBottom();
    },

    renderMessage(msg) {
        const div = document.createElement('div');
        div.className = `message ${msg.role}`;
        if (msg.isError) div.classList.add('error');

        // Basic Markdown-ish rendering (bold, code blocks) can be added here
        // For now, simple text replacement for newlines
        let content = this.escapeHtml(msg.content).replace(/\n/g, '<br>');

        // Simple code block formatting
        content = content.replace(/```([\s\S]*?)```/g, '<pre><code>$1</code></pre>');

        div.innerHTML = `
            <div class="avatar">${msg.role === 'user' ? '👤' : '🧠'}</div>
            <div class="bubble">
                <div class="content">${content}</div>
                ${msg.usage ? `<div class="meta">${msg.usage.total_tokens} tokens</div>` : ''}
            </div>
        `;

        this.msgList.appendChild(div);
    },

    renderHistory() {
        this.msgList.innerHTML = '';
        this.history.forEach(msg => this.renderMessage(msg));
        this.scrollToBottom();
    },

    scrollToBottom() {
        this.msgList.scrollTop = this.msgList.scrollHeight;
    },

    setLoading(loading) {
        if (loading) {
            const div = document.createElement('div');
            div.className = 'message assistant loading-msg';
            div.innerHTML = `
                <div class="avatar">🧠</div>
                <div class="bubble">
                    <div class="typing-indicator">
                        <span></span><span></span><span></span>
                    </div>
                </div>
            `;
            this.msgList.appendChild(div);
            this.scrollToBottom();
        } else {
            const loader = this.msgList.querySelector('.loading-msg');
            if (loader) loader.remove();
        }
    },

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
};

// Expose to window
window.Chat = Chat;
