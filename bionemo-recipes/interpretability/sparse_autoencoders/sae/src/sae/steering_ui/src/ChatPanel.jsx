import React, { useState, useRef, useEffect } from 'react'

const styles = {
  container: {
    flex: 1,
    display: 'flex',
    flexDirection: 'column',
    overflow: 'hidden',
    background: '#f5f5f5',
  },
  messages: {
    flex: 1,
    overflowY: 'auto',
    padding: '20px',
    display: 'flex',
    flexDirection: 'column',
    gap: '16px',
  },
  empty: {
    flex: 1,
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    color: '#bbb',
    gap: '8px',
  },
  emptyTitle: {
    fontSize: '18px',
    fontWeight: '600',
    color: '#999',
  },
  emptyHint: {
    fontSize: '13px',
    maxWidth: '360px',
    textAlign: 'center',
    lineHeight: '1.5',
  },
  userMessage: {
    alignSelf: 'flex-end',
    maxWidth: '70%',
    padding: '10px 14px',
    background: '#1a1a1a',
    color: '#fff',
    borderRadius: '16px 16px 4px 16px',
    fontSize: '14px',
    lineHeight: '1.5',
    whiteSpace: 'pre-wrap',
  },
  assistantRow: {
    alignSelf: 'flex-start',
    maxWidth: '90%',
    width: '100%',
  },
  compareRow: {
    display: 'grid',
    gridTemplateColumns: '1fr 1fr',
    gap: '12px',
  },
  singleRow: {
    display: 'flex',
  },
  responseCard: {
    padding: '12px 14px',
    background: '#fff',
    borderRadius: '12px',
    border: '1px solid #e0e0e0',
    fontSize: '14px',
    lineHeight: '1.6',
    whiteSpace: 'pre-wrap',
  },
  responseLabel: {
    fontSize: '10px',
    fontWeight: '600',
    textTransform: 'uppercase',
    letterSpacing: '0.5px',
    marginBottom: '6px',
    display: 'flex',
    alignItems: 'center',
    gap: '6px',
  },
  steeredLabel: {
    color: '#76b900',
  },
  baselineLabel: {
    color: '#888',
  },
  dot: {
    width: '6px',
    height: '6px',
    borderRadius: '50%',
    display: 'inline-block',
  },
  errorCard: {
    padding: '12px 14px',
    background: '#fff5f5',
    borderRadius: '12px',
    border: '1px solid #ffcdd2',
    color: '#c62828',
    fontSize: '13px',
  },
  inputArea: {
    padding: '12px 20px 16px',
    background: '#fff',
    borderTop: '1px solid #e0e0e0',
    display: 'flex',
    gap: '10px',
    alignItems: 'flex-end',
    flexShrink: 0,
  },
  textarea: {
    flex: 1,
    padding: '10px 14px',
    fontSize: '14px',
    border: '1px solid #ddd',
    borderRadius: '12px',
    outline: 'none',
    resize: 'none',
    fontFamily: 'inherit',
    lineHeight: '1.4',
    maxHeight: '120px',
    minHeight: '42px',
  },
  sendBtn: {
    padding: '10px 20px',
    fontSize: '13px',
    fontWeight: '600',
    border: 'none',
    borderRadius: '10px',
    cursor: 'pointer',
    flexShrink: 0,
    transition: 'background 0.15s',
  },
  clearBtn: {
    padding: '10px 14px',
    fontSize: '12px',
    background: 'none',
    border: '1px solid #ddd',
    borderRadius: '10px',
    cursor: 'pointer',
    color: '#888',
    flexShrink: 0,
  },
  cursor: {
    display: 'inline-block',
    width: '2px',
    height: '14px',
    background: '#76b900',
    marginLeft: '1px',
    verticalAlign: 'text-bottom',
    animation: 'blink 1s step-end infinite',
  },
}

// Add blink keyframe
if (typeof document !== 'undefined' && !document.getElementById('steering-blink-style')) {
  const style = document.createElement('style')
  style.id = 'steering-blink-style'
  style.textContent = '@keyframes blink { 50% { opacity: 0; } }'
  document.head.appendChild(style)
}

export default function ChatPanel({ messages, loading, compare, interventionCount, onSend, onClear }) {
  const [input, setInput] = useState('')
  const messagesEndRef = useRef(null)
  const textareaRef = useRef(null)

  // Auto-scroll on new messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  // Auto-resize textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto'
      textareaRef.current.style.height = Math.min(textareaRef.current.scrollHeight, 120) + 'px'
    }
  }, [input])

  const handleSubmit = () => {
    if (!input.trim() || loading) return
    onSend(input.trim())
    setInput('')
  }

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSubmit()
    }
  }

  // Group messages for compare mode rendering
  const renderMessages = () => {
    const elements = []
    let i = 0
    while (i < messages.length) {
      const msg = messages[i]

      if (msg.role === 'user') {
        elements.push(
          <div key={i} style={styles.userMessage}>{msg.content}</div>
        )
        i++
      } else if (msg.source === 'error') {
        elements.push(
          <div key={i} style={styles.assistantRow}>
            <div style={styles.errorCard}>{msg.content}</div>
          </div>
        )
        i++
      } else if (msg.source === 'steered') {
        const steered = msg
        const baseline = (i + 1 < messages.length && messages[i + 1].source === 'baseline')
          ? messages[i + 1]
          : null
        const isStreaming = loading && i >= messages.length - (baseline ? 2 : 1)

        if (baseline) {
          // Compare mode: side by side
          elements.push(
            <div key={i} style={styles.assistantRow}>
              <div style={styles.compareRow}>
                <div>
                  <div style={{ ...styles.responseLabel, ...styles.steeredLabel }}>
                    <span style={{ ...styles.dot, background: '#76b900' }} />
                    Steered
                  </div>
                  <div style={styles.responseCard}>
                    {steered.content || '\u00A0'}
                    {isStreaming && !steered.content && <span style={styles.cursor} />}
                  </div>
                </div>
                <div>
                  <div style={{ ...styles.responseLabel, ...styles.baselineLabel }}>
                    <span style={{ ...styles.dot, background: '#ccc' }} />
                    Baseline
                  </div>
                  <div style={styles.responseCard}>
                    {baseline.content || '\u00A0'}
                    {isStreaming && steered.content && !baseline.content && <span style={styles.cursor} />}
                  </div>
                </div>
              </div>
            </div>
          )
          i += 2
        } else {
          // Single mode
          elements.push(
            <div key={i} style={styles.assistantRow}>
              <div style={styles.singleRow}>
                <div style={styles.responseCard}>
                  {steered.content || '\u00A0'}
                  {isStreaming && <span style={styles.cursor} />}
                </div>
              </div>
            </div>
          )
          i++
        }
      } else {
        // Fallback for any other message type
        elements.push(
          <div key={i} style={styles.assistantRow}>
            <div style={styles.responseCard}>{msg.content}</div>
          </div>
        )
        i++
      }
    }
    return elements
  }

  return (
    <div style={styles.container}>
      <div style={styles.messages}>
        {messages.length === 0 ? (
          <div style={styles.empty}>
            <div style={styles.emptyTitle}>SAE Feature Steering</div>
            <div style={styles.emptyHint}>
              {interventionCount > 0
                ? `${interventionCount} intervention${interventionCount > 1 ? 's' : ''} active. Send a message to see how steering affects the model's output.`
                : 'Add feature interventions from the sidebar, then start chatting to see their effect on generation.'}
            </div>
          </div>
        ) : (
          renderMessages()
        )}
        <div ref={messagesEndRef} />
      </div>

      <div style={styles.inputArea}>
        <textarea
          ref={textareaRef}
          rows={1}
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={interventionCount > 0
            ? 'Send a message to test steering...'
            : 'Type a message (add interventions for steering)...'}
          style={styles.textarea}
          disabled={loading}
        />
        <button
          onClick={handleSubmit}
          disabled={!input.trim() || loading}
          style={{
            ...styles.sendBtn,
            background: !input.trim() || loading ? '#e0e0e0' : '#1a1a1a',
            color: !input.trim() || loading ? '#999' : '#fff',
          }}
        >
          {loading ? 'Generating...' : 'Send'}
        </button>
        {messages.length > 0 && (
          <button onClick={onClear} style={styles.clearBtn}>
            Clear
          </button>
        )}
      </div>
    </div>
  )
}
