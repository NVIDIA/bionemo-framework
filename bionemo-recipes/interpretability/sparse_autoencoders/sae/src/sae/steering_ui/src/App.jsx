import React, { useState, useEffect, useCallback } from 'react'
import FeaturePicker from './FeaturePicker'
import InterventionPanel from './InterventionPanel'
import ChatPanel from './ChatPanel'

const API_BASE = '/api'

const styles = {
  container: {
    display: 'flex',
    height: '100vh',
    overflow: 'hidden',
  },
  sidebar: {
    width: '360px',
    flexShrink: 0,
    background: '#fff',
    borderRight: '1px solid #e0e0e0',
    display: 'flex',
    flexDirection: 'column',
    overflow: 'hidden',
  },
  sidebarHeader: {
    padding: '16px 16px 12px',
    borderBottom: '1px solid #e0e0e0',
    flexShrink: 0,
  },
  title: {
    fontSize: '18px',
    fontWeight: '700',
    marginBottom: '2px',
  },
  subtitle: {
    fontSize: '12px',
    color: '#888',
  },
  sidebarContent: {
    flex: 1,
    display: 'flex',
    flexDirection: 'column',
    overflow: 'hidden',
    minHeight: 0,
  },
  main: {
    flex: 1,
    display: 'flex',
    flexDirection: 'column',
    overflow: 'hidden',
    minWidth: 0,
  },
  mainHeader: {
    padding: '12px 20px',
    borderBottom: '1px solid #e0e0e0',
    background: '#fff',
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    flexShrink: 0,
  },
  compareToggle: {
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
    fontSize: '13px',
    color: '#555',
  },
  toggleSwitch: {
    position: 'relative',
    width: '36px',
    height: '20px',
    borderRadius: '10px',
    cursor: 'pointer',
    transition: 'background 0.2s',
  },
  toggleKnob: {
    position: 'absolute',
    top: '2px',
    width: '16px',
    height: '16px',
    borderRadius: '50%',
    background: '#fff',
    transition: 'left 0.2s',
    boxShadow: '0 1px 3px rgba(0,0,0,0.2)',
  },
  modelInfo: {
    fontSize: '12px',
    color: '#999',
    fontFamily: 'monospace',
  },
}

export default function App() {
  const [features, setFeatures] = useState([])
  const [interventions, setInterventions] = useState([])
  const [messages, setMessages] = useState([])
  const [compare, setCompare] = useState(true)
  const [loading, setLoading] = useState(false)
  const [modelInfo, setModelInfo] = useState(null)

  // Fetch features and health on mount
  useEffect(() => {
    fetch(`${API_BASE}/features?limit=500`)
      .then(r => r.json())
      .then(setFeatures)
      .catch(err => console.error('Failed to load features:', err))

    fetch(`${API_BASE}/health`)
      .then(r => r.json())
      .then(setModelInfo)
      .catch(err => console.error('Failed to load health:', err))
  }, [])

  const handleAddIntervention = useCallback((feature) => {
    setInterventions(prev => {
      if (prev.some(iv => iv.feature_id === feature.feature_id)) return prev
      return [...prev, {
        feature_id: feature.feature_id,
        description: feature.description || `Feature ${feature.feature_id}`,
        weight: 3.0,
        mode: 'additive_code',
      }]
    })
  }, [])

  const handleUpdateIntervention = useCallback((featureId, updates) => {
    setInterventions(prev =>
      prev.map(iv => iv.feature_id === featureId ? { ...iv, ...updates } : iv)
    )
  }, [])

  const handleRemoveIntervention = useCallback((featureId) => {
    setInterventions(prev => prev.filter(iv => iv.feature_id !== featureId))
  }, [])

  const handleClearInterventions = useCallback(() => {
    setInterventions([])
  }, [])

  const handleSendMessage = useCallback(async (content) => {
    const newMessages = [...messages, { role: 'user', content }]
    setMessages(newMessages)
    setLoading(true)

    // Prepare steered (and optionally baseline) placeholders
    const steeredMsg = { role: 'assistant', content: '', source: 'steered' }
    const baselineMsg = compare ? { role: 'assistant', content: '', source: 'baseline' } : null

    setMessages(prev => [
      ...prev,
      steeredMsg,
      ...(baselineMsg ? [baselineMsg] : []),
    ])

    try {
      const response = await fetch(`${API_BASE}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          messages: newMessages,
          interventions: interventions.map(iv => ({
            feature_id: iv.feature_id,
            weight: iv.weight,
            mode: iv.mode,
          })),
          compare,
          max_tokens: 256,
        }),
      })

      const reader = response.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''
      let steeredText = ''
      let baselineText = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() || ''

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.slice(6))
              if (data.token !== undefined) {
                if (data.source === 'steered') {
                  steeredText += data.token
                } else if (data.source === 'baseline') {
                  baselineText += data.token
                }

                // Update messages in place
                setMessages(prev => {
                  const updated = [...prev]
                  // Find and update the steered message
                  const steeredIdx = updated.findIndex(
                    (m, i) => i >= newMessages.length && m.source === 'steered'
                  )
                  if (steeredIdx >= 0) {
                    updated[steeredIdx] = { ...updated[steeredIdx], content: steeredText }
                  }
                  // Find and update the baseline message
                  if (compare) {
                    const baselineIdx = updated.findIndex(
                      (m, i) => i >= newMessages.length && m.source === 'baseline'
                    )
                    if (baselineIdx >= 0) {
                      updated[baselineIdx] = { ...updated[baselineIdx], content: baselineText }
                    }
                  }
                  return updated
                })
              }
            } catch (e) {
              // Skip malformed data lines
            }
          }
        }
      }
    } catch (err) {
      console.error('Chat error:', err)
      setMessages(prev => [
        ...prev.slice(0, -1 - (compare ? 1 : 0)),
        { role: 'assistant', content: `Error: ${err.message}`, source: 'error' },
      ])
    }

    setLoading(false)
  }, [messages, interventions, compare])

  const handleClearChat = useCallback(() => {
    setMessages([])
  }, [])

  return (
    <div style={styles.container}>
      <div style={styles.sidebar}>
        <div style={styles.sidebarHeader}>
          <div style={styles.title}>SAE Steering</div>
          <div style={styles.subtitle}>
            Select features and adjust weights to steer model behavior
          </div>
        </div>
        <div style={styles.sidebarContent}>
          <InterventionPanel
            interventions={interventions}
            onUpdate={handleUpdateIntervention}
            onRemove={handleRemoveIntervention}
            onClear={handleClearInterventions}
          />
          <FeaturePicker
            features={features}
            activeFeatureIds={new Set(interventions.map(iv => iv.feature_id))}
            onSelect={handleAddIntervention}
          />
        </div>
      </div>

      <div style={styles.main}>
        <div style={styles.mainHeader}>
          <div style={styles.compareToggle}>
            <div
              onClick={() => setCompare(c => !c)}
              style={{
                ...styles.toggleSwitch,
                background: compare ? '#76b900' : '#ccc',
              }}
            >
              <div style={{
                ...styles.toggleKnob,
                left: compare ? '18px' : '2px',
              }} />
            </div>
            <span>Compare with baseline</span>
          </div>
          {modelInfo && (
            <span style={styles.modelInfo}>
              {modelInfo.model} | layer {modelInfo.layer} | {modelInfo.n_features} features
            </span>
          )}
        </div>
        <ChatPanel
          messages={messages}
          loading={loading}
          compare={compare}
          interventionCount={interventions.length}
          onSend={handleSendMessage}
          onClear={handleClearChat}
        />
      </div>
    </div>
  )
}
