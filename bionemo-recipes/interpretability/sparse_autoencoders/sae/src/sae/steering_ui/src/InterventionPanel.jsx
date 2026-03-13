import React from 'react'

const MODES = [
  { value: 'additive_code', label: 'Additive (code)' },
  { value: 'multiplicative_code', label: 'Multiply (code)' },
  { value: 'direct', label: 'Direct' },
]

const styles = {
  container: {
    flexShrink: 0,
    maxHeight: '40%',
    display: 'flex',
    flexDirection: 'column',
    overflow: 'hidden',
  },
  header: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: '10px 16px 6px',
    flexShrink: 0,
  },
  label: {
    fontSize: '11px',
    fontWeight: '600',
    textTransform: 'uppercase',
    color: '#888',
    letterSpacing: '0.5px',
  },
  clearBtn: {
    fontSize: '11px',
    color: '#c00',
    cursor: 'pointer',
    background: 'none',
    border: 'none',
    padding: '2px 6px',
  },
  list: {
    flex: 1,
    overflowY: 'auto',
    padding: '0 16px 8px',
  },
  empty: {
    padding: '12px 16px',
    fontSize: '12px',
    color: '#bbb',
    fontStyle: 'italic',
  },
  item: {
    padding: '8px 10px',
    marginBottom: '6px',
    background: '#f8f8f8',
    borderRadius: '6px',
    border: '1px solid #e8e8e8',
  },
  itemHeader: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: '6px',
  },
  itemDesc: {
    fontSize: '12px',
    fontWeight: '500',
    flex: 1,
    overflow: 'hidden',
    textOverflow: 'ellipsis',
    whiteSpace: 'nowrap',
    marginRight: '8px',
  },
  removeBtn: {
    background: 'none',
    border: 'none',
    cursor: 'pointer',
    fontSize: '14px',
    color: '#999',
    padding: '0 4px',
    lineHeight: 1,
  },
  controls: {
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
  },
  slider: {
    flex: 1,
    height: '4px',
    cursor: 'pointer',
    accentColor: '#76b900',
  },
  weightValue: {
    fontSize: '12px',
    fontFamily: 'monospace',
    fontWeight: '600',
    minWidth: '36px',
    textAlign: 'right',
  },
  modeSelect: {
    fontSize: '10px',
    padding: '2px 4px',
    border: '1px solid #ddd',
    borderRadius: '3px',
    background: 'white',
    cursor: 'pointer',
  },
  featureIdBadge: {
    fontSize: '9px',
    fontFamily: 'monospace',
    color: '#aaa',
    marginLeft: '6px',
  },
}

export default function InterventionPanel({ interventions, onUpdate, onRemove, onClear }) {
  if (interventions.length === 0) {
    return (
      <div style={styles.container}>
        <div style={styles.header}>
          <span style={styles.label}>Active Interventions</span>
        </div>
        <div style={styles.empty}>
          Click a feature below to add an intervention.
        </div>
      </div>
    )
  }

  return (
    <div style={styles.container}>
      <div style={styles.header}>
        <span style={styles.label}>
          Active Interventions ({interventions.length})
        </span>
        <button style={styles.clearBtn} onClick={onClear}>Clear all</button>
      </div>
      <div style={styles.list}>
        {interventions.map(iv => (
          <div key={iv.feature_id} style={styles.item}>
            <div style={styles.itemHeader}>
              <div style={styles.itemDesc}>
                {iv.description}
                <span style={styles.featureIdBadge}>#{iv.feature_id}</span>
              </div>
              <button
                style={styles.removeBtn}
                onClick={() => onRemove(iv.feature_id)}
                title="Remove intervention"
              >
                &times;
              </button>
            </div>
            <div style={styles.controls}>
              <select
                value={iv.mode}
                onChange={e => onUpdate(iv.feature_id, { mode: e.target.value })}
                style={styles.modeSelect}
              >
                {MODES.map(m => (
                  <option key={m.value} value={m.value}>{m.label}</option>
                ))}
              </select>
              <input
                type="range"
                min="-10"
                max="10"
                step="0.5"
                value={iv.weight}
                onChange={e => onUpdate(iv.feature_id, { weight: parseFloat(e.target.value) })}
                style={styles.slider}
              />
              <span style={{
                ...styles.weightValue,
                color: iv.weight > 0 ? '#2e7d32' : iv.weight < 0 ? '#c62828' : '#666',
              }}>
                {iv.weight > 0 ? '+' : ''}{iv.weight.toFixed(1)}
              </span>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
