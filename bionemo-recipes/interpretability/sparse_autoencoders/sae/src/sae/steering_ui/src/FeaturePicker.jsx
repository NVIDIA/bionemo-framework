import React, { useState, useMemo } from 'react'

const styles = {
  container: {
    flex: 1,
    display: 'flex',
    flexDirection: 'column',
    overflow: 'hidden',
    minHeight: 0,
    borderTop: '1px solid #e0e0e0',
  },
  header: {
    padding: '10px 16px 8px',
    flexShrink: 0,
  },
  label: {
    fontSize: '11px',
    fontWeight: '600',
    textTransform: 'uppercase',
    color: '#888',
    letterSpacing: '0.5px',
  },
  searchRow: {
    display: 'flex',
    gap: '6px',
    padding: '0 16px 8px',
    flexShrink: 0,
  },
  searchInput: {
    flex: 1,
    padding: '6px 10px',
    fontSize: '12px',
    border: '1px solid #ddd',
    borderRadius: '5px',
    outline: 'none',
  },
  sortSelect: {
    padding: '6px 8px',
    fontSize: '11px',
    border: '1px solid #ddd',
    borderRadius: '5px',
    background: 'white',
    cursor: 'pointer',
  },
  list: {
    flex: 1,
    overflowY: 'auto',
    padding: '0 16px 12px',
  },
  item: {
    padding: '8px 10px',
    borderRadius: '6px',
    cursor: 'pointer',
    marginBottom: '4px',
    border: '1px solid #eee',
    transition: 'background 0.1s',
  },
  itemActive: {
    padding: '8px 10px',
    borderRadius: '6px',
    marginBottom: '4px',
    border: '1px solid #76b900',
    background: '#f0fae0',
    opacity: 0.7,
    cursor: 'default',
  },
  itemId: {
    fontSize: '10px',
    color: '#999',
    fontFamily: 'monospace',
  },
  itemDesc: {
    fontSize: '12px',
    fontWeight: '500',
    lineHeight: '1.3',
    marginTop: '1px',
  },
  itemStats: {
    display: 'flex',
    gap: '10px',
    marginTop: '3px',
    fontSize: '10px',
    color: '#888',
    fontFamily: 'monospace',
  },
  count: {
    fontSize: '11px',
    color: '#999',
    padding: '0 16px 4px',
    flexShrink: 0,
  },
}

export default function FeaturePicker({ features, activeFeatureIds, onSelect }) {
  const [search, setSearch] = useState('')
  const [sortBy, setSortBy] = useState('frequency')

  const filtered = useMemo(() => {
    let result = features
    if (search.trim()) {
      const q = search.toLowerCase()
      result = result.filter(f =>
        (f.description || '').toLowerCase().includes(q) ||
        String(f.feature_id).includes(q)
      )
    }
    if (sortBy === 'frequency') {
      result = [...result].sort((a, b) => (b.activation_freq || 0) - (a.activation_freq || 0))
    } else if (sortBy === 'max_activation') {
      result = [...result].sort((a, b) => (b.max_activation || 0) - (a.max_activation || 0))
    } else if (sortBy === 'feature_id') {
      result = [...result].sort((a, b) => a.feature_id - b.feature_id)
    }
    return result
  }, [features, search, sortBy])

  return (
    <div style={styles.container}>
      <div style={styles.header}>
        <div style={styles.label}>Add Features</div>
      </div>
      <div style={styles.searchRow}>
        <input
          type="text"
          placeholder="Search by description or ID..."
          value={search}
          onChange={e => setSearch(e.target.value)}
          style={styles.searchInput}
        />
        <select value={sortBy} onChange={e => setSortBy(e.target.value)} style={styles.sortSelect}>
          <option value="frequency">Freq</option>
          <option value="max_activation">Max</option>
          <option value="feature_id">ID</option>
        </select>
      </div>
      <div style={styles.count}>
        {filtered.length} feature{filtered.length !== 1 ? 's' : ''}
      </div>
      <div style={styles.list}>
        {filtered.slice(0, 100).map(feature => {
          const isActive = activeFeatureIds.has(feature.feature_id)
          return (
            <div
              key={feature.feature_id}
              style={isActive ? styles.itemActive : styles.item}
              onClick={() => !isActive && onSelect(feature)}
              onMouseEnter={e => {
                if (!isActive) e.currentTarget.style.background = '#f8f8f8'
              }}
              onMouseLeave={e => {
                if (!isActive) e.currentTarget.style.background = ''
              }}
            >
              <div style={styles.itemId}>#{feature.feature_id}</div>
              <div style={styles.itemDesc}>
                {feature.description || `Feature ${feature.feature_id}`}
              </div>
              <div style={styles.itemStats}>
                <span>freq: {((feature.activation_freq || 0) * 100).toFixed(1)}%</span>
                <span>max: {(feature.max_activation || 0).toFixed(1)}</span>
              </div>
            </div>
          )
        })}
        {filtered.length > 100 && (
          <div style={{ textAlign: 'center', padding: '8px', color: '#999', fontSize: '11px' }}>
            Showing 100 of {filtered.length}. Refine your search.
          </div>
        )}
        {filtered.length === 0 && (
          <div style={{ textAlign: 'center', padding: '16px', color: '#999', fontSize: '12px' }}>
            No features match your search.
          </div>
        )}
      </div>
    </div>
  )
}
