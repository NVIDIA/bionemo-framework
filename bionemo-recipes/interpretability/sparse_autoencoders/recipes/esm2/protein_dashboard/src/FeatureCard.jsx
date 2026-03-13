import React, { useState, useEffect, useRef, forwardRef } from 'react'
import ProteinSequence from './ProteinSequence'
import MolstarThumbnail from './MolstarThumbnail'
import ProteinDetailModal from './ProteinDetailModal'
import { getAccession, uniprotUrl } from './utils'

const styles = {
  card: {
    background: '#fff',
    borderRadius: '8px',
    border: '1px solid #e0e0e0',
    flexShrink: 0,
  },
  cardHighlighted: {
    background: '#fff',
    borderRadius: '8px',
    border: '2px solid #222',
    flexShrink: 0,
    boxShadow: '0 2px 8px rgba(0, 0, 0, 0.15)',
  },
  header: {
    padding: '12px 14px',
    borderBottom: '1px solid #eee',
    cursor: 'pointer',
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
    gap: '10px',
  },
  headerLeft: {
    flex: 1,
    minWidth: 0,
  },
  featureId: {
    fontSize: '11px',
    color: '#888',
    fontFamily: 'monospace',
    marginBottom: '2px',
  },
  description: {
    fontSize: '13px',
    fontWeight: '500',
    wordBreak: 'break-word',
    lineHeight: '1.4',
  },
  stats: {
    display: 'flex',
    gap: '12px',
    fontSize: '11px',
    color: '#666',
    flexShrink: 0,
  },
  stat: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'flex-end',
  },
  statLabel: {
    color: '#999',
    fontSize: '9px',
    textTransform: 'uppercase',
  },
  statValue: {
    fontFamily: 'monospace',
    fontWeight: '500',
  },
  expandIcon: {
    color: '#999',
    fontSize: '10px',
    marginLeft: '6px',
  },
  expandedContent: {
    padding: '10px 14px',
    background: '#fafafa',
    maxHeight: '900px',
    overflowY: 'auto',
  },
  sectionHeader: {
    fontSize: '10px',
    color: '#888',
    textTransform: 'uppercase',
    marginBottom: '8px',
    fontWeight: '500',
  },
  example: {
    marginBottom: '8px',
    padding: '8px 10px',
    background: '#fff',
    borderRadius: '4px',
    border: '1px solid #eee',
  },
  exampleMeta: {
    fontSize: '10px',
    color: '#999',
    marginBottom: '4px',
    fontFamily: 'monospace',
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  proteinId: {
    color: '#2563eb',
    fontWeight: '600',
  },
  annotation: {
    color: '#666',
    fontStyle: 'italic',
    marginLeft: '8px',
  },
  uniprotLink: {
    color: '#2563eb',
    textDecoration: 'none',
    fontSize: '11px',
    marginLeft: '4px',
    opacity: 0.6,
  },
  noExamples: {
    color: '#999',
    fontSize: '12px',
    fontStyle: 'italic',
  },
  structureGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(3, 1fr)',
    gap: '8px',
    marginTop: '12px',
  },
  structureHeader: {
    fontSize: '10px',
    color: '#888',
    textTransform: 'uppercase',
    marginTop: '16px',
    marginBottom: '8px',
    fontWeight: '500',
  },
  densityBar: {
    width: '50px',
    height: '3px',
    background: '#eee',
    borderRadius: '2px',
    overflow: 'hidden',
    marginTop: '3px',
  },
  densityFill: {
    height: '100%',
    background: '#76b900',
    borderRadius: '2px',
  },
}

const FeatureCard = forwardRef(function FeatureCard({ feature, isHighlighted, forceExpanded, onClick, loadExamples }, ref) {
  const [expanded, setExpanded] = useState(false)
  const [detailProtein, setDetailProtein] = useState(null)
  const [examples, setExamples] = useState([])
  const [loadingExamples, setLoadingExamples] = useState(false)
  const examplesCacheRef = useRef(null)

  // If forceExpanded changes to true, expand the card
  useEffect(() => {
    if (forceExpanded) {
      setExpanded(true)
    }
  }, [forceExpanded])

  // Lazy-load examples from DuckDB when card is expanded
  useEffect(() => {
    if (!expanded || !loadExamples || examplesCacheRef.current) return
    let cancelled = false
    setLoadingExamples(true)
    loadExamples(feature.feature_id).then(result => {
      if (cancelled) return
      examplesCacheRef.current = result
      setExamples(result)
      setLoadingExamples(false)
    }).catch(err => {
      if (cancelled) return
      console.error('Error loading examples for feature', feature.feature_id, err)
      setLoadingExamples(false)
    })
    return () => { cancelled = true }
  }, [expanded, loadExamples, feature.feature_id])

  const freq = feature.activation_freq || 0
  const maxAct = feature.max_activation || 0
  const description = feature.description || `Feature ${feature.feature_id}`

  const handleClick = () => {
    const willExpand = !expanded
    setExpanded(willExpand)
    if (onClick) {
      onClick(feature.feature_id, willExpand)
    }
  }

  return (
    <div ref={ref} style={isHighlighted ? styles.cardHighlighted : styles.card}>
      <div style={styles.header} onClick={handleClick}>
        <div style={styles.headerLeft}>
          <div style={styles.featureId}>Feature #{feature.feature_id}</div>
          <div style={styles.description}>{description}</div>
        </div>
        <div style={styles.stats}>
          <div style={styles.stat}>
            <span style={styles.statLabel}>Freq</span>
            <span style={styles.statValue}>{(freq * 100).toFixed(1)}%</span>
            <div style={styles.densityBar}>
              <div style={{ ...styles.densityFill, width: `${Math.min(freq * 100 * 10, 100)}%` }} />
            </div>
          </div>
          <div style={styles.stat}>
            <span style={styles.statLabel}>Max</span>
            <span style={styles.statValue}>{maxAct.toFixed(1)}</span>
          </div>
          <span style={styles.expandIcon}>{expanded ? '▼' : '▶'}</span>
        </div>
      </div>

      {expanded && (
        <div style={styles.expandedContent}>
          {/* Protein sequence examples */}
          <div style={styles.sectionHeader}>Top Activating Proteins</div>
          {loadingExamples ? (
            <div style={{ textAlign: 'center', padding: '20px', color: '#888', fontSize: '13px' }}>
              Loading examples...
            </div>
          ) : examples.length > 0 ? (
            <>
              {examples.slice(0, 6).map((ex, i) => (
                <div key={i} style={styles.example}>
                  <div style={styles.exampleMeta}>
                    <span>
                      <span style={styles.proteinId}>{ex.protein_id}</span>
                      <a
                        href={uniprotUrl(getAccession(ex.protein_id, ex.alphafold_id))}
                        target="_blank"
                        rel="noopener noreferrer"
                        style={styles.uniprotLink}
                        onClick={e => e.stopPropagation()}
                        title="View on UniProt"
                      >
                        ↗
                      </a>
                      {ex.best_annotation && (
                        <span style={styles.annotation}>{ex.best_annotation}</span>
                      )}
                    </span>
                    <span>max: {ex.max_activation?.toFixed(3) || 'N/A'}</span>
                  </div>
                  <ProteinSequence
                    sequence={ex.sequence}
                    activations={ex.activations}
                    maxActivation={ex.max_activation}
                  />
                </div>
              ))}

              {/* 2x3 Mol* structure grid */}
              <div style={styles.structureHeader}>3D Structures (AlphaFold)</div>
              <div style={styles.structureGrid}>
                {examples.slice(0, 6).map((ex, i) => (
                  <MolstarThumbnail
                    key={`${feature.feature_id}-${ex.protein_id}-${i}`}
                    proteinId={ex.protein_id}
                    alphafoldId={ex.alphafold_id}
                    sequence={ex.sequence}
                    activations={ex.activations}
                    maxActivation={ex.max_activation}
                    onExpand={() => setDetailProtein(ex)}
                  />
                ))}
              </div>
            </>
          ) : (
            <div style={styles.noExamples}>No examples available</div>
          )}
        </div>
      )}

      {detailProtein && (
        <ProteinDetailModal
          protein={detailProtein}
          onClose={() => setDetailProtein(null)}
        />
      )}
    </div>
  )
})

export default FeatureCard
