import React, { useState, useEffect, useRef } from 'react'
import ProteinSequence, { computeAlignInfo } from './ProteinSequence'
import MolstarThumbnail from './MolstarThumbnail'
import { getAccession, uniprotUrl, codonToAA } from './utils'

// ── Codon-to-AA lookup for logit chart ──────────────────────────────
const CODON_AA = {
  'TTT':'F','TTC':'F','TTA':'L','TTG':'L','CTT':'L','CTC':'L','CTA':'L','CTG':'L',
  'ATT':'I','ATC':'I','ATA':'I','ATG':'M','GTT':'V','GTC':'V','GTA':'V','GTG':'V',
  'TCT':'S','TCC':'S','TCA':'S','TCG':'S','CCT':'P','CCC':'P','CCA':'P','CCG':'P',
  'ACT':'T','ACC':'T','ACA':'T','ACG':'T','GCT':'A','GCC':'A','GCA':'A','GCG':'A',
  'TAT':'Y','TAC':'Y','TAA':'*','TAG':'*','CAT':'H','CAC':'H','CAA':'Q','CAG':'Q',
  'AAT':'N','AAC':'N','AAA':'K','AAG':'K','GAT':'D','GAC':'D','GAA':'E','GAG':'E',
  'TGT':'C','TGC':'C','TGA':'*','TGG':'W','CGT':'R','CGC':'R','CGA':'R','CGG':'R',
  'AGT':'S','AGC':'S','AGA':'R','AGG':'R','GGT':'G','GGC':'G','GGA':'G','GGG':'G',
}

// Group codons by amino acid, excluding stop codons
const AA_GROUPS = {}
for (const [codon, aa] of Object.entries(CODON_AA)) {
  if (aa === '*') continue
  if (!AA_GROUPS[aa]) AA_GROUPS[aa] = []
  AA_GROUPS[aa].push(codon)
}
const AA_ORDER = Object.keys(AA_GROUPS).sort()

// Color palette for amino acid groups
const AA_COLORS = {
  // Nonpolar
  'G': '#e8e8e8', 'A': '#c8c8c8', 'V': '#b0b0b0', 'L': '#a0a0a0', 'I': '#909090',
  'P': '#d0d0a0', 'F': '#c0b0a0', 'W': '#b0a090', 'M': '#a09080',
  // Polar
  'S': '#b0d0ff', 'T': '#a0c0f0', 'C': '#90b0e0', 'Y': '#80a0d0',
  'N': '#a0d0b0', 'Q': '#90c0a0',
  // Charged
  'D': '#ffb0b0', 'E': '#ffa0a0', 'K': '#b0b0ff', 'R': '#a0a0ff', 'H': '#c0b0ff',
}

const styles = {
  overlay: {
    position: 'fixed',
    inset: 0,
    background: 'rgba(0, 0, 0, 0.5)',
    zIndex: 2000,
    overflowY: 'auto',
  },
  page: {
    maxWidth: '960px',
    margin: '20px auto',
    background: '#fff',
    borderRadius: '8px',
    boxShadow: '0 4px 24px rgba(0,0,0,0.2)',
  },
  header: {
    padding: '12px 20px',
    borderBottom: '1px solid #eee',
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  title: {
    fontSize: '14px',
    fontWeight: '700',
  },
  subtitle: {
    fontSize: '11px',
    color: '#666',
  },
  closeBtn: {
    background: 'none',
    border: '1px solid #ddd',
    borderRadius: '4px',
    padding: '3px 10px',
    cursor: 'pointer',
    fontSize: '11px',
    color: '#555',
  },
  section: {
    padding: '10px 20px',
    borderBottom: '1px solid #f0f0f0',
  },
  sectionTitle: {
    fontSize: '11px',
    fontWeight: '600',
    marginBottom: '6px',
    color: '#333',
    textTransform: 'uppercase',
  },
  sectionSubtitle: {
    fontSize: '9px',
    color: '#aaa',
    marginBottom: '6px',
  },
  statsGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(4, 1fr)',
    gap: '8px',
  },
  statCard: {
    background: '#f8f8f8',
    borderRadius: '4px',
    padding: '6px 8px',
    textAlign: 'center',
  },
  statNumber: {
    fontSize: '14px',
    fontWeight: '700',
    color: '#333',
  },
  statLabel: {
    fontSize: '8px',
    color: '#888',
    textTransform: 'uppercase',
    marginTop: '2px',
  },
  tag: {
    display: 'inline-block',
    padding: '3px 8px',
    borderRadius: '4px',
    fontSize: '11px',
    fontWeight: '500',
    marginRight: '6px',
    marginBottom: '4px',
  },
  logitChart: {
    display: 'flex',
    alignItems: 'flex-end',
    gap: '1px',
    height: '120px',
    padding: '0 4px',
  },
  logitBar: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    gap: '1px',
    flex: 1,
    minWidth: 0,
  },
  aaGroupLabel: {
    fontSize: '10px',
    fontWeight: '600',
    color: '#555',
    textAlign: 'center',
    padding: '4px 0 2px',
    borderTop: '1px solid #eee',
    marginTop: '4px',
  },
  example: {
    marginBottom: '6px',
    padding: '6px 8px',
    background: '#fafafa',
    borderRadius: '4px',
    border: '1px solid #eee',
  },
  exampleMeta: {
    fontSize: '10px',
    color: '#666',
    marginBottom: '4px',
    fontFamily: 'monospace',
    display: 'flex',
    justifyContent: 'space-between',
  },
  structureGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(3, 1fr)',
    gap: '6px',
    marginTop: '10px',
  },
}


// ── Vocab Logit Chart Component ─────────────────────────────────────

function VocabLogitChart({ logits }) {
  if (!logits) return null

  // Build codon logit map, excluding stop codons (universally suppressed, not informative)
  const codonLogitMap = {}
  for (const [codon, val] of logits.top_positive) {
    if (CODON_AA[codon] !== '*') codonLogitMap[codon] = val
  }
  for (const [codon, val] of logits.top_negative) {
    if (CODON_AA[codon] !== '*') codonLogitMap[codon] = val
  }

  const maxAbs = Math.max(...Object.values(codonLogitMap).map(Math.abs), 0.001)

  return (
    <div>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px' }}>
        {AA_ORDER.map(aa => {
          const codons = AA_GROUPS[aa] || []
          const aaColor = AA_COLORS[aa] || '#eee'
          return (
            <div key={aa}>
              <div style={{
                fontSize: '9px', fontWeight: '700', color: '#444',
                textAlign: 'center', marginBottom: '2px',
                background: aaColor, borderRadius: '2px', padding: '0 3px',
              }}>
                {aa}
              </div>
              <div style={{ display: 'flex', gap: '1px', alignItems: 'flex-end', height: '40px' }}>
                {codons.sort().map(codon => {
                  const val = codonLogitMap[codon] || 0
                  const h = Math.max(2, (Math.abs(val) / maxAbs) * 34)
                  const isPos = val > 0
                  return (
                    <div key={codon} style={{
                      display: 'flex', flexDirection: 'column', alignItems: 'center',
                      justifyContent: 'flex-end', height: '40px',
                    }} title={`${codon}: ${val.toFixed(3)}`}>
                      <div style={{
                        width: '10px', height: `${h}px`, borderRadius: '1px',
                        background: isPos ? '#76b900' : '#e57373',
                        opacity: Math.abs(val) / maxAbs * 0.7 + 0.3,
                      }} />
                      <span style={{ fontSize: '6px', color: '#aaa', marginTop: '1px', fontFamily: 'monospace' }}>
                        {codon}
                      </span>
                    </div>
                  )
                })}
              </div>
            </div>
          )
        })}
      </div>
      <div style={{ display: 'flex', gap: '10px', marginTop: '4px', fontSize: '8px', color: '#aaa' }}>
        <span><span style={{ display: 'inline-block', width: '8px', height: '8px', background: '#76b900', borderRadius: '1px', marginRight: '3px' }} />Promoted</span>
        <span><span style={{ display: 'inline-block', width: '8px', height: '8px', background: '#e57373', borderRadius: '1px', marginRight: '3px' }} />Suppressed</span>
      </div>
    </div>
  )
}


// ── Codon Annotations Component ─────────────────────────────────────

function CodonAnnotations({ annotations }) {
  if (!annotations || Object.keys(annotations).length === 0) {
    return <div style={{ color: '#999', fontSize: '12px' }}>No significant annotations detected</div>
  }

  const items = []

  if (annotations.amino_acid) {
    items.push({
      label: 'Amino Acid',
      value: `${annotations.amino_acid.aa} (${(annotations.amino_acid.fraction * 100).toFixed(0)}% of activations)`,
      color: '#e3f2fd',
    })
  }
  if (annotations.codon_usage) {
    items.push({
      label: 'Codon Usage',
      value: `Prefers ${annotations.codon_usage.bias} codons (${(annotations.codon_usage.fraction * 100).toFixed(0)}%)`,
      color: '#fff3e0',
    })
  }
  if (annotations.wobble) {
    items.push({
      label: 'Wobble Position',
      value: `${annotations.wobble.preference} preference (${(annotations.wobble.fraction * 100).toFixed(0)}%)`,
      color: '#f3e5f5',
    })
  }
  if (annotations.cpg) {
    items.push({
      label: 'CpG Context',
      value: `Enriched (${(annotations.cpg.enrichment * 100).toFixed(0)}% of activations at CpG boundaries)`,
      color: '#fce4ec',
    })
  }
  if (annotations.position) {
    items.push({
      label: 'Gene Position',
      value: `N-terminal enriched (${annotations.position.enrichment.toFixed(1)}x over expected)`,
      color: '#e8f5e9',
    })
  }

  return (
    <div style={{ display: 'flex', flexWrap: 'wrap', gap: '4px' }}>
      {items.map((item, i) => (
        <div key={i} style={{ background: item.color, borderRadius: '3px', padding: '4px 8px' }}>
          <span style={{ fontSize: '8px', fontWeight: '600', color: '#555', textTransform: 'uppercase' }}>
            {item.label}:
          </span>{' '}
          <span style={{ fontSize: '10px', color: '#333' }}>{item.value}</span>
        </div>
      ))}
    </div>
  )
}


// ── Main Detail Page ────────────────────────────────────────────────

export default function FeatureDetailPage({ feature, examples, vocabLogits, featureAnalysis, onClose }) {
  const [alignMode, setAlignMode] = useState('max_activation')
  const scrollGroupRef = useRef(null)

  const fid = String(feature.feature_id)
  const logits = vocabLogits ? vocabLogits[fid] : null
  const analysis = featureAnalysis ? featureAnalysis[fid] : null

  const freq = feature.activation_freq || 0
  const maxAct = feature.max_activation || 0
  const description = feature.description || `Feature ${feature.feature_id}`

  // Close on Escape
  useEffect(() => {
    const handleKey = (e) => { if (e.key === 'Escape') onClose() }
    document.addEventListener('keydown', handleKey)
    return () => document.removeEventListener('keydown', handleKey)
  }, [onClose])

  const visibleExamples = (examples || []).slice(0, 6)
  const { anchor: alignAnchor, totalLength } = computeAlignInfo(visibleExamples, alignMode)

  return (
    <div style={styles.overlay} onClick={(e) => { if (e.target === e.currentTarget) onClose() }}>
      <div style={styles.page}>

        {/* Header + stats in one row */}
        <div style={styles.header}>
          <div>
            <div style={styles.title}>Feature #{feature.feature_id} <span style={{ fontWeight: 400, fontSize: '11px', color: '#666', marginLeft: '8px' }}>{description}</span></div>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
            <div style={{ display: 'flex', gap: '10px', fontSize: '10px', color: '#666' }}>
              <span>freq: <strong>{(freq * 100).toFixed(1)}%</strong></span>
              <span>max: <strong>{maxAct.toFixed(1)}</strong></span>
            </div>
            <button style={styles.closeBtn} onClick={onClose}>✕</button>
          </div>
        </div>

        {/* Vocabulary Logits */}
        <div style={styles.section}>
          <div style={styles.sectionTitle}>Decoder Logits — Promoted / Suppressed Codons</div>
          <div style={styles.sectionSubtitle}>
            Projection of this feature's decoder weight through the Encodon LM head.
            Shows which codons this feature promotes (green) or suppresses (red) in the model's predictions.
            Stop codons (TAA/TAG/TGA) are excluded — they are uniformly suppressed across all features since the model was trained on coding sequences.
          </div>
          <VocabLogitChart logits={logits} />
        </div>

        {/* Codon Annotations */}
        <div style={styles.section}>
          <div style={styles.sectionTitle}>Codon-Level Annotations</div>
          <div style={styles.sectionSubtitle}>
            Computed per-codon properties correlated with this feature's activations.
          </div>
          <CodonAnnotations annotations={analysis?.codon_annotations} />
        </div>

        {/* Top Activating Sequences */}
        <div style={styles.section}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '6px' }}>
            <div style={styles.sectionTitle}>Top Activating Sequences</div>
            <div style={{ display: 'flex', gap: '4px', fontSize: '10px' }}>
              {['start', 'first_activation', 'max_activation'].map(mode => (
                <button
                  key={mode}
                  onClick={() => setAlignMode(mode)}
                  style={{
                    padding: '2px 8px', borderRadius: '3px', cursor: 'pointer', fontSize: '10px',
                    border: alignMode === mode ? '1px solid #76b900' : '1px solid #ddd',
                    background: alignMode === mode ? '#f0f9e0' : '#fff',
                    color: alignMode === mode ? '#333' : '#555',
                    fontWeight: alignMode === mode ? '600' : '400',
                  }}
                >
                  {mode === 'start' ? 'seq start' : mode === 'first_activation' ? 'first act.' : 'max act.'}
                </button>
              ))}
            </div>
          </div>

          {visibleExamples.length > 0 ? (
            <>
              {visibleExamples.map((ex, i) => (
                <div key={i} style={styles.example}>
                  <div style={styles.exampleMeta}>
                    <span>
                      <strong>{ex.protein_id}</strong>
                      <a
                        href={uniprotUrl(getAccession(ex.protein_id, ex.alphafold_id))}
                        target="_blank"
                        rel="noopener noreferrer"
                        style={{ color: '#2563eb', textDecoration: 'none', marginLeft: '4px', fontSize: '11px' }}
                      >
                        UniProt ↗
                      </a>
                    </span>
                    <span style={{ fontFamily: 'monospace' }}>max: {ex.max_activation?.toFixed(3)}</span>
                  </div>
                  <ProteinSequence
                    sequence={ex.sequence}
                    activations={ex.activations}
                    maxActivation={ex.max_activation}
                    alignMode={alignMode}
                    alignAnchor={alignAnchor}
                    totalLength={totalLength}
                    scrollGroupRef={scrollGroupRef}
                  />
                </div>
              ))}

              {/* 3D Structures */}
              <div style={{ marginTop: '10px' }}>
                <div style={styles.sectionTitle}>3D Structures (AlphaFold)</div>
                <div style={styles.structureGrid}>
                  {visibleExamples.map((ex, i) => (
                    <MolstarThumbnail
                      key={`detail-${feature.feature_id}-${ex.protein_id}-${i}`}
                      proteinId={ex.protein_id}
                      alphafoldId={ex.alphafold_id}
                      sequence={ex.sequence}
                      activations={ex.activations}
                      maxActivation={ex.max_activation}
                    />
                  ))}
                </div>
              </div>
            </>
          ) : (
            <div style={{ color: '#999', fontSize: '12px', fontStyle: 'italic' }}>No examples loaded</div>
          )}
        </div>

      </div>
    </div>
  )
}
