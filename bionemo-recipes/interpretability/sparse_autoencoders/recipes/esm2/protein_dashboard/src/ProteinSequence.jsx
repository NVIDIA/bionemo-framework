import React, { useState } from 'react'

// White-to-NVIDIA-green (#76b900) gradient based on activation value
function activationColorHex(value, maxValue) {
  if (maxValue <= 0 || value <= 0) return 'transparent'
  const n = Math.min(value / maxValue, 1)
  const r = Math.round(255 - n * 137)  // 255 -> 118
  const g = Math.round(255 - n * 70)   // 255 -> 185
  const b = Math.round(255 * (1 - n))  // 255 -> 0
  const toHex = (c) => c.toString(16).padStart(2, '0')
  return `#${toHex(r)}${toHex(g)}${toHex(b)}`
}

const styles = {
  container: {
    fontFamily: 'Monaco, Menlo, "Courier New", monospace',
    fontSize: '11px',
    lineHeight: '1.4',
    overflowX: 'auto',
    whiteSpace: 'nowrap',
    position: 'relative',
  },
  residueRow: {
    display: 'inline-flex',
  },
  residue: {
    display: 'inline-block',
    textAlign: 'center',
    minWidth: '12px',
    cursor: 'default',
    borderRadius: '1px',
  },
  tooltip: {
    position: 'fixed',
    background: '#333',
    color: '#fff',
    padding: '4px 8px',
    borderRadius: '4px',
    fontSize: '10px',
    fontFamily: 'monospace',
    zIndex: 1000,
    pointerEvents: 'none',
    whiteSpace: 'nowrap',
  },
}

export default function ProteinSequence({ sequence, activations, maxActivation }) {
  const [tooltip, setTooltip] = useState(null)

  if (!sequence || sequence.length === 0) {
    return <span style={{ color: '#999' }}>No sequence</span>
  }

  // Trim activations to sequence length (ESM2 may add an extra token)
  const acts = activations ? activations.slice(0, sequence.length) : []
  const maxAct = maxActivation || Math.max(...acts, 0.001)

  const handleMouseEnter = (e, residue, idx, act) => {
    setTooltip({
      x: e.clientX + 10,
      y: e.clientY - 25,
      text: `${residue}${idx + 1} — activation: ${act.toFixed(4)}`,
    })
  }

  const handleMouseMove = (e) => {
    if (tooltip) {
      setTooltip((prev) => prev ? { ...prev, x: e.clientX + 10, y: e.clientY - 25 } : null)
    }
  }

  const handleMouseLeave = () => {
    setTooltip(null)
  }

  return (
    <div style={styles.container}>
      <div style={styles.residueRow}>
        {sequence.split('').map((residue, idx) => {
          const act = acts[idx] || 0
          const bg = activationColorHex(act, maxAct)
          return (
            <span
              key={idx}
              style={{ ...styles.residue, backgroundColor: bg }}
              onMouseEnter={(e) => handleMouseEnter(e, residue, idx, act)}
              onMouseMove={handleMouseMove}
              onMouseLeave={handleMouseLeave}
            >
              {residue}
            </span>
          )
        })}
      </div>
      {tooltip && (
        <span style={{ ...styles.tooltip, left: tooltip.x, top: tooltip.y }}>
          {tooltip.text}
        </span>
      )}
    </div>
  )
}
