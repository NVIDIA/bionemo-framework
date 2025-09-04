import React, { useState } from 'react';
import * as d3 from 'd3';
import './HyenaHighlight.css';

const HyenaHighlight = () => {
  const [hoveredIndex, setHoveredIndex] = useState(null);

  // Mock data - DNA sequence (220 characters for more compact view) - Fixed sequence
  const [text] = useState(() => {
    const bases = ['A', 'T', 'G', 'C'];
    let sequence = '';
    // Use a seed-like approach for consistent generation
    const seedValues = [0.2, 0.8, 0.1, 0.9, 0.3, 0.7, 0.4, 0.6]; // Fixed values
    for (let i = 0; i < 220; i++) {
      const seedIndex = i % seedValues.length;
      const baseIndex = Math.floor(seedValues[seedIndex] * 4);
      sequence += bases[baseIndex];
    }
    return sequence;
  });
  const chars = text.split('');

  // Mock attention weights for different Hyena variants
  const generateHyenaWeights = variant => {
    const weights = [];
    for (let i = 0; i < chars.length; i++) {
      const charWeights = [];
      for (let j = 0; j <= i; j++) {
        const distance = i - j;
        let weight = 0;

        if (variant === 'short') {
          // Hyena-SE: Short explicit filters (4-7 length), local attention
          if (distance < 7) {
            weight = Math.exp(-distance * 0.5) * (0.3 + Math.random() * 0.7);
          } else {
            weight = 0; // No attention beyond filter length
          }
        } else if (variant === 'medium') {
          // Hyena-MR: Medium regularized filters (~100 length), exponential decay
          if (distance < 100) {
            const alpha = 0.03; // Fixed decay rate for ~100 character length
            weight = Math.exp(-alpha * distance) * (0.4 + Math.random() * 0.4);
          } else {
            weight = 0; // Cut off at 100
          }
        } else if (variant === 'long') {
          // Hyena-LI: Long implicit filters, global convolution with moderate sparsity
          // Global attention with slight decay and random variation
          const globalDecay = 0.001; // Very slight decay for global pattern
          weight =
            Math.exp(-globalDecay * distance) * (0.2 + Math.random() * 0.3);

          // Add some structure/peaks at regular intervals
          if (distance % 20 === 0) {
            weight *= 1.5;
          }

          // Moderate amount of sparsity - randomly zero out ~25% of weights
          if (Math.random() < 0.25) {
            weight = 0;
          }
        }

        // Boost attention for complementary DNA bases (A-T, G-C pairs)
        if (
          (chars[j] === 'A' && chars[i] === 'T') ||
          (chars[j] === 'T' && chars[i] === 'A') ||
          (chars[j] === 'G' && chars[i] === 'C') ||
          (chars[j] === 'C' && chars[i] === 'G')
        ) {
          weight *= 1.3;
        }

        // Boost attention for same bases (repetitive patterns)
        if (chars[j] === chars[i]) {
          weight *= 1.2;
        }

        charWeights.push(Math.min(Math.max(weight, 0), 1));
      }
      weights.push(charWeights);
    }
    return weights;
  };

  const [shortWeights] = useState(() => generateHyenaWeights('short'));
  const [mediumWeights] = useState(() => generateHyenaWeights('medium'));
  const [longWeights] = useState(() => generateHyenaWeights('long'));

  // Select weights based on section being visualized
  const getWeightsForSection = sectionIndex => {
    if (sectionIndex === 0) return shortWeights;
    if (sectionIndex === 1) return mediumWeights;
    return longWeights;
  };

  // Color scale for attention weights using CSS custom properties
  const colorScale = d3
    .scaleSequential()
    .domain([0, 1])
    .interpolator(t =>
      d3.interpolateRgb(
        '#e8f5d0',
        getComputedStyle(document.documentElement)
          .getPropertyValue('--primary')
          .trim() || '#76b900'
      )(t)
    );

  const getCharacterClasses = (charIndex, sectionIndex) => {
    let classes = ['dna-character'];

    if (hoveredIndex !== null) {
      if (charIndex === hoveredIndex) {
        classes.push('dna-character--hovered');
      } else {
        const weights = getWeightsForSection(sectionIndex);
        if (charIndex <= hoveredIndex && weights[hoveredIndex]) {
          const weight = weights[hoveredIndex][charIndex];
          if (weight > 0.05) {
            classes.push('dna-character--attended');
          }
        }
      }
    }

    return classes.join(' ');
  };

  const getCharacterStyle = (charIndex, sectionIndex) => {
    if (hoveredIndex === null) return {};

    const weights = getWeightsForSection(sectionIndex);

    // If this character is attended to by the hovered character
    if (
      charIndex <= hoveredIndex &&
      weights[hoveredIndex] &&
      charIndex !== hoveredIndex
    ) {
      const weight = weights[hoveredIndex][charIndex];
      if (weight > 0.05) {
        return {
          backgroundColor: colorScale(weight),
          color: weight > 0.5 ? 'white' : 'var(--text-primary, #2d2d2d)',
        };
      }
    }

    return {};
  };

  const handleMouseEnter = index => {
    setHoveredIndex(index);
  };

  const handleMouseLeave = () => {
    setHoveredIndex(null);
  };

  return (
    <>
      <p className="body-text">
        <b>Hyena Operators and DNA</b>
        <br />
        Let's build intuition for how Hyena operators work by exploring their
        attention patterns on a DNA sequence. Interact with the figure below
        yourself to see what information the different operators use for their
        predictions:
      </p>
      <div className="hyena-outer-container">
        <div className="hyena-container">
          <div className="hyena-header">
            <h1 className="hyena-title">Casual Hyena Maps of Attention</h1>

            {/* Legend */}
            <div className="hyena-legend">
              <div className="legend-item">
                <div className="legend-color legend-color--current"></div>
                <span className="legend-text">Current</span>
              </div>
              <div className="legend-item">
                <div className="legend-color legend-color--low"></div>
                <span className="legend-text">Low attention</span>
              </div>
              <div className="legend-item">
                <div className="legend-color legend-color--high"></div>
                <span className="legend-text">High attention</span>
              </div>
            </div>
          </div>

          {/* Three text sections stacked vertically */}
          {[
            {
              name: 'Hyena-SE',
              desc: 'Short Explicit filters (4-7): Focuses on recalling information from nearby tokens (local context).',
            },
            {
              name: 'Hyena-MR',
              desc: 'Medium Regularized filters (~100): Handles dependencies with exponential decay over ~100 tokens.',
            },
            {
              name: 'Hyena-LI',
              desc: 'Long Implicit filters: Global convolution with moderate sparsity across the entire sequence.',
            },
          ].map((modelType, sectionIndex) => (
            <div key={sectionIndex} className="hyena-section">
              <div className="section-header">
                <h3 className="section-title">{modelType.name}</h3>
                <span className="section-description">{modelType.desc}</span>
              </div>

              <div className="dna-container">
                {/* DNA sequence with interactive characters - 5 rows to accommodate 220 characters */}
                <div className="dna-sequence">
                  {/* Split DNA into 5 rows of 44 characters each */}
                  {[0, 44, 88, 132, 176].map((rowStart, rowIndex) => (
                    <div key={rowIndex} className="dna-row">
                      {chars
                        .slice(rowStart, rowStart + 44)
                        .map((char, localIndex) => {
                          const globalIndex = rowStart + localIndex;
                          if (globalIndex >= chars.length) return null;
                          return (
                            <span
                              key={globalIndex}
                              className={getCharacterClasses(
                                globalIndex,
                                sectionIndex
                              )}
                              style={getCharacterStyle(
                                globalIndex,
                                sectionIndex
                              )}
                              onMouseEnter={() => handleMouseEnter(globalIndex)}
                              onMouseLeave={handleMouseLeave}
                              title={`${modelType.name} - Position ${globalIndex}: "${char}"`}
                            >
                              {char}
                            </span>
                          );
                        })}
                    </div>
                  ))}
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
      <p className="body-text">
        The key thing to keep in mind is the interaction between these as the
        sequence propagates. Imagine a stretch of genomic DNA containing both a
        gene and its regulatory elements. At the earliest stages, the Hyena-SE
        operator identifies short-range features like transcription factor
        binding motifs. Next, the Hyena-MR operator recognizes patterns that
        extend across larger regions, such as entire exons or introns. The
        Hyena-LI operator then brings in information from faraway regulatory
        elements—like distant enhancers—located thousands of bases away.
        Finally, the multi-head attention mechanism, enhanced with rotary
        positional embeddings, fuses these local and long-range signals to
        create a unified, detailed representation of the whole genomic region.
        <br />
        <br />
        Combining convolutions and attention layers in this manner provides a
        powerful way to capture both local and long-range dependencies in the
        DNA sequences at a low cost.
      </p>
    </>
  );
};

export default HyenaHighlight;
