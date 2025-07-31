import React from 'react';

const StripedHyenaBlocks = () => {
  const width = 750;
  // More compact bars
  const barWidth = 8;
  const barSpacing = 2;
  const barHeight = 16;

  // Mapping for patterns
  const patternMap = {
    S: { text: 'SE', color: '#c8ff8c' },
    D: { text: 'MR', color: '#e4c1f9' },
    H: { text: 'LI', color: '#f6e3a3' },
    '*': { text: 'MHA', color: '#a8dadc' },
  };

  // Pattern data for each model
  const modelPatterns = {
    '1B': 'SDH*SDHSDH*SDHSDH*SDHSDH*',
    '7B': 'SDH*SDHSDH*SDHSDH*SDHSDH*SDHSDH*',
    '40B': 'SDH*SDHSDH*SDHSDH*SDHSDH*SDHSDH*SDH*SDHSDH*SDHSDH*',
  };

  const getDataFromPattern = pattern =>
    pattern.split('').map((char, i) => ({
      ...patternMap[char],
      id: `${char}-${i}`,
      index: i,
    }));

  return (
    <div
      style={{
        margin: '0 auto',
        width: `${width}px`,
        fontFamily: 'var(--font-main)',
        fontSize: 'var(--font-size-sm)',
      }}
    >
      {/* Legend */}
      <div
        style={{
          display: 'flex',
          justifyContent: 'center',
          gap: '12px',
          marginBottom: '8px',
        }}
      >
        {Object.values(patternMap).map(({ text, color }) => (
          <div
            key={text}
            style={{ display: 'flex', alignItems: 'center', gap: '4px' }}
          >
            <div
              style={{
                width: '10px',
                height: '10px',
                backgroundColor: color,
                border: '1px solid #333',
                borderRadius: '2px',
              }}
            />
            <span>{text}</span>
          </div>
        ))}
      </div>

      <table style={{ width: '100%', borderCollapse: 'collapse' }}>
        <thead>
          <tr>
            <th
              style={{ padding: '4px 6px', fontWeight: 600, textAlign: 'left' }}
            >
              Spec
            </th>
            <th
              style={{ padding: '4px 6px', fontWeight: 600, textAlign: 'left' }}
            >
              Num Layers
            </th>
            <th
              style={{ padding: '4px 6px', fontWeight: 600, textAlign: 'left' }}
            >
              Block Layout
            </th>
          </tr>
        </thead>
        <tbody>
          {Object.entries(modelPatterns).map(([model, pattern]) => {
            const data = getDataFromPattern(pattern);
            const layers = data.length;
            const totalBarsWidth =
              layers * barWidth + (layers - 1) * barSpacing;

            return (
              <tr key={model}>
                <td style={{ padding: '4px 6px' }}>{model}</td>
                <td style={{ padding: '4px 6px' }}>{layers}</td>
                <td style={{ padding: '4px 6px' }}>
                  <svg
                    width="100%"
                    height={barHeight}
                    viewBox={`0 0 ${totalBarsWidth} ${barHeight}`}
                    preserveAspectRatio="xMinYMid meet"
                  >
                    <g>
                      {data.map(bar => (
                        <rect
                          key={`${model}-${bar.id}`}
                          x={bar.index * (barWidth + barSpacing)}
                          y={0}
                          width={barWidth}
                          height={barHeight}
                          fill={bar.color}
                          stroke="#333"
                          strokeWidth={1}
                          rx={1}
                        />
                      ))}
                    </g>
                  </svg>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
};

export default StripedHyenaBlocks;
