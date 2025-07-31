import React, { useState, useEffect, useRef } from 'react';
import './ProgressScroll.css';

const ProgressScroll = () => {
  const [scrollProgress, setScrollProgress] = useState(0);
  const containerRef = useRef(null);

  // Grid configuration
  const rows = 6;
  const cols = 4;
  const totalNodes = rows * cols;
  const totalElements = totalNodes + rows * (cols - 1) + (rows - 1) * cols; // nodes + horizontal arrows + vertical arrows

  useEffect(() => {
    const handleScroll = () => {
      if (!containerRef.current) return;

      const container = containerRef.current;
      const rect = container.getBoundingClientRect();
      const containerHeight = container.offsetHeight;
      const windowHeight = window.innerHeight;

      // Calculate how far through the scroll we are (0 to 1)
      const scrolled = Math.max(0, -rect.top);
      const maxScroll = containerHeight - windowHeight;
      const progress = Math.min(1, Math.max(0, scrolled / maxScroll));

      setScrollProgress(progress);
    };

    window.addEventListener('scroll', handleScroll);
    handleScroll(); // Initial calculation

    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  // Calculate how many elements should be visible based on scroll
  const visibleElements = Math.floor(scrollProgress * totalElements);

  // Helper function to determine if an element should be visible
  const isElementVisible = elementIndex => {
    return elementIndex < visibleElements;
  };

  // Helper function to get the drawing order index for nodes and arrows
  const getDrawOrder = (type, row, col) => {
    let index = 0;

    // Count all elements that come before this one in drawing order
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        // Add the node
        if (r < row || (r === row && c < col)) {
          index++;
        } else if (r === row && c === col && type === 'node') {
          return index;
        }

        // Add horizontal arrow (if not last column)
        if (c < cols - 1) {
          if (
            r < row ||
            (r === row && c < col) ||
            (r === row && c === col && type === 'h-arrow')
          ) {
            index++;
          } else if (r === row && c === col && type === 'h-arrow') {
            return index;
          }
        }

        // Add vertical arrow (if not last row)
        if (r < rows - 1 && c === cols - 1) {
          if (r < row || (r === row && type === 'v-arrow' && c === col)) {
            index++;
          } else if (r === row && c === col && type === 'v-arrow') {
            return index;
          }
        }
      }
    }

    return index;
  };

  const gridSize = 80;
  const nodeRadius = 25;
  const spacing = 100;

  return (
    <>
      <h2 className="body-header">StripedHyena2</h2>
      <p className="body-text">
        Scroll to draw the grid progressively, left-to-right, top-to-bottom:
      </p>
      <section>
        <div className="progress-section-container" ref={containerRef}>
          <div className="progress-content-container">
            <div className="progress-step">
              <div className="progress-step-content">
                <h2>Drawing Progress: {(scrollProgress * 100).toFixed(1)}%</h2>
                <p>
                  Elements drawn: {visibleElements} / {totalElements}
                </p>
                <div className="progress-bar">
                  <div
                    className="progress-fill"
                    style={{
                      width: `${scrollProgress * 100}%`,
                      backgroundColor: `hsl(${250 + scrollProgress * 60}, 70%, 60%)`,
                    }}
                  />
                </div>
              </div>
            </div>

            <div className="progress-step">
              <div className="progress-step-content">
                <h3>Grid Status</h3>
                <p>
                  Nodes drawn: {Math.min(totalNodes, visibleElements)} /{' '}
                  {totalNodes}
                </p>
                <p>
                  Current row:{' '}
                  {Math.floor(
                    Math.min(totalNodes - 1, visibleElements) / cols
                  ) + 1}
                </p>
                <p>
                  Current column:{' '}
                  {(Math.min(totalNodes - 1, visibleElements) % cols) + 1}
                </p>
              </div>
            </div>
            <div className="progress-step">
              <div className="progress-step-content">
                <h3>Grid Status</h3>
                <p>
                  Nodes drawn: {Math.min(totalNodes, visibleElements)} /{' '}
                  {totalNodes}
                </p>
                <p>
                  Current row:{' '}
                  {Math.floor(
                    Math.min(totalNodes - 1, visibleElements) / cols
                  ) + 1}
                </p>
                <p>
                  Current column:{' '}
                  {(Math.min(totalNodes - 1, visibleElements) % cols) + 1}
                </p>
              </div>
            </div>
            <div className="progress-step">
              <div className="progress-step-content">
                <h3>Grid Status</h3>
                <p>
                  Nodes drawn: {Math.min(totalNodes, visibleElements)} /{' '}
                  {totalNodes}
                </p>
                <p>
                  Current row:{' '}
                  {Math.floor(
                    Math.min(totalNodes - 1, visibleElements) / cols
                  ) + 1}
                </p>
                <p>
                  Current column:{' '}
                  {(Math.min(totalNodes - 1, visibleElements) % cols) + 1}
                </p>
              </div>
            </div>
            <div className="progress-step">
              <div className="progress-step-content">
                <h3>Grid Status</h3>
                <p>
                  Nodes drawn: {Math.min(totalNodes, visibleElements)} /{' '}
                  {totalNodes}
                </p>
                <p>
                  Current row:{' '}
                  {Math.floor(
                    Math.min(totalNodes - 1, visibleElements) / cols
                  ) + 1}
                </p>
                <p>
                  Current column:{' '}
                  {(Math.min(totalNodes - 1, visibleElements) % cols) + 1}
                </p>
              </div>
            </div>
            <div className="progress-step">
              <div className="progress-step-content">
                <h3>Grid Status</h3>
                <p>
                  Nodes drawn: {Math.min(totalNodes, visibleElements)} /{' '}
                  {totalNodes}
                </p>
                <p>
                  Current row:{' '}
                  {Math.floor(
                    Math.min(totalNodes - 1, visibleElements) / cols
                  ) + 1}
                </p>
                <p>
                  Current column:{' '}
                  {(Math.min(totalNodes - 1, visibleElements) % cols) + 1}
                </p>
              </div>
            </div>

            <div className="progress-spacer" />
          </div>

          <div className="progress-charts-container">
            <div className="grid-container">
              <svg
                width={cols * spacing + 100}
                height={rows * spacing + 100}
                viewBox={`0 0 ${cols * spacing + 100} ${rows * spacing + 100}`}
              >
                {/* Draw the grid */}
                {Array.from({ length: rows }, (_, row) =>
                  Array.from({ length: cols }, (_, col) => {
                    const x = col * spacing + 50;
                    const y = row * spacing + 50;
                    const nodeOrder = getDrawOrder('node', row, col);
                    const isNodeVisible = isElementVisible(nodeOrder);
                    const isLastNode = row === 1 && col === 3; // Purple circle at row 2, col 4

                    return (
                      <g key={`cell-${row}-${col}`}>
                        {/* Horizontal arrow (except for last column) */}
                        {col < cols - 1 && (
                          <g
                            className={`arrow-group ${isElementVisible(getDrawOrder('h-arrow', row, col)) ? 'visible' : ''}`}
                            style={{
                              opacity: isElementVisible(
                                getDrawOrder('h-arrow', row, col)
                              )
                                ? 1
                                : 0,
                              transform: isElementVisible(
                                getDrawOrder('h-arrow', row, col)
                              )
                                ? 'scale(1)'
                                : 'scale(0.8)',
                            }}
                          >
                            <line
                              x1={x + nodeRadius}
                              y1={y}
                              x2={x + spacing - nodeRadius}
                              y2={y}
                              stroke="#9CA3AF"
                              strokeWidth="2"
                              markerEnd="url(#arrowhead)"
                            />
                          </g>
                        )}

                        {/* Vertical arrow (only for last column, not last row) */}
                        {col === cols - 1 && row < rows - 1 && (
                          <g
                            className={`arrow-group ${isElementVisible(getDrawOrder('v-arrow', row, col)) ? 'visible' : ''}`}
                            style={{
                              opacity: isElementVisible(
                                getDrawOrder('v-arrow', row, col)
                              )
                                ? 1
                                : 0,
                              transform: isElementVisible(
                                getDrawOrder('v-arrow', row, col)
                              )
                                ? 'scale(1)'
                                : 'scale(0.8)',
                            }}
                          >
                            <line
                              x1={x}
                              y1={y + nodeRadius}
                              x2={x}
                              y2={y + spacing - nodeRadius}
                              stroke="#9CA3AF"
                              strokeWidth="2"
                              markerEnd="url(#arrowhead)"
                            />
                          </g>
                        )}

                        {/* Node circle */}
                        <g
                          className={`node-group ${isNodeVisible ? 'visible' : ''}`}
                          style={{
                            opacity: isNodeVisible ? 1 : 0,
                            transform: isNodeVisible ? 'scale(1)' : 'scale(0)',
                          }}
                        >
                          <circle
                            cx={x}
                            cy={y}
                            r={nodeRadius}
                            fill={isLastNode ? '#A855F7' : '#E5E7EB'}
                            stroke={isLastNode ? '#9333EA' : '#D1D5DB'}
                            strokeWidth="2"
                          />
                          <text
                            x={x}
                            y={y - 35}
                            textAnchor="middle"
                            fontSize="16"
                            fill="#374151"
                            fontFamily="serif"
                            fontStyle="italic"
                          >
                            ℓ
                            <tspan fontSize="12" dy="-5">
                              {col}
                            </tspan>
                          </text>
                        </g>
                      </g>
                    );
                  })
                )}

                {/* Arrow marker definition */}
                <defs>
                  <marker
                    id="arrowhead"
                    markerWidth="10"
                    markerHeight="10"
                    refX="9"
                    refY="3"
                    orient="auto"
                  >
                    <polygon points="0 0, 10 3, 0 6" fill="#9CA3AF" />
                  </marker>
                </defs>
              </svg>
            </div>
          </div>
        </div>
        <br />
        <br />
        <p className="body-text">The grid drawing is complete!</p>
      </section>
    </>
  );
};

export default ProgressScroll;
