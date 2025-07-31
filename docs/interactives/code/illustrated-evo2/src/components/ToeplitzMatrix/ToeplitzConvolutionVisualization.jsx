import React, { useState, useRef } from 'react';

const ToeplitzConvolutionVisualization = () => {
  const [inputSize, setInputSize] = useState(5);
  const [kernelSize, setKernelSize] = useState(3);
  const [padding, setPadding] = useState(2);
  const [hoveredInputIndex, setHoveredInputIndex] = useState(null);
  const [hoveredOutputIndex, setHoveredOutputIndex] = useState(null);
  const [kernelType, setKernelType] = useState('uniform');
  const [convolutionType, setConvolutionType] = useState('non-causal');

  const svgRef = useRef();
  const width = 800,
    height = 300;
  const MARGIN = { top: 60, right: 20, bottom: 20, left: 20 };

  // --- only pad left when causal, both sides otherwise
  const paddedInputSize =
    convolutionType === 'causal'
      ? inputSize + padding
      : inputSize + 2 * padding;

  const outputSize = paddedInputSize - kernelSize + 1;
  const cellSize = Math.min(30, (width - 200) / (paddedInputSize * 3));
  const cellPadding = 1;
  const elementSpacing = 40;

  const totalWidth =
    paddedInputSize * cellSize + // horizontal padded input
    paddedInputSize * cellSize + // matrix width
    cellSize + // spacing for ×
    outputSize * cellSize + // vertical output
    elementSpacing * 3;
  const startX = (width - totalWidth) / 2;

  const paddedInputVectorX = startX;
  const toeplitzMatrixX =
    paddedInputVectorX + paddedInputSize * cellSize + elementSpacing;
  const inputVectorVerticalX =
    toeplitzMatrixX + paddedInputSize * cellSize + elementSpacing;
  const outputVectorX = inputVectorVerticalX + cellSize + elementSpacing;

  const vectorsY = MARGIN.top + 30;
  const matrixY = vectorsY;

  // raw (unpadded) input
  const inputVector = Array(inputSize)
    .fill(null)
    .map((_, i) => ({ value: i + 1, index: i }));

  // --- build padded input vector based on convolutionType
  const paddedInputVector =
    convolutionType === 'causal'
      ? [...Array(padding).fill({ value: 0, isPadding: true }), ...inputVector]
      : [
          ...Array(padding).fill({ value: 0, isPadding: true }),
          ...inputVector,
          ...Array(padding).fill({ value: 0, isPadding: true }),
        ];

  // integer-only kernels
  const getKernelValues = () => {
    if (kernelType === 'uniform') {
      return Array(kernelSize).fill(1);
    } else {
      if (kernelSize === 1) return [1];
      if (kernelSize === 2) return [1, -1];
      if (kernelSize === 3) return [1, 2, 1];
      if (kernelSize === 4) return [1, 3, 3, 1];
      if (kernelSize === 5) return [1, 2, 4, 2, 1];
      return Array(kernelSize).fill(1);
    }
  };

  const kernelValues = getKernelValues();
  const kernelVector = kernelValues.map((v, i) => ({ value: v, index: i }));

  const createToeplitzMatrix = () => {
    const matrix = [];
    if (convolutionType === 'causal') {
      // For causal: kernel appears in reverse order, shifted left in each row
      const flipped = [...kernelVector].reverse();
      for (let i = 0; i < outputSize; i++) {
        const row = Array(paddedInputSize).fill(0);
        // Place the flipped kernel starting from position i
        for (let j = 0; j < kernelSize; j++) {
          const col = i + j;
          if (col < paddedInputSize) {
            row[col] = flipped[j].value;
          }
        }
        matrix.push(row);
      }
    } else {
      // For non-causal: kernel appears in normal order, shifted right in each row
      for (let i = 0; i < outputSize; i++) {
        const row = Array(paddedInputSize).fill(0);
        for (let j = 0; j < kernelSize; j++) {
          if (i + j < paddedInputSize) {
            row[i + j] = kernelVector[j].value;
          }
        }
        matrix.push(row);
      }
    }
    return matrix;
  };

  const toeplitzMatrix = createToeplitzMatrix();

  const calculateConvolution = () =>
    toeplitzMatrix.map((row, idx) => ({
      value: row.reduce((sum, v, j) => sum + v * paddedInputVector[j].value, 0),
      index: idx,
    }));
  const outputVector = calculateConvolution();

  const getHighlightedCells = () => {
    const base = {
      inputs: [],
      outputs: [],
      matrixCells: [],
      kernelPositions: [],
    };

    // Default highlighting: highlight the first convolution if nothing is hovered
    const shouldShowDefault =
      hoveredInputIndex === null && hoveredOutputIndex === null;
    const effectiveOutputIndex = shouldShowDefault ? 0 : hoveredOutputIndex;
    const effectiveInputIndex = shouldShowDefault ? null : hoveredInputIndex;

    if (effectiveInputIndex !== null) {
      if (convolutionType === 'causal') {
        // For causal: find ALL output rows that use this input
        for (let outputRow = 0; outputRow < outputSize; outputRow++) {
          // Check if this input is used in this output row
          const inputStartCol = outputRow;
          const inputEndCol = outputRow + kernelSize - 1;

          if (
            effectiveInputIndex >= inputStartCol &&
            effectiveInputIndex <= inputEndCol
          ) {
            base.outputs.push(outputRow);
            // Highlight all inputs used in this convolution
            for (let k = 0; k < kernelSize; k++) {
              const inputIdx = outputRow + k;
              if (inputIdx >= 0 && inputIdx < paddedInputSize) {
                base.inputs.push(inputIdx);
                base.matrixCells.push({ row: outputRow, col: inputIdx });
              }
            }
            // For clarity, typically we want to show the first valid output that uses this input
            break;
          }
        }
      } else {
        // For non-causal: find which output has this input at the CENTER of its kernel
        // For odd kernel sizes, center is at floor(kernelSize/2)
        // For even kernel sizes, we'll use floor(kernelSize/2) as center
        const kernelCenter = Math.floor(kernelSize / 2);
        const outputRow = effectiveInputIndex - kernelCenter;

        if (outputRow >= 0 && outputRow < outputSize) {
          base.outputs.push(outputRow);
          // Highlight all cells in this row that have non-zero values
          for (let c = 0; c < paddedInputSize; c++) {
            if (toeplitzMatrix[outputRow][c] !== 0) {
              base.inputs.push(c);
              base.matrixCells.push({ row: outputRow, col: c });
            }
          }
        }
      }
      return base;
    }

    if (effectiveOutputIndex !== null) {
      base.outputs.push(effectiveOutputIndex);
      for (let c = 0; c < paddedInputSize; c++) {
        if (toeplitzMatrix[effectiveOutputIndex][c] !== 0) {
          base.inputs.push(c);
          base.matrixCells.push({ row: effectiveOutputIndex, col: c });
        }
      }
      return base;
    }

    return base;
  };
  const highlighted = getHighlightedCells();

  const renderVector = (
    vector,
    x,
    y,
    vertical = false,
    padded = false,
    hoverFn = null,
    secondPadded = false
  ) =>
    vector.map((item, i) => {
      const cx = vertical ? x : x + i * cellSize;
      const cy = vertical ? y + i * cellSize : y;
      let fill = '#fff',
        sw = 1,
        stroke = '#333';

      if (padded && item.isPadding) fill = 'var(--gray-200)';
      if (
        !vertical &&
        padded &&
        (hoveredInputIndex === i || highlighted.inputs.includes(i))
      ) {
        fill = item.isPadding ? 'var(--pippin)' : 'var(--jonquil)';
        sw = 2;
      } else if (
        vertical &&
        !padded &&
        (hoveredOutputIndex === i || highlighted.outputs.includes(i))
      ) {
        fill = 'var(--jonquil,#76B900)';
        sw = 2;
      } else if (vertical && secondPadded && highlighted.inputs.includes(i)) {
        fill = item.isPadding
          ? 'var(--pippin,#FFD23F)'
          : 'var(--jonquil,#76B900)';
        sw = 2;
      }

      return (
        <g key={i}>
          <rect
            x={cx}
            y={cy}
            width={cellSize - cellPadding}
            height={cellSize - cellPadding}
            fill={fill}
            stroke={stroke}
            strokeWidth={sw}
            onMouseEnter={() => hoverFn && hoverFn(i)}
            onMouseLeave={() => hoverFn && hoverFn(null)}
            className={hoverFn ? 'tcv-cell-hoverable' : ''}
          />
          <text
            x={cx + (cellSize - cellPadding) / 2}
            y={cy + (cellSize - cellPadding) / 2}
            textAnchor="middle"
            dominantBaseline="middle"
            className="tcv-cell-text"
          >
            {item.value}
          </text>
        </g>
      );
    });

  const renderToeplitzMatrix = () => {
    const cells = [];
    for (let r = 0; r < outputSize; r++) {
      for (let c = 0; c < paddedInputSize; c++) {
        const x = toeplitzMatrixX + c * cellSize;
        const y = matrixY + r * cellSize;
        const v = toeplitzMatrix[r][c];
        const zero = v === 0;
        const hl = highlighted.matrixCells.some(
          cell => cell.row === r && cell.col === c
        );

        cells.push(
          <g key={`${r}-${c}`}>
            <rect
              x={x}
              y={y}
              width={cellSize - cellPadding}
              height={cellSize - cellPadding}
              fill={hl ? 'var(--jonquil,#76B900)' : zero ? '#f8f9fa' : '#fff'}
              stroke="#333"
              strokeWidth={hl ? 2 : 1}
            />
            {v !== 0 && (
              <text
                x={x + (cellSize - cellPadding) / 2}
                y={y + (cellSize - cellPadding) / 2}
                textAnchor="middle"
                dominantBaseline="middle"
                className="tcv-cell-text"
              >
                {v}
              </text>
            )}
          </g>
        );
      }
    }
    return cells;
  };

  const Slider = ({ label, value, setValue, min, max, step = 1 }) => {
    const id = `slider-${label.replace(/\s+/g, '-')}`;
    return (
      <div className="tcv-slider">
        <div className="tcv-slider-header">
          <label htmlFor={id} className="tcv-slider-label">
            {label}:
          </label>
          <input
            type="text"
            className="tcv-slider-input"
            value={value}
            onChange={e => {
              const v = parseInt(e.target.value) || min;
              setValue(Math.max(min, Math.min(max, v)));
            }}
          />
        </div>
        <input
          id={id}
          type="range"
          className="tcv-range-input"
          min={min}
          max={max}
          step={step}
          value={value}
          onChange={e => setValue(parseInt(e.target.value))}
        />
      </div>
    );
  };

  return (
    <>
      <style>{`
        /* Container & Panels */
        .tcv-container {
          max-width: 800px;
          margin: 0 auto;
          font-family: Nvidia Sans, -apple-system, BlinkMacSystemFont, "Segoe UI",
            sans-serif;
        }
        .tcv-panel {
          background-color: #fff;
          border-radius: 8px;
          margin-bottom: 20px;
        }
        .tcv-svg {
          display: block;
        }

        /* Titles & Operators */
        .tcv-title {
          text-anchor: middle;
          font-size: 14px;
          font-weight: 600;
          fill: #333;
        }
        .tcv-operator {
          text-anchor: middle;
          font-size: 20px;
          fill: #333;
        }

        /* Cell styling */
        .tcv-cell-hoverable {
          cursor: pointer;
        }
        .tcv-cell-text {
          font-size: 8px;
          font-weight: 400;
          pointer-events: none;
        }

        /* Controls area */
        .tcv-controls {
          background-color: #f8f9fa;
          padding: 20px;
          border-radius: 8px;
        }
        .tcv-sliders {
          display: flex;
          gap: 20px;
          margin-bottom: 15px;
          flex-wrap: wrap;
        }
        .tcv-controls-section {
          display: flex;
          gap: 10px;
          align-items: center;
          border-top: 1px solid #ddd;
          padding-top: 15px;
          margin-top: 15px;
        }

        /* Slider component */
        .tcv-slider {
          flex: 1;
          min-width: 200px;
        }
        .tcv-slider-header {
          display: flex;
          align-items: center;
          margin-bottom: 8px;
        }
        .tcv-slider-label {
          font-size: 14px;
          font-weight: 500;
          color: #333;
          margin-right: 10px;
          width: 80px;
        }
        .tcv-slider-input {
          width: 40px;
          padding: 4px;
          border: 1px solid #ddd;
          border-radius: 4px;
          font-size: 14px;
          text-align: center;
        }
        /* Range slider styling */
        .tcv-range-input {
          width: 100%;
          height: 6px;
          cursor: pointer;
          background: transparent;
          outline: none;
          -webkit-appearance: none;
          appearance: none;
        }
        .tcv-range-input::-webkit-slider-thumb {
          -webkit-appearance: none;
          appearance: none;
          width: 16px;
          height: 16px;
          background: var(--nvgreen, #76b900);
          cursor: pointer;
          border-radius: 2px;
          margin-top: -5px;
        }
        .tcv-range-input::-moz-range-thumb {
          width: 16px;
          height: 16px;
          background: var(--nvgreen, #76b900);
          cursor: pointer;
          border: none;
          border-radius: 2px;
        }
        .tcv-range-input::-webkit-slider-runnable-track {
          background: var(--gray-200, #e5e5e5);
          height: 6px;
          border-radius: 3px;
        }
        .tcv-range-input::-moz-range-track {
          background: var(--gray-200, #e5e5e5);
          height: 6px;
          border-radius: 3px;
        }

        /* Buttons */
        .tcv-button {
          padding: 6px 16px;
          border: 1px solid #ddd;
          border-radius: 4px;
          background-color: #fff;
          color: #333;
          cursor: pointer;
          font-size: 14px;
          font-weight: 500;
        }
        .tcv-button.active {
          background-color: var(--nvgreen, #76b900);
          color: #fff;
        }

        /* Labels & Info */
        .tcv-label {
          font-size: 14px;
          font-weight: 500;
        }
        .tcv-info {
          margin-left: auto;
          font-size: 13px;
          color: #666;
          display: flex;
        }
        .tcv-info-separator {
          margin-left: 15px;
        }
      `}</style>

      <div className="tcv-container">
        <div className="tcv-panel">
          <svg ref={svgRef} width={width} height={height} className="tcv-svg">
            {/* Padded Input */}
            <g>
              <text
                x={paddedInputVectorX + (paddedInputSize * cellSize) / 2}
                y={vectorsY - 10}
                className="tcv-title"
              >
                Padded Input ({paddedInputSize})
              </text>
              {renderVector(
                paddedInputVector,
                paddedInputVectorX,
                vectorsY,
                false,
                true,
                setHoveredInputIndex
              )}
            </g>

            {/* Matrix */}
            <g>
              <text
                x={toeplitzMatrixX + (paddedInputSize * cellSize) / 2}
                y={matrixY - 10}
                className="tcv-title"
              >
                Toeplitz Matrix
              </text>
              {renderToeplitzMatrix()}
            </g>

            {/* × */}
            <text
              x={
                toeplitzMatrixX +
                paddedInputSize * cellSize +
                elementSpacing / 2
              }
              y={matrixY + (outputSize * cellSize) / 2}
              className="tcv-operator"
            >
              ×
            </text>

            {/* Vertical Input */}
            <g>
              <text
                x={inputVectorVerticalX + cellSize / 2}
                y={matrixY - 10}
                className="tcv-title"
              >
                Input
              </text>
              {renderVector(
                paddedInputVector,
                inputVectorVerticalX,
                matrixY,
                true,
                true,
                null,
                true
              )}
            </g>

            {/* = */}
            <text
              x={inputVectorVerticalX + cellSize + elementSpacing / 2}
              y={matrixY + (outputSize * cellSize) / 2}
              className="tcv-operator"
            >
              =
            </text>

            {/* Output */}
            <g>
              <text
                x={outputVectorX + cellSize / 2}
                y={matrixY - 10}
                className="tcv-title"
              >
                Output
              </text>
              {renderVector(
                outputVector,
                outputVectorX,
                matrixY,
                true,
                false,
                setHoveredOutputIndex
              )}
            </g>
          </svg>
        </div>

        <div className="tcv-controls">
          <div className="tcv-sliders">
            <Slider
              label="Input (L)"
              value={inputSize}
              setValue={setInputSize}
              min={3}
              max={7}
            />
            <Slider
              label="Kernel (h)"
              value={kernelSize}
              setValue={setKernelSize}
              min={1}
              max={4}
            />
            <Slider
              label="Padding (P)"
              value={padding}
              setValue={setPadding}
              min={0}
              max={kernelSize}
            />
          </div>
          <div className="tcv-controls-section">
            <span className="tcv-label">Kernel Type:</span>
            {['uniform', 'varied'].map(type => (
              <button
                key={type}
                className={`tcv-button${kernelType === type ? ' active' : ''}`}
                onClick={() => setKernelType(type)}
              >
                {type.charAt(0).toUpperCase() + type.slice(1)}
              </button>
            ))}
            <span className="tcv-label">Type:</span>
            {['non-causal', 'causal'].map(type => (
              <button
                key={type}
                className={`tcv-button${convolutionType === type ? ' active' : ''}`}
                onClick={() => setConvolutionType(type)}
              >
                {type === 'non-causal' ? 'Non‑causal' : 'Causal'}
              </button>
            ))}
            <div className="tcv-info">
              <span>Kernel: [{kernelVector.map(k => k.value).join(', ')}]</span>
              <span className="tcv-info-separator">
                Output size: {outputSize}
              </span>
            </div>
          </div>
        </div>
      </div>
    </>
  );
};

export default ToeplitzConvolutionVisualization;
