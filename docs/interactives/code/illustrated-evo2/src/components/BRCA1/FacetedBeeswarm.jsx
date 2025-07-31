import React, { useMemo, useState, useEffect, useRef } from 'react';
import * as d3 from 'd3';
import { beeswarmData } from './beeswarmData';

// Component for animated number display
const AnimatedNumber = ({ value, format = d => d }) => {
  const [displayValue, setDisplayValue] = useState(value);
  const animationRef = useRef(null);

  useEffect(() => {
    if (animationRef.current) animationRef.current.stop();

    const interpolate = d3.interpolateNumber(displayValue, value);
    const duration = 50; // Nearly instant

    const timer = d3.timer(elapsed => {
      const t = Math.min(elapsed / duration, 1);
      setDisplayValue(interpolate(t));
      if (t >= 1) {
        timer.stop();
        animationRef.current = null;
      }
    });

    animationRef.current = timer;
    return () => {
      if (animationRef.current) animationRef.current.stop();
    };
  }, [value]);

  return <>{format(displayValue)}</>;
};

// Barcode chart component
const BarcodeChart = ({
  width = 700,
  height = 80,
  margin = { top: 20, right: 10, bottom: 30, left: 80 },
  hoveredNode = null,
  data = [],
}) => {
  const innerWidth = width - margin.left - margin.right;
  const innerHeight = height - margin.top - margin.bottom;
  const geneStart = 41196312;
  const windowSize = 79922;
  const barHeight = 20;

  const xScale = d3
    .scaleLinear()
    .domain([0, windowSize])
    .range([0, innerWidth]);

  const xTicks = [0, windowSize];
  const formatTick = d => d3.format(',')(d + geneStart);

  const scaledPosition = useMemo(() => {
    if (!hoveredNode) return null;
    const offset = hoveredNode.pos - geneStart;
    return Math.max(0, Math.min(windowSize, offset));
  }, [hoveredNode]);

  const axisOpacity = hoveredNode ? 0 : 1;

  return (
    <svg width={width} height={height} style={{ border: '0px solid red' }}>
      <g transform={`translate(${margin.left},${margin.top})`}>
        {/* Background bar */}
        <rect
          x={0}
          y={innerHeight / 2 - 5}
          width={innerWidth}
          height={barHeight}
          fill="var(--gray-200)"
          stroke="#ccc"
          strokeWidth={1}
        />

        {/* All ticks */}
        {data.map((d, i) => {
          const rawOffset = d.pos - geneStart;
          const clamped = Math.max(0, Math.min(windowSize, rawOffset));
          const x = xScale(clamped);
          const isHighlighted =
            hoveredNode &&
            d.pos === hoveredNode.pos &&
            d.ref === hoveredNode.ref &&
            d.alt === hoveredNode.alt;

          return (
            <line
              key={`${d.pos}-${i}`}
              x1={x}
              x2={x}
              opacity="0.2"
              y1={innerHeight / 2 - barHeight / 2 + 5}
              y2={innerHeight / 2 + barHeight / 2 + 5}
              stroke={data.class !== 'LOF' ? 'black' : 'var(--jonquil)'}
              strokeOpacity={isHighlighted ? 1 : 0.2}
              strokeWidth={isHighlighted ? 2 : 1}
            />
          );
        })}

        {/* X-axis */}
        <g transform={`translate(0,${innerHeight})`} opacity={axisOpacity}>
          {/* <line x1={0} x2={innerWidth} y1={0} y2={0} stroke="black" /> */}
          {xTicks.map(tick => (
            <g key={tick} transform={`translate(${xScale(tick)},0)`}>
              {/* <line y2={6} stroke="black" /> */}
              <text
                y={0}
                dy="0.71em"
                textAnchor="middle"
                fontSize="8"
                fill="black"
              >
                {formatTick(tick)}
              </text>
            </g>
          ))}
        </g>

        {/* Y-axis label */}
        <text
          x={0}
          y={5}
          dy="0.32em"
          textAnchor="start"
          fontSize="10"
          fill="black"
          opacity={axisOpacity}
        >
          Chromosome 17: SNV positions
        </text>

        {/* Hovered position marker */}
        {scaledPosition !== null && hoveredNode && (
          <g>
            <line
              x1={xScale(scaledPosition)}
              x2={xScale(scaledPosition)}
              y1={8}
              y2={innerHeight - 6}
              stroke="black"
              strokeWidth={2}
            />
            <text
              x={xScale(scaledPosition)}
              y={6}
              textAnchor="middle"
              fontSize="11"
              fill="black"
              fontWeight="600"
            >
              {hoveredNode.ref} → {hoveredNode.alt}
            </text>
            <text
              x={xScale(scaledPosition)}
              y={innerHeight + 6}
              textAnchor="middle"
              fontSize="9"
              fill="black"
              fontWeight="600"
            >
              {formatTick(Math.round(scaledPosition))}
            </text>
          </g>
        )}
      </g>
    </svg>
  );
};

const FacetedBeeswarm = ({
  data = [],
  width = 600,
  height = 20,
  margin = { top: 0, right: 50, bottom: 50, left: 40 },
  selectedModel = '1B',
}) => {
  const [hoveredNode, setHoveredNode] = useState(null);
  const [animatedNodes, setAnimatedNodes] = useState([]);
  const [animatedMedians, setAnimatedMedians] = useState({});
  const animationRef = useRef(null);
  const medianAnimationRef = useRef(null);

  const innerWidth = width - margin.left - margin.right;
  const innerHeight = height - margin.top - margin.bottom;
  const categories = ['LOF', 'FUNC/INT'];

  const xScale = d3
    .scaleLinear()
    .domain([-0.016, 0.004])
    .range([0, innerWidth]);

  const yScale = d3
    .scaleBand()
    .domain(categories)
    .range([innerHeight, 0])
    .padding(0.3);

  const rectWidth = 11;
  const rectHeight = 11;

  const categoryStats = useMemo(() => {
    return categories.map(category => {
      const categoryData = data.filter(d => d.class === category);
      return {
        category,
        median: d3.median(categoryData, d => d.evo2_delta_score),
      };
    });
  }, [data, categories]);

  // Animate median transitions
  useEffect(() => {
    if (!categoryStats.length) return;
    if (medianAnimationRef.current) medianAnimationRef.current.stop();

    const newMedians = {};
    categoryStats.forEach(({ category, median }) => {
      if (median !== undefined) newMedians[category] = median;
    });

    if (Object.keys(animatedMedians).length) {
      const interpolators = {};
      categories.forEach(cat => {
        if (
          animatedMedians[cat] !== undefined &&
          newMedians[cat] !== undefined
        ) {
          interpolators[cat] = d3.interpolateNumber(
            animatedMedians[cat],
            newMedians[cat]
          );
        }
      });

      const timer = d3.timer(elapsed => {
        const t = Math.min(elapsed / 100, 1); // Very fast - 100ms
        const current = {};
        categories.forEach(cat => {
          current[cat] = interpolators[cat]
            ? interpolators[cat](t)
            : newMedians[cat];
        });
        setAnimatedMedians(current);
        if (t >= 1) {
          timer.stop();
          medianAnimationRef.current = null;
        }
      });

      medianAnimationRef.current = timer;
    } else {
      setAnimatedMedians(newMedians);
    }

    return () => {
      if (medianAnimationRef.current) medianAnimationRef.current.stop();
    };
  }, [categoryStats, selectedModel]);

  // Force simulation for nodes — only run when data changes
  useEffect(() => {
    if (!data.length) return;
    if (animationRef.current) animationRef.current.stop();

    const nodeData = data.map((d, i) => ({
      ...d,
      id: `${d.class}-${i}`,
      radius: rectWidth / 2 + 2,
      targetX: xScale(d.evo2_delta_score),
      targetY: yScale(d.class) + yScale.bandwidth() / 2,
      color: d.class === 'LOF' ? 'var(--pippin)' : 'var(--jonquil)',
      stroke: d.class === 'LOF' ? 'var(--hot-cinnamon)' : 'var(--nvgreen)',
    }));

    // Preserve old positions
    const oldMap = new Map(animatedNodes.map(n => [n.id, n]));
    nodeData.forEach(n => {
      const old = oldMap.get(n.id);
      n.x = old?.x ?? n.targetX;
      n.y = old?.y ?? n.targetY;
    });

    const sim = d3
      .forceSimulation(nodeData)
      .force('x', d3.forceX(d => d.targetX).strength(1))
      .force('y', d3.forceY(d => d.targetY).strength(1))
      .force('collide', d3.forceCollide(d => d.radius + 2).strength(1))
      .velocityDecay(0.8)
      .alpha(0.75)
      .alphaTarget(0)
      .alphaDecay(0.08);

    animationRef.current = sim;

    let tickCount = 0;
    const maxTicks = 300; // Just 20 ticks

    sim.on('tick', () => {
      tickCount += 1;
      setAnimatedNodes(nodeData.map(d => ({ ...d })));

      if (tickCount >= maxTicks) {
        sim.stop();
        // Snap to final positions
        nodeData.forEach(node => {
          node.x = node.targetX;
          node.y = node.targetY;
        });
        setAnimatedNodes(nodeData.map(d => ({ ...d })));
        animationRef.current = null;
      }
    });

    return () => {
      if (animationRef.current) animationRef.current.stop();
    };
  }, [data]); // Only depend on data changes, not scale functions

  // Voronoi overlay for better hover regions
  const voronoi = useMemo(() => {
    if (!animatedNodes.length) return null;
    const points = animatedNodes.map(n => [n.x, n.y]);
    const delaunay = d3.Delaunay.from(points);
    return delaunay.voronoi([0, 0, innerWidth, innerHeight]);
  }, [animatedNodes, innerWidth, innerHeight]);

  const xTicks = xScale.ticks();
  const formatX = d3.format('.3f');

  return (
    <div
      style={{
        position: 'relative',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
      }}
    >
      {/* Barcode chart directly above beeswarm */}
      <BarcodeChart
        width={width}
        // height={80}
        margin={{ top: 20, right: 50, bottom: 15, left: 40 }}
        hoveredNode={hoveredNode}
        data={animatedNodes}
      />

      {/* Beeswarm + Voronoi with no top gap */}
      <svg width={width} height={height}>
        <g transform={`translate(${margin.left},${margin.top})`}>
          {/* Grid lines */}
          {xTicks.map(tick => (
            <line
              key={tick}
              x1={xScale(tick)}
              x2={xScale(tick)}
              y1={0}
              y2={innerHeight}
              stroke="#e0e0e0"
              opacity={0.8}
              strokeWidth={1}
            />
          ))}

          {/* X-axis */}
          <g transform={`translate(0,${innerHeight})`}>
            {xTicks.map(tick => (
              <g key={tick} transform={`translate(${xScale(tick)},0)`}>
                <text y={9} dy="0.71em" textAnchor="middle" fontSize="10">
                  {formatX(tick)}
                </text>
              </g>
            ))}
            <text
              x={innerWidth / 2}
              y={35}
              textAnchor="middle"
              fontSize="12"
              fill="black"
            >
              Delta Likelihood Score, Evo 2
            </text>
          </g>

          {/* Y-axis labels */}
          <g>
            {categories.map(cat => (
              <g
                key={cat}
                transform={`translate(0,${
                  yScale(cat) + yScale.bandwidth() / 2
                })`}
              >
                <text
                  x={-9}
                  dy="0.32em"
                  textAnchor="end"
                  fontSize="10"
                  fill="black"
                >
                  {cat}
                </text>
              </g>
            ))}
          </g>

          {/* Voronoi cells */}
          {voronoi &&
            animatedNodes.map((node, i) => (
              <path
                key={node.id}
                d={voronoi.renderCell(i)}
                fill="none"
                stroke="rgba(0,0,0,0)"
                pointerEvents="all"
                onMouseEnter={() => setHoveredNode(node)}
                onMouseLeave={() => setHoveredNode(null)}
              />
            ))}

          {/* Data rectangles */}
          {animatedNodes.map(node => (
            <g key={node.id}>
              <rect
                x={node.x - rectWidth / 2}
                y={node.y - rectHeight / 2}
                width={rectWidth}
                height={rectHeight}
                fill={node.color}
                stroke={hoveredNode?.id === node.id ? 'black' : node.stroke}
                strokeWidth={hoveredNode?.id === node.id ? 2 : 1.5}
                rx={2}
                ry={2}
                style={{ cursor: 'pointer' }}
                onMouseEnter={() => setHoveredNode(node)}
                onMouseLeave={() => setHoveredNode(null)}
              />
              <text
                x={node.x}
                y={node.y + 1}
                textAnchor="middle"
                dominantBaseline="middle"
                fontSize="7"
                fontWeight="400"
                fill="black"
                style={{ pointerEvents: 'none' }}
              >
                {node.ref}
              </text>
            </g>
          ))}

          {/* Median bars & labels */}
          {categories.map(cat => {
            const median = animatedMedians[cat];
            if (median === undefined) return null;
            const yPos = yScale(cat) + yScale.bandwidth() / 2;
            const xPos = xScale(median);
            const medianValue = median;
            return (
              <g key={cat}>
                <line
                  x1={xPos}
                  x2={xPos}
                  y1={yPos - 55}
                  y2={yPos + 55}
                  stroke="white"
                  strokeWidth={8}
                />
                <line
                  x1={xPos}
                  x2={xPos}
                  y1={yPos - 55}
                  y2={yPos + 55}
                  stroke="#2c3e50"
                  strokeWidth={6}
                />
                <text
                  x={xPos}
                  y={yPos - 60}
                  textAnchor="middle"
                  fontSize="14"
                  fontWeight="600"
                  fill="black"
                  stroke="white"
                  strokeWidth="3"
                  strokeLinejoin="round"
                  paintOrder="stroke fill"
                >
                  {cat !== 'LOF' ? 'Median: ' : ''}
                  <AnimatedNumber
                    value={medianValue}
                    format={v => `${v.toFixed(5)}`}
                  />
                </text>
              </g>
            );
          })}
        </g>
      </svg>

      {/* Smaller tooltip */}
      {hoveredNode && (
        <div
          style={{
            position: 'absolute',
            left: margin.left + hoveredNode.x + 10,
            top: margin.top + hoveredNode.y + 90,
            padding: '6px 8px',
            background: 'rgba(0, 0, 0, 0.8)',
            color: 'white',
            borderRadius: '4px',
            pointerEvents: 'none',
            fontSize: '11px',
            lineHeight: '1.2',
            zIndex: 10,
          }}
        >
          <div>
            <strong>Position:</strong> chr{hoveredNode.chrom}:{hoveredNode.pos}
          </div>
          <div>
            <strong>Ref:</strong> {hoveredNode.ref}
          </div>
          <div>
            <strong>Alt:</strong> {hoveredNode.alt}
          </div>
          <div>
            <strong>Ref Log Prob:</strong>{' '}
            {hoveredNode.ref_log_probs.toFixed(3)}
          </div>
          <div>
            <strong>Var Log Prob:</strong>{' '}
            {hoveredNode.var_log_probs.toFixed(3)}
          </div>
          <div>
            <strong>Delta:</strong> {hoveredNode.evo2_delta_score.toFixed(3)}
          </div>
        </div>
      )}
    </div>
  );
};

const FacetedBeeswarmExample = () => {
  const [selectedModel, setSelectedModel] = useState('1b');
  const data = useMemo(() => {
    const key = selectedModel.toLowerCase();
    return (beeswarmData[key] || []).map(d => ({
      ...d,
      class: d.class || 'FUNC/INT',
      chrom: d.chrom || 17,
      model: key.toUpperCase(),
    }));
  }, [selectedModel]);

  return (
    <div
      style={{
        margin: '2rem auto',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
      }}
    >
      {/* Model selection */}
      <div style={{ display: 'flex', gap: '10px' }}>
        <button
          onClick={() => setSelectedModel('1b')}
          className={`model-button ${selectedModel === '1b' ? 'active' : ''}`}
        >
          1B Model
        </button>
        <button
          onClick={() => setSelectedModel('7b')}
          className={`model-button ${selectedModel === '7b' ? 'active' : ''}`}
        >
          7B Model
        </button>
      </div>

      <FacetedBeeswarm
        data={data}
        width={900}
        height={400}
        selectedModel={selectedModel.toUpperCase()}
      />
    </div>
  );
};

export default FacetedBeeswarmExample;
