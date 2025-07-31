import React, { useState, useEffect, useRef, useMemo } from 'react';
import * as d3 from 'd3';
import './ChromatinScroll.css';

const ChromatinScroll = () => {
  const [scrollProgress, setScrollProgress] = useState(0);
  const [currentStep, setCurrentStep] = useState(0);
  const containerRef = useRef(null);

  // Determine which step we're on based on scroll progress
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

      // Determine current step (0-3 for 4 steps)
      const step = Math.floor(progress * 4);
      setCurrentStep(Math.min(3, step));
    };

    window.addEventListener('scroll', handleScroll);
    handleScroll(); // Initial calculation

    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  // DesignBar component
  const DesignBar = ({
    dnaSequence = 'ATCGCGATTACGAT',
    borzoi = [],
    enformer = [],
    peakPattern = [],
    nodeSize = 14,
    height = 20,
    barColor = '#76B900', // NVIDIA green
    borzoiColor = '#FF6347',
    enformerColor = '#32CD32',
    strokeWidth = 3,
    curveType = d3.curveBasis,
  }) => {
    const dnaValueMap = { A: 2, T: 4, C: 6, G: 8 };

    const data = useMemo(
      () =>
        dnaSequence
          .toUpperCase()
          .split('')
          .map(b => dnaValueMap[b] || 0),
      [dnaSequence]
    );

    const MARGIN = { top: strokeWidth, right: 0, bottom: strokeWidth, left: 0 };
    const boundsHeight = height - MARGIN.top - MARGIN.bottom;
    const barCount = data.length;

    const boundsWidth = nodeSize * barCount;
    const svgWidth = boundsWidth + MARGIN.left + MARGIN.right;

    const xScale = useMemo(
      () =>
        d3
          .scaleBand()
          .domain(d3.range(barCount))
          .range([0, boundsWidth])
          .padding(0),
      [barCount, boundsWidth]
    );

    const yScale = useMemo(
      () => d3.scaleLinear().domain([0, 10]).range([boundsHeight, 0]),
      [boundsHeight]
    );

    const peakPatternPath = useMemo(() => {
      if (!peakPattern.length) return null;
      let path = '';
      let inPeak = false;
      let startX = 0;
      for (let i = 0; i < peakPattern.length; i++) {
        if (peakPattern[i] === 1 && !inPeak) {
          startX = xScale(i);
          inPeak = true;
        } else if (peakPattern[i] === 0 && inPeak) {
          const endX = xScale(i - 1) + xScale.bandwidth();
          path += `M${startX},${yScale(10)} H${endX} V${yScale(0)} H${startX} Z `;
          inPeak = false;
        }
      }
      if (inPeak) {
        const last = peakPattern.length - 1;
        const endX = xScale(last) + xScale.bandwidth();
        path += `M${startX},${yScale(10)} H${endX} V${yScale(0)} H${startX} Z `;
      }
      return path;
    }, [peakPattern, xScale, yScale]);

    const densityPaths = useMemo(() => {
      const makePath = arr =>
        arr.length
          ? d3
              .line()
              .x((_, i) => xScale(i) + xScale.bandwidth() / 2)
              .y(d => yScale(d))
              .curve(curveType)(arr)
          : null;
      return {
        borzoiPath: makePath(borzoi),
        enformerPath: makePath(enformer),
      };
    }, [borzoi, enformer, xScale, yScale, curveType]);

    if (!data.length) return null;

    return (
      <svg width={svgWidth} height={height}>
        <g transform={`translate(${MARGIN.left},${MARGIN.top})`}>
          {data.map((val, i) => (
            <rect
              key={i}
              x={xScale(i)}
              y={yScale(val)}
              width={xScale.bandwidth()}
              height={boundsHeight - yScale(val)}
              fill={barColor}
              opacity={0.9}
              rx={1}
              ry={1}
            />
          ))}

          {densityPaths.borzoiPath && (
            <path
              d={densityPaths.borzoiPath}
              stroke={borzoiColor}
              strokeWidth={2}
              fill="none"
            />
          )}

          {densityPaths.enformerPath && (
            <path
              d={densityPaths.enformerPath}
              stroke={enformerColor}
              strokeWidth={2}
              fill="none"
            />
          )}

          {peakPatternPath && (
            <path
              d={peakPatternPath}
              stroke="black"
              strokeWidth={strokeWidth}
              fill="transparent"
            />
          )}

          <line
            x1={0}
            x2={boundsWidth}
            y1={boundsHeight}
            y2={boundsHeight}
            stroke="#000"
            strokeWidth={strokeWidth}
          />
        </g>
      </svg>
    );
  };

  // ProgressScrolly component with progressive rendering
  const ProgressScrolly = ({ scrollProgress, width = 1200, height = 600 }) => {
    const margin = { top: 5, right: 100, bottom: 20, left: 100 };
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;

    const dnaFontSize = '8px';
    const nodeSize = 11;
    const designBarHeight = 20;
    const designBarSpacing = 5;

    const DNASequence = ({ sequence = 'ATCGA', node }) => {
      const blockWidth = nodeSize;
      const strokeColor = '#003C32'; // Deep fir
      const isHypothesis = node.data.isHypothesis;
      const isActive = node.data.active;

      const dnaBlockWidth = nodeSize * sequence.length;

      return (
        <g transform={`translate(${-dnaBlockWidth / 2}, 0)`}>
          <rect
            x={0}
            y={-nodeSize / 2}
            width={dnaBlockWidth}
            height={nodeSize}
            fill="white"
            stroke={strokeColor}
            strokeWidth={0}
          />
          {sequence.split('').map((base, i) => (
            <g
              key={i}
              transform={`translate(${i * blockWidth + blockWidth / 2}, 0)`}
            >
              <rect
                x={-nodeSize / 2}
                y={-nodeSize / 2}
                width={nodeSize}
                height={nodeSize}
                fill={
                  isHypothesis
                    ? 'var(--inch-worm)'
                    : isActive
                      ? 'var(--pippin)'
                      : '#E5E7EB'
                }
                stroke={
                  isHypothesis ? '#003C32' : isActive ? '#CC5500' : 'black'
                }
                strokeWidth={1.5}
                rx={2}
                ry={2}
              />
              <text
                x={0}
                y={3}
                textAnchor="middle"
                fontSize={dnaFontSize}
                fontWeight="600"
                fill="black"
                style={{ fontFamily: 'system-ui, -apple-system, sans-serif' }}
              >
                {base}
              </text>
            </g>
          ))}
        </g>
      );
    };

    const createHypothesisNode = (
      sequence,
      active,
      children,
      step,
      drawOrder
    ) => ({
      name: sequence,
      sequence,
      isHypothesis: true,
      active,
      visibleFromStep: step,
      drawOrder: drawOrder,
      colorOrder: drawOrder,
      children: children || [],
    });

    const createExtensionNode = (
      nucleotide,
      score,
      active,
      step,
      drawOrder,
      colorOrder
    ) => ({
      name: nucleotide,
      sequence: nucleotide,
      isExtension: true,
      score: score,
      active: active,
      visibleFromStep: step,
      drawOrder: drawOrder,
      colorOrder: colorOrder || drawOrder,
      children: [],
    });

    const createCompleteTreeData = () => {
      let drawOrderCounter = 0;

      const root = {
        name: 'ATCG',
        sequence: 'ATCG',
        isHypothesis: true,
        isFirst: true,
        isPrompt: true,
        active: true,
        visibleFromStep: -1,
        drawOrder: drawOrderCounter++,
        colorOrder: drawOrderCounter,
        children: [],
      };

      // Step 0: Initial extensions - all appear gray first
      const ext0_1 = createExtensionNode(
        'TCAG',
        -0.39,
        true,
        0,
        drawOrderCounter++,
        null
      );
      const ext0_2 = createExtensionNode(
        'ATCG',
        -0.6,
        false,
        0,
        drawOrderCounter++,
        null
      );
      const ext0_3 = createExtensionNode(
        'CGAT',
        -0.45,
        true,
        0,
        drawOrderCounter++,
        null
      );

      // After all level 0 extensions are drawn, color the active ones
      let colorOrderCounter = drawOrderCounter;
      ext0_1.colorOrder = colorOrderCounter++;
      ext0_3.colorOrder = colorOrderCounter++;

      root.children = [ext0_1, ext0_2, ext0_3];

      // Step 1: Hypothesis nodes appear after coloring
      const hyp1 = createHypothesisNode(
        'ATCGTCAG',
        true,
        [],
        1,
        colorOrderCounter++
      );
      const hyp2 = createHypothesisNode(
        'ATCGCGAT',
        true,
        [],
        1,
        colorOrderCounter++
      );

      // All level 1 extension nodes appear gray first
      const ext1_1 = createExtensionNode(
        'ATCA',
        -1.39,
        false,
        1,
        colorOrderCounter++,
        null
      );
      const ext1_2 = createExtensionNode(
        'CGAT',
        -0.4,
        true,
        1,
        colorOrderCounter++,
        null
      );
      const ext1_3 = createExtensionNode(
        'CGCA',
        -1.39,
        false,
        1,
        colorOrderCounter++,
        null
      );
      const ext2_1 = createExtensionNode(
        'ATAT',
        -0.97,
        false,
        1,
        colorOrderCounter++,
        null
      );
      const ext2_2 = createExtensionNode(
        'CGAT',
        -0.97,
        false,
        1,
        colorOrderCounter++,
        null
      );
      const ext2_3 = createExtensionNode(
        'GCTA',
        -0.84,
        true,
        1,
        colorOrderCounter++,
        null
      );

      // After all level 1 extensions are drawn, color the active ones
      let level1ColorStart = colorOrderCounter;
      ext1_2.colorOrder = level1ColorStart++;
      ext2_3.colorOrder = level1ColorStart++;
      colorOrderCounter = level1ColorStart;

      hyp1.children = [ext1_1, ext1_2, ext1_3];
      hyp2.children = [ext2_1, ext2_2, ext2_3];
      root.children[0].children = [hyp1];
      root.children[2].children = [hyp2];

      // Step 2: Hypothesis nodes for level 2
      const hyp3 = createHypothesisNode(
        'ATCGTCAGCGAT',
        true,
        [],
        2,
        colorOrderCounter++
      );
      const hyp4 = createHypothesisNode(
        'ATCGCGATGCTA',
        false,
        [],
        2,
        colorOrderCounter++
      );

      // All level 2 extension nodes appear gray first
      const ext3_1 = createExtensionNode(
        'TGCG',
        -1.1,
        true,
        2,
        colorOrderCounter++,
        null
      );
      const ext3_2 = createExtensionNode(
        'ATCG',
        -0.9,
        false,
        2,
        colorOrderCounter++,
        null
      );
      const ext3_3 = createExtensionNode(
        'ATAT',
        -0.95,
        false,
        2,
        colorOrderCounter++,
        null
      );
      const ext4_1 = createExtensionNode(
        'CGCG',
        -1.5,
        false,
        2,
        colorOrderCounter++,
        null
      );
      const ext4_2 = createExtensionNode(
        'ATCG',
        -1.35,
        true,
        2,
        colorOrderCounter++,
        null
      );
      const ext4_3 = createExtensionNode(
        'GCGC',
        -1.4,
        false,
        2,
        colorOrderCounter++,
        null
      );

      // After all level 2 extensions are drawn, color the active ones
      let level2ColorStart = colorOrderCounter;
      ext3_1.colorOrder = level2ColorStart++;
      ext4_2.colorOrder = level2ColorStart++;
      colorOrderCounter = level2ColorStart;

      hyp3.children = [ext3_1, ext3_2, ext3_3];
      hyp4.children = [ext4_1, ext4_2, ext4_3];
      root.children[0].children[0].children[1].children = [hyp3];
      root.children[2].children[0].children[2].children = [hyp4];

      // Step 3: Hypothesis nodes for level 3
      const hyp5 = createHypothesisNode(
        'ATCGCGATGCTAATCG',
        true,
        [],
        3,
        colorOrderCounter++
      );
      const hyp6 = createHypothesisNode(
        'ATCGTCAGCGATTGCG',
        true,
        [],
        3,
        colorOrderCounter++
      );

      // All level 3 extension nodes appear gray first
      const ext5_1 = createExtensionNode(
        'ATCG',
        -1.6,
        false,
        3,
        colorOrderCounter++,
        null
      );
      const ext5_2 = createExtensionNode(
        'ACTG',
        -1.2,
        false,
        3,
        colorOrderCounter++,
        null
      );
      const ext5_3 = createExtensionNode(
        'ATAT',
        -1.6,
        false,
        3,
        colorOrderCounter++,
        null
      );
      const ext6_1 = createExtensionNode(
        'CGCG',
        -1.1,
        false,
        3,
        colorOrderCounter++,
        null
      );
      const ext6_2 = createExtensionNode(
        'GCTA',
        -1.35,
        false,
        3,
        colorOrderCounter++,
        null
      );
      const ext6_3 = createExtensionNode(
        'TACG',
        -1.25,
        true,
        3,
        colorOrderCounter++,
        null
      );

      // After all level 3 extensions are drawn, color the active ones
      let level3ColorStart = colorOrderCounter;
      ext6_3.colorOrder = level3ColorStart++;
      colorOrderCounter = level3ColorStart;

      hyp5.children = [ext5_1, ext5_2, ext5_3];
      hyp6.children = [ext6_1, ext6_2, ext6_3];
      root.children[2].children[0].children[2].children[0].children[1].children =
        [hyp5];
      root.children[0].children[0].children[1].children[0].children[0].children =
        [hyp6];

      // Final level
      const hypFinal = createHypothesisNode(
        'ATCGTCAGCGATTGCGTACG',
        true,
        [],
        3,
        colorOrderCounter++
      );
      root.children[0].children[0].children[1].children[0].children[0].children[0].children[2].children =
        [hypFinal];

      return { root, totalElements: colorOrderCounter };
    };

    const { root: treeData, totalElements } = useMemo(
      () => createCompleteTreeData(),
      []
    );

    const visibleElements = Math.floor(scrollProgress * totalElements);

    const generateDesignBarData = sequence => {
      const peakPattern = Array.from(sequence).map(b =>
        b === 'C' || b === 'G' ? 1 : 0
      );
      return { peakPattern };
    };

    const { root, links } = useMemo(() => {
      const treeLayout = d3
        .tree()
        .size([innerWidth, innerHeight])
        .separation((a, b) => {
          // More separation for nodes with longer sequences
          const aWidth = a.data.sequence
            ? a.data.sequence.length * nodeSize
            : nodeSize;
          const bWidth = b.data.sequence
            ? b.data.sequence.length * nodeSize
            : nodeSize;
          return (aWidth + bWidth) / (2 * nodeSize) + 0.5;
        });

      const hierarchy = d3.hierarchy(treeData);
      const rootNode = treeLayout(hierarchy);

      // Collect nodes by depth
      const nodesByDepth = {};
      rootNode.descendants().forEach(n => {
        nodesByDepth[n.depth] = (nodesByDepth[n.depth] || []).concat(n);
      });

      // Calculate Y positions for each depth level
      const depthY = {};
      let cumY = 50; // Start position
      const nodeHeight = nodeSize;

      for (
        let d = 0;
        d <= Math.max(...Object.keys(nodesByDepth).map(Number));
        d++
      ) {
        depthY[d] = cumY;

        // Calculate spacing based on whether this level has hypothesis nodes
        const hasHypothesis =
          nodesByDepth[d] && nodesByDepth[d].some(n => n.data.isHypothesis);
        const baseSpacing = d % 2 === 0 ? 60 : 40;
        const extraSpacing = hasHypothesis
          ? designBarHeight + designBarSpacing
          : 0;
        const totalSpacing = baseSpacing + extraSpacing;

        cumY += nodeHeight + totalSpacing;
      }

      // Apply Y positions
      rootNode.descendants().forEach(n => {
        n.y = depthY[n.depth];
      });

      const linksWithOrder = rootNode.links().map(link => ({
        ...link,
        drawOrder: link.target.data.drawOrder - 0.5,
        colorOrder: link.target.data.colorOrder - 0.5,
      }));

      return { root: rootNode, links: linksWithOrder };
    }, [
      treeData,
      innerWidth,
      innerHeight,
      nodeSize,
      designBarHeight,
      designBarSpacing,
    ]);

    const renderLinkPath = link => {
      const nodeH = nodeSize;
      const sx = link.source.x;
      const sy = link.source.y + nodeH / 2;
      const tx = link.target.x;

      // Adjust target y based on whether target has a DesignBar
      const hasDesignBar = link.target.data.isHypothesis;
      const tyOffset = hasDesignBar
        ? nodeH / 2 + designBarHeight + designBarSpacing
        : nodeH / 2;
      const ty = link.target.y - tyOffset;

      const my = (sy + ty) / 2;

      return `
        M ${sx} ${sy}
        L ${sx} ${my}
        L ${tx} ${my}
        L ${tx} ${ty}
      `;
    };

    const renderNode = node => {
      const isVisible = node.data.drawOrder <= visibleElements;
      const isColored = node.data.colorOrder <= visibleElements;
      const opacity = isVisible ? 1 : 0;

      if (!node.data.isHypothesis) {
        // Extension nodes - show gray first, then color if active
        const isActive = node.data.active && isColored;
        const modifiedNode = {
          ...node,
          data: {
            ...node.data,
            active: isActive,
          },
        };

        return (
          <g
            key={`node-${node.depth}-${node.data.name}-${node.x}-${node.y}`}
            transform={`translate(${node.x},${node.y})`}
            style={{
              opacity,
              transition: 'opacity 0.3s ease',
            }}
          >
            <DNASequence
              sequence={node.data.sequence || node.data.name}
              node={modifiedNode}
            />
          </g>
        );
      }

      // Hypothesis nodes with DesignBar
      const designBarData = generateDesignBarData(node.data.sequence);
      const designBarWidth = nodeSize * node.data.sequence.length;

      return (
        <g
          key={`node-${node.depth}-${node.data.name}-${node.x}-${node.y}`}
          transform={`translate(${node.x},${node.y})`}
          style={{
            opacity,
            transition: 'opacity 0.3s ease',
          }}
        >
          {/* DesignBar positioned above the DNA sequence */}
          <g
            transform={`translate(${-designBarWidth / 2}, ${-designBarHeight - designBarSpacing - nodeSize / 2})`}
          >
            <DesignBar
              dnaSequence={node.data.sequence}
              borzoi={[node.data.sequence]}
              enformer={[]}
              peakPattern={designBarData.peakPattern}
              nodeSize={nodeSize}
              height={designBarHeight}
            />
          </g>
          {/* DNA sequence node */}
          <DNASequence sequence={node.data.sequence} node={node} />
        </g>
      );
    };

    return (
      <svg
        width={width}
        height={height}
        style={{
          fontFamily: 'system-ui, -apple-system, sans-serif',
          border: '0',
        }}
      >
        <defs>
          <marker
            id="arrow"
            viewBox="0 -5 10 10"
            refX={8}
            refY={0}
            markerWidth={5}
            markerHeight={5}
            orient="auto"
          >
            <path d="M0,-5L10,0L0,5" fill="#374151" />
          </marker>
        </defs>

        <g transform={`translate(${margin.left},${margin.top})`}>
          <text
            x={0}
            y={-20}
            textAnchor="start"
            fontSize="12px"
            fontWeight="100"
            fill="#6b7280"
          >
            Using beam search to generate DNA sequences with a beam size of two.
          </text>

          {links.map((link, i) => {
            const isVisible = link.drawOrder <= visibleElements;
            return (
              <g
                key={`link-${i}`}
                style={{
                  opacity: isVisible ? 1 : 0,
                  transition: 'opacity 0.3s ease',
                }}
              >
                <path
                  d={renderLinkPath(link)}
                  stroke="#9CA3AF"
                  strokeWidth={1}
                  fill="none"
                  markerEnd="url(#arrow)"
                />
              </g>
            );
          })}

          {root.descendants().map(renderNode)}
        </g>
      </svg>
    );
  };

  // Step descriptions
  const stepDescriptions = [
    {
      number: '',
      title: 'Generate candidate sequences',
      description:
        'We start with a genomic prompt—a short DNA sequence that serves as our starting point. Next, we generate multiple possible 4 base-pair extensions (the original experiment used 128-base-pair extensions) of this prompt.',
    },
    {
      number: '',
      title: 'Predict biological effects',
      description:
        'Each generated DNA chunk gets evaluated by two specialized models: Enformer and Borzoi. These models predict how each sequence would affect chromatin accessibility—essentially, how "open" or "closed" different regions of the genome would be if this sequence were present. We pick the two sequences scores best matching the target pattern.',
    },
    {
      number: '',
      title: 'Score and select the best candidates',
      description:
        'This process continues. We compare each predicted accessibility profile against the current target pattern in at the corresonding stage in the sequence.',
    },
    {
      number: '',
      title: 'Iterating with beam search',
      description:
        'Iterating through the generation process in this manner allows us to maintain several promising sequences simultaneously. The highest-scoring sequences are appended to our growing DNA construct, and the process repeats. This approach explores multiple promising paths while focusing computational resources on the most successful candidates.',
    },
    {
      number: '',
      title: 'Final sequence',
      description:
        'After several iterations, we end up with a DNA sequence that is highly likely to produce the desired biological effect. This sequence is then used to create a new cell, which can be used to study the biological effect in more detail.',
    },
  ];

  return (
    <div style={{ fontFamily: 'NVIDIA Sans' }}>
      <section>
        <div className="progress-section-container" ref={containerRef}>
          <div className="progress-content-container">
            {stepDescriptions.map((step, index) => (
              <div
                key={index}
                className={`progress-step ${currentStep >= index ? 'active' : ''}`}
              >
                <div className="progress-step-content">
                  <div className="progress-step-number">{step.number}</div>
                  <p>
                    <b>{step.title}</b>
                  </p>
                  <p>{step.description}</p>
                </div>
              </div>
            ))}
          </div>

          <div className="progress-charts-container">
            <ProgressScrolly
              scrollProgress={scrollProgress}
              width={900}
              height={700}
            />
          </div>
        </div>
      </section>
      <br />
      <br />
      <p className="body-text">
        This generation strategy was even extended to generate DNA sequences
        with chromatin accessibility patterns that encode Morse code messages..
        While more fun than practical, the experiment shows the ability to guide
        genome generation according to some specific functional goal.
      </p>
    </div>
  );
};

export default ChromatinScroll;
