import React, { useEffect, useRef } from 'react'
import * as vg from '@uwdata/vgplot'

const FILL_COLOR = "#76b900"
const BACKGROUND_COLOR = "#e0e0e0"

export default function Histogram({ brush, column, label }) {
  const containerRef = useRef(null)

  useEffect(() => {
    if (!containerRef.current || !brush) return

    // Clear previous content
    containerRef.current.innerHTML = ''

    const width = containerRef.current.clientWidth - 20
    const height = 60
    const marginLeft = 45
    const marginBottom = 20
    const marginRight = 10
    const marginTop = 5

    // Wrapper to position the axis line over the plot
    const wrapper = document.createElement('div')
    wrapper.style.position = 'relative'

    const plot = vg.plot(
      // Background histogram: full data (no filterBy)
      vg.rectY(
        vg.from("features"),
        { x: vg.bin(column), y: vg.count(), fill: BACKGROUND_COLOR, inset: 1 }
      ),
      // Foreground histogram: filtered data
      vg.rectY(
        vg.from("features", { filterBy: brush }),
        { x: vg.bin(column), y: vg.count(), fill: FILL_COLOR, inset: 1 }
      ),
      vg.intervalX({ as: brush }),
      vg.xLabel(null),
      vg.yLabel(null),
      vg.width(width),
      vg.height(height),
      vg.marginLeft(marginLeft),
      vg.marginBottom(marginBottom),
      vg.marginTop(marginTop),
      vg.marginRight(marginRight)
    )

    wrapper.appendChild(plot)

    // Gray x-axis line positioned at the bottom of the plot area
    const axisLine = document.createElement('div')
    axisLine.style.cssText = `
      position: absolute;
      left: ${marginLeft}px;
      right: ${marginRight}px;
      bottom: ${marginBottom}px;
      height: 1px;
      background: #888;
      pointer-events: none;
    `
    wrapper.appendChild(axisLine)

    containerRef.current.appendChild(wrapper)

    return () => {
      if (containerRef.current) {
        containerRef.current.innerHTML = ''
      }
    }
  }, [brush, column, label])

  return (
    <div
      ref={containerRef}
      style={{ width: '100%', minHeight: '90px', marginTop: '4px' }}
    />
  )
}
