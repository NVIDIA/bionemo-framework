import React from 'react';
import SideNote from '../UI/SideNote/SideNote';
import AttentionDependence from './AttentionDependence';

const HyenaDataDependence = () => (
  <>
    <p className="body-text">
      <strong>Data-dependence</strong>
      <br />
      Recall, data dependence refers to the ability of a model to capture
      dependencies between different parts of the input data. Attention achieves
      data-dependence through data-controlled operations: The matrix 𝐴 of the
      operator is formed on the fly, <b>constructed explicitly</b> as a function
      of the input. Hyena operators, on the other hand, never explicitly
      materialize the matrix 𝐴. Instead, they achieve input dependency via{' '}
      <b>element-wise gating</b>.
      <br />
      <br />
      <div
        style={{
          display: 'flex',
          justifyContent: 'center',
          alignItems: 'center',
        }}
      >
        <AttentionDependence />
      </div>
      <br />
      <br />
      <br />
      <br />
      Hyena defines a different data-controlled linear operator given by a
      combination of convolutions and element-wise gating. The resulting
      operator does not compute 𝐴(𝑥) explicitly, but instead models an efficient
      implicit decomposition. Note that, because of the <i>Hyena Recurrence</i>,
      this gating may be applied N times in sequence to increase the number of
      data-dependent gates.{' '}
      <SideNote>
        Earlier studies<sup>1,5</sup> have shown that N+1=3 provides a good
        balance between expressivity and efficiency.
      </SideNote>
    </p>
  </>
);

export default HyenaDataDependence;
