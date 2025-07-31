import React from 'react';
import katexify from '../../utils/katexify';
import SideNote from '../UI/SideNote/SideNote';
import Hyenafilter from './HyenaFilter';

const HyenaLongContext = () => (
  <>
    <p className="body-text">
      <strong>Global Context</strong>
      <br />
      The Hyena operator captures context over the entire sequence
      <SideNote>
        A receptive field spanning all&nbsp;
        <span dangerouslySetInnerHTML={{ __html: katexify('N', false) }} />
        &nbsp;positions
      </SideNote>
      using <i>implicit convolutions</i>. Instead of an explicit kernel of
      size&nbsp;
      <span dangerouslySetInnerHTML={{ __html: katexify('N', false) }} />, it
      employs a small MLP, called the <i>Hyena Filter</i>, to generate a
      fixed-size kernel from position indices&nbsp;
      <span
        dangerouslySetInnerHTML={{ __html: katexify('1\\ldots N', false) }}
      />
      . This MLP takes in positional encodings, runs them through 3
      sine-activated MLP layers, and outputs a fixed-size kernel:
    </p>
    <div
      style={{
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center',
      }}
    >
      <Hyenafilter />
    </div>
    <p className="body-text">
      This fixed-size kernel approximates the full receptive field (like an
      analytic line equation versus listing every point). Naively convolving
      with an implicit kernel still costs&nbsp;
      <span dangerouslySetInnerHTML={{ __html: katexify('O(N^2)', false) }} />,
      so Hyena leverages the convolution theorem to perform it via the Fast
      Fourier Transform (FFT), reducing the cost to&nbsp;
      <span
        dangerouslySetInnerHTML={{ __html: katexify('O(N \\log N)', false) }}
      />
      . This FFT-based step is known as FFT Convolution (FFTConv), and is what
      allows the Hyena operator to achieve global context at a sub-quadratic
      cost.
    </p>
  </>
);

export default HyenaLongContext;
