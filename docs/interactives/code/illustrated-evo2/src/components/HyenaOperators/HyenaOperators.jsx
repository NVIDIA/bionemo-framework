import React, { useState } from 'react';
import katexify from '../../utils/katexify';
import './HyenaOperators.css';
import ToeplitzMatrix from '../ToeplitzMatrix/ToeplitzMatrix';

const HyenaOperators = () => {
  // the three hyena operator tabs
  const operators = [
    {
      name: 'Hyena-SE',
      size: 'short',
      title: 'Hyena-SE (Short Explicit)',
      description: `A convolution with short, explicit filters (kernel size4-7). When the filters are short, a simple explicit parametrization is sufficient to achieve convergence. Hyena-SE is key in achieving speedups across a range of input regimes, including short sequences, while still excelling at local, multi-token recall. With a hardware-aware implementation using tensor cores, Hyena-SE achieves the highest throughput of any sequence mixing operator and can also be utilized as a replacement for feed-forward layers.`,
    },
    {
      name: 'Hyena-MR',
      size: 'medium',
      title: 'Hyena-MR (Medium Regularized)',
      description: `A convolution with explicitly parametrized filters of length in the hundreds. 
      While it can be difficult to optimize longer explicit convolutions, this simple exponential-decay regularizer 
      (i.e., ${katexify('h_t = \\hat{h}_t \\lambda^{-\\alpha t}', false)}, where α is swept across channels and ${katexify('\\hat{h}_t', false)} 
      is the learnable parameter) is sufficient for convergence. The effect of this regularization is to concentrate the filter’s influence on local and 
      modestly distant positions by providing a soft bias toward locality, but allow for “medium-range” interactions as needed by modulating the decay.`,
    },
    {
      name: 'Hyena-LI',
      size: 'long',
      title: 'Hyena-LI (Long Implicit)',
      description: `A close relative to the original Hyena Operator design, Hynea-LI is a long implicit convolution that achieves global context at a sub-quadratic cost (see <i>Global Context</i> section below).
      In Hyena-LI, the inner filter is obtained as a linear combination of real exponentials ${katexify('h_t = \\sum_{n=1}^R R_n \\lambda_n^{t-1}', false)}, ${katexify('R_n, \\lambda_n \\in \\mathbb{R}', false)} 
      . The filter global context is able to capture long-range dependencies in the input sequence. Hyena-LI retains the ability to switch to a recurrent parametrization for constant memory.`,
    },
  ];

  const [selectedTab, setSelectedTab] = useState(0);

  return (
    <>
      <div className="tabs">
        {operators.map((op, idx) => (
          <button
            key={op.name}
            className={`tab-button${idx === selectedTab ? ' active' : ''}`}
            onClick={() => setSelectedTab(idx)}
          >
            {op.name}
          </button>
        ))}
      </div>

      <div className="tab-content body-text">
        <ToeplitzMatrix size={operators[selectedTab].size} />
        <b>{operators[selectedTab].title}</b>
        <br />
        <span
          dangerouslySetInnerHTML={{
            __html: operators[selectedTab].description,
          }}
        />
      </div>
      <p className="body-text">
        To better understand how the Hyena operator works, we can trace the path
        of a simple length 5 DNA sequence through the Hyena operator, for a N+1
        = 3 Hyena recurrence, as follows:
      </p>
    </>
  );
};

export default HyenaOperators;
